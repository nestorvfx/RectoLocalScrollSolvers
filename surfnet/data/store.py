"""Band store of one object: the CT sampled along the mesh normals, kept tile-major on disk.

    <root>/meta.json   object id, grid shape, tiles, band spec, processing level / pitch, volume, provenance
    <root>/geom.npz    per cell: P (xyz, float32, processing-level voxels), V (valid), N (unit normals toward the
                       scroll axis), tier (uint8), split (uint8: 0 train, 1 held-out, 2 excluded), winding (int16, -1
                       unknown), optional nbr (float16 [2K, H, W]: offset along N to windings -K..-1, 1..K, NaN unknown),
                       optional neg (bool: user-rejected cells, verify negatives)
    <root>/fine.u8     memmap [nty, ntx, Df, 3T, 3T]   depths -fine_half..fine_half (step fine_step), 3 px per cell
    <root>/coarse.u8   memmap [nty, ntx, Dc, T, T]     depths -coarse_half..coarse_half (step coarse_step), from level+1
    <root>/done.npy    bool [nty, ntx] tiles written (resumable builds)

Cell r of the grid sits at fine pixel 3r + 1 (the centre of its 3 x 3 block) and at coarse pixel r.
Region reads assemble tiles; outside the grid or unwritten tiles read as 0."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SPLIT_TRAIN, SPLIT_HELDOUT, SPLIT_EXCLUDED = 0, 1, 2


@dataclass
class BandSpec:
    fine_half: int = 32
    fine_step: float = 1.0
    ppc: int = 3
    coarse_half: int = 232
    coarse_step: float = 2.0
    tile: int = 32

    @property
    def Df(self) -> int:
        return 2 * self.fine_half + 1

    @property
    def Dc(self) -> int:
        return 2 * int(round(self.coarse_half / self.coarse_step)) + 1

    def fine_depths(self) -> np.ndarray:
        return np.arange(-self.fine_half, self.fine_half + 1) * self.fine_step

    def coarse_depths(self) -> np.ndarray:
        n = int(round(self.coarse_half / self.coarse_step))
        return np.arange(-n, n + 1) * self.coarse_step

    @classmethod
    def from_dict(cls, d: dict) -> "BandSpec":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class BandStore:
    def __init__(self, root: str | Path, mode: str = "r"):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.spec = BandSpec.from_dict(self.meta["spec"])
        self.H, self.W = self.meta["shape"]
        self.nty, self.ntx = self.meta["tiles"]
        T, p = self.spec.tile, self.spec.ppc
        self.fine = np.memmap(self.root / "fine.u8", np.uint8, mode, shape=(self.nty, self.ntx, self.spec.Df, p * T, p * T))
        self.coarse = np.memmap(self.root / "coarse.u8", np.uint8, mode, shape=(self.nty, self.ntx, self.spec.Dc, T, T))
        self._geom = None
        self.done = np.load(self.root / "done.npy")

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, root: str | Path, geom: dict, spec: BandSpec, meta: dict) -> "BandStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        V = geom["V"]
        H, W = V.shape
        T, p = spec.tile, spec.ppc
        nty, ntx = (H + T - 1) // T, (W + T - 1) // T
        m = dict(meta)
        m.update(shape=[H, W], tiles=[nty, ntx], spec=asdict(spec))
        (root / "meta.json").write_text(json.dumps(m, indent=1))
        np.savez(root / "geom.npz", **geom)
        np.memmap(root / "fine.u8", np.uint8, "w+", shape=(nty, ntx, spec.Df, p * T, p * T)).flush()
        np.memmap(root / "coarse.u8", np.uint8, "w+", shape=(nty, ntx, spec.Dc, T, T)).flush()
        np.save(root / "done.npy", np.zeros((nty, ntx), bool))
        return cls(root, "r+")

    @staticmethod
    def exists(root: str | Path) -> bool:
        root = Path(root)
        return all((root / f).exists() for f in ("meta.json", "geom.npz", "fine.u8", "coarse.u8", "done.npy"))

    # ------------------------------------------------------------------ geometry
    def update_geom(self, **arrays) -> None:
        """Add / replace per-cell arrays in geom.npz (e.g. nbr after the neighbour search)."""
        g = dict(self.geom)
        g.update(arrays)
        tmp = self.root / "geom.tmp.npz"
        np.savez(tmp, **g)
        tmp.replace(self.root / "geom.npz")
        self._geom = g

    def update_meta(self, **kv) -> None:
        self.meta.update(kv)
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @property
    def geom(self) -> dict:
        if self._geom is None:
            with np.load(self.root / "geom.npz") as z:
                self._geom = {k: z[k] for k in z.files}
        return self._geom

    def tile_ok(self) -> np.ndarray:
        """Tiles holding any valid cell."""
        T = self.spec.tile
        V = self.geom["V"]
        ok = np.zeros((self.nty, self.ntx), bool)
        for ty in range(self.nty):
            for tx in range(self.ntx):
                ok[ty, tx] = V[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].any()
        return ok

    # ------------------------------------------------------------------ tiles
    def write_tile(self, ty: int, tx: int, fine: np.ndarray, coarse: np.ndarray) -> None:
        self.fine[ty, tx] = fine
        self.coarse[ty, tx] = coarse
        self.done[ty, tx] = True

    def flush(self) -> None:
        self.fine.flush()
        self.coarse.flush()
        np.save(self.root / "done.npy", self.done)

    def read_region(self, r0: int, c0: int, R: int, which: tuple[str, ...] = ("fine", "coarse")) -> dict:
        """Cells [r0, r0 + R) x [c0, c0 + R) -> {'fine': [Df, 3R, 3R], 'coarse': [Dc, R, R]} uint8 (0 outside)."""
        T, p = self.spec.tile, self.spec.ppc
        out = {}
        if "fine" in which:
            out["fine"] = np.zeros((self.spec.Df, p * R, p * R), np.uint8)
        if "coarse" in which:
            out["coarse"] = np.zeros((self.spec.Dc, R, R), np.uint8)
        for ty in range(max(0, r0 // T), min(self.nty, (r0 + R + T - 1) // T)):
            for tx in range(max(0, c0 // T), min(self.ntx, (c0 + R + T - 1) // T)):
                if not self.done[ty, tx]:
                    continue
                # overlap in cells
                a0, a1 = max(r0, ty * T), min(r0 + R, (ty + 1) * T)
                b0, b1 = max(c0, tx * T), min(c0 + R, (tx + 1) * T)
                if a0 >= a1 or b0 >= b1:
                    continue
                if "fine" in out:
                    out["fine"][:, p * (a0 - r0):p * (a1 - r0), p * (b0 - c0):p * (b1 - c0)] = \
                        self.fine[ty, tx, :, p * (a0 - ty * T):p * (a1 - ty * T), p * (b0 - tx * T):p * (b1 - tx * T)]
                if "coarse" in out:
                    out["coarse"][:, a0 - r0:a1 - r0, b0 - c0:b1 - c0] = \
                        self.coarse[ty, tx, :, a0 - ty * T:a1 - ty * T, b0 - tx * T:b1 - tx * T]
        return out

    def cells(self, key: str, r0: int, c0: int, R: int, fill=0) -> np.ndarray:
        """A per-cell geometry array cropped to the region (padded with `fill`)."""
        a = self.geom[key]
        vec = key in ("P", "N")
        if vec:
            a = np.moveaxis(a, -1, 0)                       # [3, H, W]
        out = np.full(a.shape[:-2] + (R, R), fill, dtype=a.dtype)
        ra, rb = max(r0, 0), min(r0 + R, a.shape[-2])
        ca, cb = max(c0, 0), min(c0 + R, a.shape[-1])
        if ra < rb and ca < cb:
            out[..., ra - r0:rb - r0, ca - c0:cb - c0] = a[..., ra:rb, ca:cb]
        return np.moveaxis(out, 0, -1) if vec else out
