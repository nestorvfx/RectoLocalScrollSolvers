"""A synthetic rolled scroll for end-to-end tests: an Archimedean spiral of papyrus sheets around a vertical axis.

Each sheet is a bright band (thickness `thickness` vox) with an asymmetric texture: the recto half (toward the axis)
carries fibres running around the scroll (intensity varies along z), the verso half fibres running along z (intensity
varies along the arc), so recto and verso are distinguishable as in real papyrus. Gaps are dark; Gaussian noise, a
slow intensity drift and a few bright inclusions are added. Sheet n's recto face lies at
    r_n(theta) = r0 + spacing * (n + theta / 2 pi) - thickness / 2.
Meshes: one grid per winding on the recto face (rows along z, columns along the arc, ~cell_vox voxels per cell),
written as objects with exact neighbour-winding offsets (along the inward normal, winding n + k lies at about
-k * spacing)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surfnet.data.store import SPLIT_HELDOUT, SPLIT_TRAIN, BandSpec, BandStore


@dataclass
class PhantomSpec:
    shape: tuple = (128, 208, 208)
    r0: float = 16.0
    spacing: float = 12.0
    thickness: float = 4.0
    turns: int = 6
    fibre_period: float = 5.0
    gap: float = 55.0
    sheet: float = 150.0
    texture: float = 30.0
    noise: float = 8.0
    inclusions: int = 12
    cell_vox: float = 5.06943
    seed: int = 0

    @property
    def centre(self) -> tuple[float, float]:
        return (self.shape[1] - 1) / 2.0, (self.shape[2] - 1) / 2.0


def make_volume(spec: PhantomSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    Z, Y, X = spec.shape
    cy, cx = spec.centre
    z = np.arange(Z, dtype=np.float32)[:, None, None]
    y = np.arange(Y, dtype=np.float32)[None, :, None] - cy
    x = np.arange(X, dtype=np.float32)[None, None, :] - cx
    r = np.sqrt(x * x + y * y)
    th = np.mod(np.arctan2(y, x), 2 * np.pi)
    s = (r - spec.r0) / spec.spacing - th / (2 * np.pi)
    n = np.round(s)
    delta = (s - n) * spec.spacing                                              # radial offset from the sheet centre
    inside = 1.0 / (1.0 + np.exp(-(spec.thickness / 2 - np.abs(delta)) / 0.6))
    inside = inside * ((n >= 0) & (n < spec.turns))
    phase = rng.uniform(0, 2 * np.pi, size=spec.turns + 2)
    ph = phase[np.clip(n, 0, spec.turns + 1).astype(int)]
    rect = np.cos(2 * np.pi * z / spec.fibre_period + ph)                       # recto: fibres around the scroll
    vers = np.cos(2 * np.pi * th * r / spec.fibre_period + ph)                  # verso: fibres along z
    tex = np.where(delta < 0, rect, vers)
    vol = spec.gap + inside * (spec.sheet - spec.gap + spec.texture * tex)
    vol = vol * (1.0 + 0.1 * np.sin(2 * np.pi * z / Z) * np.cos(th))            # slow drift
    vol = vol + rng.normal(0, spec.noise, size=vol.shape).astype(np.float32)
    for _ in range(spec.inclusions):
        c = rng.uniform([4, 4, 4], [Z - 4, Y - 4, X - 4])
        rad = rng.uniform(1.0, 2.5)
        d2 = (z - c[0]) ** 2 + (y + cy - c[1]) ** 2 + (x + cx - c[2]) ** 2
        vol = vol + 90.0 * np.exp(-d2 / (2 * rad * rad))
    return np.clip(np.rint(vol), 0, 255).astype(np.uint8)


def recto_radius(spec: PhantomSpec, n: int, th: np.ndarray) -> np.ndarray:
    return spec.r0 + spec.spacing * (n + th / (2 * np.pi)) - spec.thickness / 2


def winding_mesh(spec: PhantomSpec, n: int, margin_z: float = 6.0, th_margin: float = 0.15
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Recto-face grid of winding n: P [H, W, 3] xyz, V [H, W]; one cell every cell_vox along z and along the arc (as a
    VC3D mesh: the same step everywhere)."""
    Z = spec.shape[0]
    cy, cx = spec.centre
    zs = np.arange(margin_z, Z - 1 - margin_z, spec.cell_vox)
    tf = np.linspace(th_margin, 2 * np.pi - th_margin, 20001)
    ds = np.hypot(recto_radius(spec, n, tf), spec.spacing / (2 * np.pi))          # arc length per radian
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(tf))])
    ths = np.interp(np.arange(0.0, s[-1], spec.cell_vox), s, tf)
    TH, ZZ = np.meshgrid(ths, zs)
    R = recto_radius(spec, n, TH)
    P = np.stack([cx + R * np.cos(TH), cy + R * np.sin(TH), ZZ], -1)
    return P.astype(np.float64), np.ones(TH.shape, bool)


def axis_json(spec: PhantomSpec) -> dict:
    cy, cx = spec.centre
    return {"z": [0.0, float(spec.shape[0])], "y": [cy, cy], "x": [cx, cx]}


def build_objects(spec: PhantomSpec, root: str | Path, band: BandSpec | None = None, device: str = "cpu",
                  windings: list[int] | None = None, heldout: tuple[int, ...] = (), K: int = 3,
                  vol: np.ndarray | None = None) -> list[BandStore]:
    """Phantom volume -> one BandStore per winding under root/objects/phantom_w<n> (bands built from the volume)."""
    import torch

    from surfnet.common.geometry import Axis, as_t, grid_normals, orient_to_axis
    from surfnet.data.bands import BandBuilder
    from surfnet.volume import ArraySource
    root = Path(root)
    band = band or BandSpec(fine_half=32, coarse_half=96, tile=16)
    vol = make_volume(spec) if vol is None else vol
    src = ArraySource.from_volume(vol, n_levels=2)
    ax = axis_json(spec)
    axis = Axis(ax["z"], ax["y"], ax["x"])
    stores = []
    for n in windings if windings is not None else range(1, spec.turns - 1):
        P, V = winding_mesh(spec, n)
        Pt, Vt = as_t(P, device, torch.float64), as_t(V, device, torch.bool)
        N, ok = grid_normals(Pt, Vt)
        N = orient_to_axis(N, Pt, ok, axis).cpu().numpy()
        H, W = V.shape
        # neighbour offsets along the inward normal: winding n + k is k turns outward
        nbr = np.full((2 * K, H, W), np.nan, np.float16)
        ks = list(range(-K, 0)) + list(range(1, K + 1))
        for i, k in enumerate(ks):
            if 0 <= n + k < spec.turns:
                nbr[i] = -k * spec.spacing
        geom = dict(P=P.astype(np.float32), V=ok.cpu().numpy() & V, N=N.astype(np.float32),
                    tier=np.full((H, W), 2, np.uint8),
                    split=np.full((H, W), SPLIT_HELDOUT if n in heldout else SPLIT_TRAIN, np.uint8),
                    winding=np.full((H, W), n, np.int16), nbr=nbr)
        meta = dict(id=f"phantom_w{n:02d}", scroll="phantom", volume="phantom", level=0, pitch_um=9.362,
                    cell_um=spec.cell_vox * 9.362, winding_index_offsets=ks)
        stores.append(BandStore.create(root / "objects" / meta["id"], geom, band, meta))
    BandBuilder(src, 0, device=device, fine_capacity=256, coarse_capacity=64, group=8).build(stores)
    return stores
