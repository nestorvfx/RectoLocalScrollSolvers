"""Volume access: chunk sources + a GPU chunk cache with trilinear sampling.

Sources
- `S3Source`: an official OME-Zarr v2 volume in the open-data bucket (uint8, 128^3 chunks, unsharded; one object per
  chunk at `<level>/<cz>/<cy>/<cx>`; absent chunk = outside the mask = 0). Chunks are fetched with s5cmd in batches:
  list the chunk row -> exact keys -> `s5cmd run` -> size check against the listing -> atomic rename. Never one HTTP
  request per chunk. Absent chunks get an empty marker file. Uncompressed chunks are read with np.frombuffer;
  blosc-compressed volumes (PHerc0172) are decoded with numcodecs.
- `LocalZarrSource`: a local OME-Zarr v2 volume (uint8, cubic chunks, '/' or '.' chunk-key separator; absent chunk = 0),
  e.g. a volume downloaded for VC3D;
- `ArraySource`: an in-memory pyramid (tests).
- `open_volume(spec, cache)`: an s3:// URL -> `S3Source`, a local path -> `LocalZarrSource`.

`GPUChunkCache` keeps chunks of ONE level resident on the GPU in a slot array [slots, C, C, C] (uint8) with a slot
table over the chunk grid. Sampling is a pure GPU gather of the 8 trilinear corners (no host round trips). Slots are
evicted least-recently-used when the cache is full; pinned keys (the current work set) are never evicted.

Coordinates are zyx in voxels of the cached level. Level k+1 voxel i covers level-k voxels 2i, 2i+1: a level-k
coordinate q maps to (q - 0.5) / 2 at level k+1 (`to_coarser`)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

CHUNK = 128


def to_coarser(q: torch.Tensor, steps: int = 1) -> torch.Tensor:
    for _ in range(steps):
        q = (q - 0.5) / 2.0
    return q


# ------------------------------------------------------------------------------------------------ sources
class ChunkSource:
    chunk: int = CHUNK

    def shape(self, level: int) -> tuple[int, int, int]:
        raise NotImplementedError

    def grid(self, level: int) -> tuple[int, int, int]:
        return tuple((s + self.chunk - 1) // self.chunk for s in self.shape(level))  # type: ignore[return-value]

    def fetch(self, level: int, keys: np.ndarray) -> None:
        """Make the chunks locally readable (download). Keys [N, 3] (cz, cy, cx)."""

    def read(self, level: int, key: tuple[int, int, int]) -> np.ndarray | None:
        raise NotImplementedError

    def release(self, level: int, keys: np.ndarray) -> None:
        """The chunks are no longer needed locally (transient downloads may be deleted)."""


class ArraySource(ChunkSource):
    """levels: list of uint8 arrays (zyx), level k = 2^k downsampled."""

    def __init__(self, levels: list[np.ndarray], chunk: int = CHUNK):
        self.levels = levels
        self.chunk = chunk

    @classmethod
    def from_volume(cls, vol: np.ndarray, n_levels: int = 2, chunk: int = CHUNK) -> "ArraySource":
        levels = [vol.astype(np.uint8)]
        for _ in range(1, n_levels):
            a = levels[-1].astype(np.float32)
            z, y, x = (s // 2 * 2 for s in a.shape)
            a = a[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean((1, 3, 5))
            levels.append(np.clip(np.rint(a), 0, 255).astype(np.uint8))
        return cls(levels, chunk)

    def shape(self, level):
        return self.levels[level].shape

    def read(self, level, key):
        C = self.chunk
        a = self.levels[level]
        z, y, x = (k * C for k in key)
        blk = a[z:z + C, y:y + C, x:x + C]
        if blk.size == 0 or not blk.any():
            return None
        out = np.zeros((C, C, C), np.uint8)
        out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
        return out


class S3Source(ChunkSource):
    """An OME-Zarr v2 volume in the bucket, cached under cache_dir/<level>/<cz>/<cy>/<cx>."""

    def __init__(self, url: str, cache_dir: str | Path, s5cmd: str = "s5cmd", workers: int = 64,
                 transient: bool = False):
        self.url = url.rstrip("/")
        self.cache = Path(cache_dir)
        self.s5 = s5cmd
        self.workers = workers
        self.transient = transient
        self._meta: dict[int, dict] = {}
        self._listed: dict[tuple[int, int], dict[str, int]] = {}
        self._codec = None
        self._lock = threading.Lock()

    def _zarray(self, level: int) -> dict:
        if level not in self._meta:
            p = self.cache / str(level) / ".zarray"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([self.s5, "--no-sign-request", "cat", f"{self.url}/{level}/.zarray"],
                                   capture_output=True, text=True, check=True)
                p.write_text(r.stdout)
            m = json.loads(p.read_text())
            if list(m["chunks"]) != [self.chunk] * 3 or m["dtype"] not in ("|u1", "uint8"):
                raise ValueError(f"{self.url}/{level}: unexpected array {m['chunks']} {m['dtype']}")
            self._meta[level] = m
        return self._meta[level]

    def shape(self, level):
        return tuple(self._zarray(level)["shape"])

    def path(self, level, key) -> Path:
        return self.cache / str(level) / str(key[0]) / str(key[1]) / str(key[2])

    def _list_row(self, level: int, cz: int) -> dict[str, int]:
        k = (level, cz)
        if k not in self._listed:
            r = subprocess.run([self.s5, "--no-sign-request", "--json", "ls", f"{self.url}/{level}/{cz}/*"],
                               capture_output=True, text=True)
            out = {}
            pre = f"{self.url}/{level}/"
            for line in r.stdout.splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "directory":
                    continue
                key = d["key"][len(pre):] if d["key"].startswith(pre) else d["key"].split(f"/{level}/", 1)[-1]
                out[key] = int(d.get("size", 0))
            self._listed[k] = out
        return self._listed[k]

    def fetch(self, level, keys):
        self._zarray(level)
        todo = [tuple(int(t) for t in k) for k in np.asarray(keys).reshape(-1, 3) if not self.path(level, k).exists()]
        if not todo:
            return
        with self._lock:
            stored: dict[str, int] = {}
            rows = sorted({k[0] for k in todo})
            new = [cz for cz in rows if (level, cz) not in self._listed]
            if len(new) > 1:                       # the rows not listed yet, listed side by side
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(min(16, len(new))) as ex:
                    list(ex.map(lambda cz: self._list_row(level, cz), new))
            for cz in rows:
                stored.update(self._list_row(level, cz))
            get = [k for k in todo if "/".join(map(str, k)) in stored]
            for k in get:
                self.path(level, k).parent.mkdir(parents=True, exist_ok=True)
            if get:
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                    f.write("\n".join(f'cp "{self.url}/{level}/{k[0]}/{k[1]}/{k[2]}" "{self.path(level, k)}.part"'
                                      for k in get))
                    cmds = f.name
                subprocess.run([self.s5, "--no-sign-request", "--numworkers", str(self.workers), "run", cmds],
                               capture_output=True, text=True)
                os.unlink(cmds)
            bad = []
            for k in get:
                part = Path(f"{self.path(level, k)}.part")
                if part.exists() and part.stat().st_size == stored["/".join(map(str, k))]:
                    os.replace(part, self.path(level, k))
                else:
                    bad.append(k)
            if bad:
                raise RuntimeError(f"{self.url} level {level}: {len(bad)} chunks failed the size check (first {bad[:3]})")
            for k in todo:
                if "/".join(map(str, k)) not in stored:
                    p = self.path(level, k)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"")

    def read(self, level, key):
        p = self.path(level, key)
        if not p.exists():
            self.fetch(level, np.asarray([key]))
        raw = p.read_bytes()
        if not raw:
            return None
        m = self._zarray(level)
        if m.get("compressor"):
            if self._codec is None:
                import numcodecs
                self._codec = numcodecs.get_codec(m["compressor"])
            raw = self._codec.decode(raw)
        return np.frombuffer(raw, np.uint8).reshape(self.chunk, self.chunk, self.chunk)

    def release(self, level, keys):
        if not self.transient:
            return
        for k in np.asarray(keys).reshape(-1, 3):
            p = self.path(level, k)
            if p.exists():
                p.unlink()

    def disk_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.cache.rglob("*") if f.is_file())

    def clear(self) -> None:
        shutil.rmtree(self.cache, ignore_errors=True)



class LocalZarrSource(ChunkSource):
    """A local OME-Zarr v2 volume: <root>/<level>/.zarray and one file per chunk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._meta: dict[int, dict] = {}
        self._codec: dict[int, object] = {}
        m = self._zarray(0)
        self.chunk = int(m["chunks"][0])

    def _zarray(self, level: int) -> dict:
        if level not in self._meta:
            p = self.root / str(level) / ".zarray"
            if not p.exists():
                raise FileNotFoundError(f"{p}: not an OME-Zarr v2 level (zarr v3 volumes are not supported)")
            m = json.loads(p.read_text())
            if len(set(m["chunks"])) != 1 or m["dtype"] not in ("|u1", "uint8"):
                raise ValueError(f"{p}: needs uint8 with cubic chunks, got {m['chunks']} {m['dtype']}")
            if level > 0 and int(m["chunks"][0]) != self.chunk:
                raise ValueError(f"{p}: chunk {m['chunks'][0]} differs from level 0 ({self.chunk})")
            self._meta[level] = m
        return self._meta[level]

    def shape(self, level):
        return tuple(self._zarray(level)["shape"])

    def path(self, level, key) -> Path:
        sep = self._zarray(level).get("dimension_separator", ".")
        name = sep.join(str(int(k)) for k in key)
        return self.root / str(level) / name

    def read(self, level, key):
        p = self.path(level, key)
        if not p.exists():
            return None
        raw = p.read_bytes()
        m = self._zarray(level)
        if m.get("compressor"):
            if level not in self._codec:
                import numcodecs
                self._codec[level] = numcodecs.get_codec(m["compressor"])
            raw = self._codec[level].decode(raw)
        C = self.chunk
        a = np.frombuffer(raw, np.uint8).reshape(C, C, C)
        return a if m.get("order", "C") == "C" else np.ascontiguousarray(a.reshape(C, C, C, order="F"))


def open_volume(spec: str, cache: str | Path | None = None, **kw) -> ChunkSource:
    """An s3:// URL (cached under `cache`) or a local OME-Zarr directory."""
    if str(spec).startswith("s3://"):
        if cache is None:
            raise ValueError("an S3 volume needs a cache directory")
        return S3Source(str(spec), cache, **kw)
    return LocalZarrSource(spec)


# ------------------------------------------------------------------------------------------------ GPU cache
class GPUChunkCache:
    """Resident chunks of one level on the device; trilinear sampling by gather."""

    def __init__(self, source: ChunkSource, level: int, device="cuda", capacity: int = 2048):
        self.src = source
        self.level = level
        self.device = torch.device(device)
        self.C = source.chunk
        self.shape = source.shape(level)
        self.grid = source.grid(level)
        self.capacity = capacity
        self.table = torch.full(self.grid, -1, dtype=torch.int32, device=self.device)   # slot, -1 = not resident
        self.absent = torch.zeros(self.grid, dtype=torch.bool, device=self.device)      # known to be all-zero
        self.slots = torch.zeros((capacity, self.C, self.C, self.C), dtype=torch.uint8, device=self.device)
        self.lru: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        self.free = list(range(capacity - 1, -1, -1))
        self._shape_t = torch.as_tensor(self.shape, device=self.device)

    # -------------------------------------------------------------------- planning
    def keys_for(self, pts: torch.Tensor, margin: float = 1.0) -> np.ndarray:
        """Unique chunk keys [N, 3] touched by the trilinear stencils of the points (zyx, this level), clipped to the
        grid."""
        q = pts.reshape(-1, 3)
        if q.numel() == 0:
            return np.zeros((0, 3), np.int64)
        C = self.C
        g = torch.as_tensor(self.grid, device=q.device)
        lo = torch.div(torch.floor(q - margin).long(), C, rounding_mode="floor")
        hi = torch.div(torch.floor(q + 1 + margin).long(), C, rounding_mode="floor")
        lo = torch.minimum(lo.clamp(min=0), g - 1)
        hi = torch.minimum(hi.clamp(min=0), g - 1)
        # each (cz, cy, cx) packed into one int64 (20 bits per axis; the same order as rows): a 1-D unique, about 50 times
        # faster than torch.unique(dim=0) on the [8N, 3] rows, with the same keys in the same order
        B = 1 << 20
        lin = torch.cat([((lo, hi)[i][:, 0] * B + (lo, hi)[j][:, 1]) * B + (lo, hi)[k][:, 2]
                         for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        u = torch.unique(lin)
        return torch.stack([u // (B * B), (u // B) % B, u % B], 1).cpu().numpy()

    # -------------------------------------------------------------------- residency
    def ensure(self, keys: np.ndarray, pin: set | None = None) -> None:
        """Load the missing chunks (reading from the source; the caller fetched them). `pin`: keys never evicted now."""
        keys = [tuple(int(t) for t in k) for k in np.asarray(keys).reshape(-1, 3)]
        pin = set(pin or ()) | set(keys)
        need = []
        for k in keys:
            if k in self.lru:
                self.lru.move_to_end(k)
            elif not bool(self.absent[k]):
                need.append(k)
        if not need:
            return
        host = []
        got = []
        for k in need:
            a = self.src.read(self.level, k)
            if a is None:
                self.absent[k] = True
            else:
                host.append(a)
                got.append(k)
        if not got:
            return
        while len(self.free) < len(got):
            victim = next((v for v in self.lru if v not in pin), None)
            if victim is None:
                raise RuntimeError(f"GPU chunk cache too small: {len(got)} new chunks, capacity {self.capacity}")
            s = self.lru.pop(victim)
            self.table[victim] = -1
            self.free.append(s)
        slots = [self.free.pop() for _ in got]
        arr = torch.from_numpy(np.stack(host))
        if self.device.type == "cuda":
            arr = arr.pin_memory()
        idx = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        self.slots[idx] = arr.to(self.device, non_blocking=True)
        kt = torch.as_tensor(got, device=self.device, dtype=torch.long)
        self.table[kt[:, 0], kt[:, 1], kt[:, 2]] = torch.as_tensor(slots, device=self.device, dtype=torch.int32)
        for k, s in zip(got, slots):
            self.lru[k] = s

    def drop(self, keys: np.ndarray) -> None:
        for k in np.asarray(keys).reshape(-1, 3):
            k = tuple(int(t) for t in k)
            s = self.lru.pop(k, None)
            if s is not None:
                self.table[k] = -1
                self.free.append(s)

    # -------------------------------------------------------------------- sampling
    def sample(self, pts: torch.Tensor) -> torch.Tensor:
        """Trilinear values (float32) at zyx points [..., 3] of this level. Non-resident / outside = 0."""
        shp = pts.shape[:-1]
        q = pts.reshape(-1, 3).to(self.device)
        base = torch.floor(q)
        frac = (q - base).float()
        base = base.long()
        C = self.C
        nz, ny, nx = self.grid
        # per axis, once: for the corner at +0 and at +1, its weight, whether it lies in the volume, its chunk index and
        # its index inside the chunk; then per corner one gather from the slot table and one from the slots, on flat
        # indices. The weights and the sum are the same float operations in the same order as the corner-by-corner
        # loop this replaces (bit-identical, tests/test_cpu.py), with about a third of the memory traffic.
        axes = []
        for a in range(3):
            b = base[:, a]
            per = []
            for d, w in ((0, 1 - frac[:, a]), (1, frac[:, a])):
                c = b + d
                ci = torch.div(c, C, rounding_mode="floor")
                per.append((w, (c >= 0) & (c < self.shape[a]), ci.clamp(0, self.grid[a] - 1), (c - ci * C).clamp(0, C - 1)))
            axes.append(per)
        table, vox = self.table.reshape(-1), self.slots.reshape(-1)
        out = torch.zeros(q.shape[0], dtype=torch.float32, device=self.device)
        for dz in (0, 1):
            wz, iz, cz, lz = axes[0][dz]
            for dy in (0, 1):
                wy, iy, cy, ly = axes[1][dy]
                for dx in (0, 1):
                    wx, ix, cx, lx = axes[2][dx]
                    slot = table[(cz * ny + cy) * nx + cx]
                    ok = iz & iy & ix & (slot >= 0)
                    v = vox[((slot.clamp(min=0).long() * C + lz) * C + ly) * C + lx].float()
                    out += torch.where(ok, wz * wy * wx * v, torch.zeros_like(v))
        return out.reshape(shp)


class LevelOffset(ChunkSource):
    """Level k of this source = level `offset` + k of `src`: lets the model's level-0 / level-1 caches read a
    2.4 um scan at its levels 2 / 3 (the ~9.4 um the model was trained at)."""

    def __init__(self, src: ChunkSource, offset: int):
        self.src, self.offset, self.chunk = src, int(offset), src.chunk

    def shape(self, level):
        return self.src.shape(level + self.offset)

    def fetch(self, level, keys):
        self.src.fetch(level + self.offset, keys)

    def read(self, level, key):
        return self.src.read(level + self.offset, key)

    def release(self, level, keys):
        self.src.release(level + self.offset, keys)
