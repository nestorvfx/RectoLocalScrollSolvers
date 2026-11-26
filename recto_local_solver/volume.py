"""THE CT VOLUME: chunk store, GPU chunk cache, trilinear sampling, the grid's normals, the stack along the normals.

The volume is the level 0 of an OME-Zarr (uint8, 128^3 raw chunks, '/' separator: every Vesuvius Challenge scroll
volume in the open-data bucket): an s3:// URL, read with s5cmd into a local chunk cache (an absent chunk = an empty
file), or a local copy of it (the same chunk files, read in place). The chunks are then held in a GPU cache and sampled
there. configure() is called once, before any solve: the volume (its shape is read from .zarray), the scroll axis (the
normals point toward it), the cache folder.

Normals: the cell grid's masked-Gaussian-smoothed positions (sigma 2 cells), cross product of the two grid derivatives,
one global sign toward the scroll axis."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import kernels as KN

DEV = os.environ.get("RECTO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
S5 = os.environ.get("RECTO_S5CMD", "s5cmd")
CT_S3 = None            # the volume's level 0 (configure; None: a local directory, CT_CACHE itself)
SHAPE = None
CT_CACHE = None
AXIS = None
GPU_CACHE_GB = float(os.environ["RECTO_GPU_CACHE_GB"]) if os.environ.get("RECTO_GPU_CACHE_GB") else None   # None: _cache_gb


def configure(url, cache, axis, gpu_cache_gb=None):
    """the volume (level 0: an s3:// URL, or a local directory read in place), the chunk cache folder (s3 only), the
    scroll axis (a json file or dict {z, y, x}: the axis centre per height, level-0 voxels). Reads the shape from the
    volume's .zarray and checks its format."""
    global CT_S3, SHAPE, CT_CACHE, AXIS, GPU_CACHE_GB
    url = str(url).rstrip("/")
    if url.startswith("s3://"):
        r = subprocess.run([S5, "--no-sign-request", "cat", f"{url}/.zarray"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"cannot read {url}/.zarray: {r.stderr.strip()[-300:]}")
        za = json.loads(r.stdout)
    else:
        if not (Path(url) / ".zarray").exists():
            raise FileNotFoundError(f"{url}/.zarray: --volume is an s3:// URL or a local OME-Zarr level directory")
        za = json.loads((Path(url) / ".zarray").read_text())
    if (tuple(za["chunks"]) != (128, 128, 128) or za["dtype"] not in ("|u1", "uint8") or za.get("compressor")
            or za.get("dimension_separator", ".") != "/"):
        raise ValueError(f"{url}: expected uint8, 128^3 raw chunks, '/' separator; got {za}")
    SHAPE = tuple(int(s) for s in za["shape"])
    CT_S3, CT_CACHE = (url, Path(cache)) if url.startswith("s3://") else (None, Path(url))
    AXIS = json.loads(Path(axis).read_text()) if isinstance(axis, (str, Path)) else dict(axis)
    if gpu_cache_gb is not None:
        GPU_CACHE_GB = float(gpu_cache_gb)
    _VOLS.clear()


# ------------------------------------------------------------------------------------------------ chunk store
_LISTINGS = {}
_FETCH_LOCK = threading.Lock()
_PRESENT = set()                                   # chunks known to be in the disk cache (stored or empty)


class Store:
    """the CT's chunks in the local cache, fetched by s5cmd (each z-slab's listing made once and kept); for a local
    volume, its own chunk files (nothing fetched)."""

    def __init__(self):
        self.C = 128
        self.cache = CT_CACHE
        self.s3 = CT_S3
        if self.s3:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.shape = SHAPE
        self.n = tuple((s + self.C - 1) // self.C for s in self.shape)
        self._ls = _LISTINGS.setdefault(str(self.cache), {})      # (shared by every Store of this cache)

    def path(self, c):
        z, y, x = (int(t) for t in c)
        return self.cache / str(z) / str(y) / str(x)

    def stored_keys(self, cz):
        """the chunks stored in z-slab cz -> {"z/y/x": size}: listed once (kept in memory and as _ls_<z>.json)."""
        if cz not in self._ls:
            f = self.cache / f"_ls_{cz}.json"
            if f.exists():
                self._ls[cz] = json.loads(f.read_text())
            else:
                r = subprocess.run([S5, "--no-sign-request", "--json", "ls", f"{self.s3}/{cz}/*"], capture_output=True, text=True)
                d = {}
                for line in r.stdout.splitlines():
                    dd = json.loads(line)
                    if dd.get("error"):                    # an empty slab lists "no object found"; anything else (a
                        if "no object found" in dd["error"]:   # wrong region, no network) would mark every chunk absent
                            continue
                        raise RuntimeError(f"s5cmd ls of slab {cz} failed: {dd['error'][:200]}")
                    if dd.get("type") != "directory":
                        d[dd["key"][len(self.s3) + 1:]] = int(dd.get("size", 0))
                if d:
                    f.write_text(json.dumps(d))
                self._ls[cz] = d
        return self._ls[cz]

    def fetch(self, chunks):
        if not self.s3:                            # a local volume: absent chunk files read as empty
            return 0
        todo = [c for c in (tuple(int(t) for t in c) for c in chunks) if c not in _PRESENT and not self._here(c)]
        if not todo:                               # (chunks on disk are checked once per process, outside the lock)
            return 0
        with _FETCH_LOCK:                          # (one transfer at a time: a background prefetch and a solve's own
            return self._fetch(todo)               # fetch never download the same chunk twice)

    def _here(self, c):
        if self.path(c).exists():
            _PRESENT.add(c)
            return True
        return False

    def _fetch(self, chunks):
        todo = [c for c in chunks if not self._here(c)]
        if not todo:
            return 0
        stored = {}
        for cz in sorted({c[0] for c in todo}):
            stored.update(self.stored_keys(cz))
        get = [c for c in todo if "/".join(map(str, c)) in stored]
        for c in get:
            (self.cache / str(c[0]) / str(c[1])).mkdir(parents=True, exist_ok=True)
        for attempt in range(3):                                 # a failed transfer is tried twice more
            if not get:
                break
            if attempt:
                time.sleep(5 * attempt)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write("\n".join(f'cp "{self.s3}/{"/".join(map(str, c))}" "{self.cache}/{c[0]}/{c[1]}/{c[2]}.part"' for c in get))
                cmds = f.name
            subprocess.run([S5, "--no-sign-request", "--numworkers", "64", "run", cmds], capture_output=True, text=True)
            os.unlink(cmds)
            bad = []
            for c in get:
                part = self.cache / str(c[0]) / str(c[1]) / f"{c[2]}.part"
                if part.exists() and part.stat().st_size == stored["/".join(map(str, c))]:
                    os.replace(part, self.path(c))
                elif not self.path(c).exists():                  # (not placed meanwhile by another process)
                    bad.append(c)
            get = bad
        if get:
            raise RuntimeError(f"ct: {len(get)} chunks failed after 3 attempts (first {get[:3]})")
        for c in todo:
            if "/".join(map(str, c)) not in stored:
                q = self.cache / str(c[0]) / str(c[1]) / str(c[2])
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(b"")
        return len(todo)

    def read(self, c, out):
        p = self.path(c)
        if not p.exists() or p.stat().st_size == 0:
            return False
        out[...] = np.frombuffer(p.read_bytes(), np.uint8).reshape(out.shape)
        return True


def _cache_gb():
    """the GPU chunk cache's cap (GB): as configured, else 8 GB and at most 40 % of the GPU's memory (the solve itself
    takes up to ~3.5 GB more on the benchmark's largest areas)."""
    if GPU_CACHE_GB is not None:
        return GPU_CACHE_GB
    if DEV.startswith("cuda") and torch.cuda.is_available():
        return min(8.0, 0.4 * torch.cuda.get_device_properties(torch.device(DEV)).total_memory / 2 ** 30)
    return 8.0


class GPUVolume:
    """Persistent GPU cache of the CT (per process): chunks are read from the disk cache ONCE, stored in a growing GPU
    slab [slots, C, C, C] (uint8) with a slot table over the chunk grid; every later sample is a pure GPU gather. The
    slab is capped at GPU_CACHE_GB: a batch that would go over it resets the cache first."""

    def __init__(self):
        self.st = Store()
        self.C = self.st.C
        self.table = torch.full(self.st.n, -1, dtype=torch.int32, device=DEV)
        self.known = torch.zeros(self.st.n, dtype=torch.bool, device=DEV)     # looked up (stored or not)
        self.vol = torch.zeros((0, self.C, self.C, self.C), dtype=torch.uint8, device=DEV)
        self.used = 0
        self.shape = torch.as_tensor(self.st.shape, device=DEV)

    def ensure(self, pts_zyx, margin=1):
        """make every chunk touched by the points' trilinear stencil resident. Fast paths: every chunk of the points'
        bounding box already looked up (six reductions), or every chunk of their stencils (eight gathers); else the
        full lookup (_ensure_all)."""
        q = pts_zyx.reshape(-1, 3)
        C = self.C
        nch = torch.as_tensor(self.st.n, device=q.device)
        fl = torch.floor(q).long()
        bl = torch.minimum(torch.div(fl.amin(0) - margin, C, rounding_mode="floor").clamp(min=0), nch - 1).tolist()
        bh = torch.minimum(torch.div(fl.amax(0) + 1 + margin, C, rounding_mode="floor").clamp(min=0), nch - 1).tolist()
        if bool(self.known[bl[0]:bh[0] + 1, bl[1]:bh[1] + 1, bl[2]:bh[2] + 1].all()):
            return
        lo = torch.minimum(torch.div(fl - margin, C, rounding_mode="floor").clamp(min=0), nch - 1)
        hi = torch.minimum(torch.div(fl + 1 + margin, C, rounding_mode="floor").clamp(min=0), nch - 1)
        del fl
        allk = True
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    if not bool(self.known[(lo, hi)[i][:, 0], (lo, hi)[j][:, 1], (lo, hi)[k][:, 2]].all()):
                        allk = False
                        break
                if not allk:
                    break
            if not allk:
                break
        del lo, hi
        if allk:
            return
        return self._ensure_all(pts_zyx, margin)

    def _ensure_all(self, pts_zyx, margin=1):
        q = pts_zyx.reshape(-1, 3)
        C = self.C
        nch = torch.as_tensor(self.st.n, device=q.device)
        lo = torch.minimum(torch.div(torch.floor(q).long() - margin, C, rounding_mode="floor").clamp(min=0), nch - 1)
        hi = torch.minimum(torch.div(torch.floor(q).long() + 1 + margin, C, rounding_mode="floor").clamp(min=0), nch - 1)
        ny, nx = int(nch[1]), int(nch[2])                  # the 8 stencil corners' chunks as one integer code each:
        code = torch.cat([((lo, hi)[i][:, 0] * ny + (lo, hi)[j][:, 1]) * nx + (lo, hi)[k][:, 2]   # a 1-D unique (the
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)])                           # same keys, in the
        code = torch.unique(code)                                                                     # same order as the
        keys = torch.stack([code // (ny * nx), (code // nx) % ny, code % nx], 1)                      # rows' unique)
        del code, lo, hi
        self._load(keys)

    def ensure_samples(self, pz, nz, ks):
        """make resident exactly the chunks that the samples pz[i] + ks[k] nz[i] (zyx, float64 [n, 3]) read: their
        trilinear stencils' corners, by the sampler's own arithmetic (kernels.chunk_mark; GPU only)."""
        flags = KN.run("chunk_mark", lambda: KN.chunk_mark(pz, nz, ks, self.table, self.st.shape, self.C),
                       lambda: self._mark_torch(pz, nz, ks), 0.0)
        keys = flags.nonzero()
        if len(keys) and not bool(self.known[keys[:, 0], keys[:, 1], keys[:, 2]].all()):
            self._load(keys)

    def _mark_torch(self, pz, nz, ks):
        """chunk_mark in PyTorch (its check): the corners inside the volume of every sample's stencil -> chunk flags."""
        flags = torch.zeros(tuple(self.table.shape), dtype=torch.uint8, device=pz.device)
        kt = torch.as_tensor(np.asarray(ks, np.float64), device=pz.device)
        for a0 in range(0, len(kt), 8):
            b = torch.floor(pz.double()[None] + kt[a0:a0 + 8, None, None] * nz.double()[None]).long().reshape(-1, 3)
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        c = b + torch.tensor((dz, dy, dx), device=pz.device)
                        c = c[((c >= 0) & (c < self.shape)).all(-1)] // self.C
                        flags[c[:, 0], c[:, 1], c[:, 2]] = 1
        return flags

    def _load(self, keys):
        """the chunks `keys` [m, 3] resident (fetched to the disk cache, read once, copied into the GPU slab)."""
        C = self.C
        new = keys[~self.known[keys[:, 0], keys[:, 1], keys[:, 2]]]
        budget = int(_cache_gb() * 2 ** 30) // C ** 3
        if self.used + len(new) > budget and self.used:  # over the budget: drop the resident chunks, reload this batch's
            self.table.fill_(-1); self.known.zero_(); self.used = 0
            self.vol = torch.zeros((0, C, C, C), dtype=torch.uint8, device=DEV)
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()
            new = keys
        keys = new
        if len(keys) == 0:
            return
        kn = keys.cpu().numpy()
        self.st.fetch(kn)
        host = np.zeros((len(kn), C, C, C), np.uint8)
        stored = np.array([self.st.read(tuple(k), host[i]) for i, k in enumerate(kn)], bool)
        sk = keys[torch.as_tensor(stored, device=DEV)]
        if stored.any():
            need = self.used + int(stored.sum())
            if need > len(self.vol):                       # grow the slab (doubling)
                cap = max(need, min(2 * len(self.vol), budget), 64)
                nv = torch.zeros((cap, C, C, C), dtype=torch.uint8, device=DEV)
                nv[:self.used] = self.vol[:self.used]
                self.vol = nv
            self.vol[self.used:need] = torch.from_numpy(host[stored]).to(DEV)
            self.table[sk[:, 0], sk[:, 1], sk[:, 2]] = torch.arange(self.used, need, dtype=torch.int32, device=DEV)
            self.used = need
        self.known[keys[:, 0], keys[:, 1], keys[:, 2]] = True

    def sample(self, pts):
        """trilinear values at level-0 points (zyx, float tensor [..., 3]); chunks must be resident (ensure). Fast path
        (bitwise identical): a point whose 2 x 2 x 2 stencil lies inside ONE resident chunk is read from the flat chunk
        store at fixed offsets; the other points through the general gather (_sample_all)."""
        shp = pts.shape[:-1]
        q = pts.reshape(-1, 3)
        base = torch.floor(q)
        frac = (q - base).float()
        base = base.long()
        C = self.C
        nz, ny, nx = self.table.shape
        ci = torch.div(base, C, rounding_mode="floor")
        li = base - ci * C
        inr = (ci[:, 0] >= 0) & (ci[:, 0] < nz) & (ci[:, 1] >= 0) & (ci[:, 1] < ny) & (ci[:, 2] >= 0) & (ci[:, 2] < nx)
        inr &= ((base >= 0) & (base + 1 < self.shape)).all(-1)
        slot = self.table[ci[:, 0].clamp(0, nz - 1), ci[:, 1].clamp(0, ny - 1), ci[:, 2].clamp(0, nx - 1)]
        fast = inr & (li < C - 1).all(-1) & (slot >= 0)
        flat = self.vol.view(-1)
        idx = ((slot.clamp(min=0).long() * C + li[:, 0].clamp(0, C - 1)) * C + li[:, 1].clamp(0, C - 1)) * C + li[:, 2].clamp(0, C - 1)
        idx = torch.where(fast, idx, torch.zeros_like(idx))
        del ci, li, slot, base
        out = torch.zeros(q.shape[0], dtype=torch.float32, device=q.device)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    vv = flat[idx + (dz * C * C + dy * C + dx)].float()
                    w = (frac[:, 0] if dz else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]) * (frac[:, 2] if dx else 1 - frac[:, 2])
                    out += torch.where(fast, w * vv, torch.zeros_like(vv))
        slow = ~fast
        if bool(slow.any()):
            si = slow.nonzero()[:, 0]
            out[si] = self._sample_all(q[si])
        return out.reshape(shp)

    def _sample_all(self, pts):
        base = torch.floor(pts)
        frac = (pts - base).float()
        base = base.long()
        out = torch.zeros(pts.shape[:-1], dtype=torch.float32, device=DEV)
        C = self.C
        nz, ny, nx = self.table.shape
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    c = base + torch.tensor((dz, dy, dx), device=DEV)
                    inb = ((c >= 0) & (c < self.shape)).all(-1)
                    ci = torch.div(c, C, rounding_mode="floor")
                    slot = self.table[ci[..., 0].clamp(0, nz - 1), ci[..., 1].clamp(0, ny - 1), ci[..., 2].clamp(0, nx - 1)]
                    ok = inb & (slot >= 0)
                    li = (c - ci * C).clamp(0, C - 1)
                    vv = self.vol[slot.clamp(min=0).long(), li[..., 0], li[..., 1], li[..., 2]].float()
                    w = (frac[..., 0] if dz else 1 - frac[..., 0]) * (frac[..., 1] if dy else 1 - frac[..., 1]) \
                        * (frac[..., 2] if dx else 1 - frac[..., 2])
                    out += torch.where(ok, w * vv, torch.zeros_like(vv))
        return out


_VOLS = {}


def volume():
    """the process-wide GPU chunk cache of the configured CT."""
    if "ct" not in _VOLS:
        _VOLS["ct"] = GPUVolume()
    return _VOLS["ct"]


class Sampler:
    """trilinear sampling on the GPU (zyx level-0 points) through the process-wide GPU chunk cache; the chunks of
    pts_zyx made resident first."""

    def __init__(self, pts_zyx, margin=1):
        self.v = volume()
        self.v.ensure(pts_zyx, margin)

    def __call__(self, pts):
        return self.v.sample(pts)


def interp_axis(z, key):
    return np.interp(z, AXIS["z"], AXIS[key])


def normals(P, v, sigma=2.0):
    """unit normals [H, W, 3] (xyz) toward the scroll axis, and where defined: the masked Gaussian smoothing of the
    positions (sigma cells, zero padding), the cross product of the two grid derivatives (np.gradient's central /
    one-sided differences), one global sign. float64, on the device (numpy in / out)."""
    import torch.nn.functional as F
    xyz = np.asarray(P, np.float64); vt = np.asarray(v, bool)
    r = int(4 * sigma + 0.5)
    g = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float64) / sigma) ** 2); g = g / g.sum()
    vv = vt.astype(np.float64)
    X = torch.as_tensor(np.concatenate([np.moveaxis(xyz * vv[..., None], -1, 0), vv[None]], 0), device=DEV)   # [4, H, W]
    gk = torch.as_tensor(g, device=DEV)
    sm = lambda T: (F.pad((F.pad(T, (r, r)).unfold(1, 2 * r + 1, 1) @ gk).t(), (r, r)).unfold(1, 2 * r + 1, 1) @ gk).t()
    X = torch.stack([sm(X[c]) for c in range(4)])                     # along the columns, then the rows; zero padding
    f = X[:3] / X[3].clamp(min=1e-6)                                  # [3, H, W] xyz (window sums as matrix products)
    def grad(T, ax):                                                  # np.gradient (unit spacing, first-order edges)
        d = torch.empty_like(T); n = T.shape[ax]
        d.narrow(ax, 1, n - 2).copy_((T.narrow(ax, 2, n - 2) - T.narrow(ax, 0, n - 2)) / 2.0)
        d.narrow(ax, 0, 1).copy_(T.narrow(ax, 1, 1) - T.narrow(ax, 0, 1))
        d.narrow(ax, n - 1, 1).copy_(T.narrow(ax, n - 1, 1) - T.narrow(ax, n - 2, 1))
        return d
    du, dv = grad(f, 1), grad(f, 2)
    n = torch.stack([du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]], -1)
    nn = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    n = (n / nn.clamp(min=1e-9)).cpu().numpy(); nn = nn.cpu().numpy()
    ok = vt & (nn[..., 0] > 1e-9)
    z = xyz[..., 2]
    inward = np.stack([interp_axis(z, "x") - xyz[..., 0], interp_axis(z, "y") - xyz[..., 1], np.zeros_like(z)], -1)
    s = (n * inward).sum(-1)
    sign = 1.0 if float((s[ok] > 0).mean()) >= 0.5 else -1.0                      # one global sign
    return n * sign, ok


# ------------------------------------------------------------------------------------------------ the stack
def prefetch(P, v, N, klo, khi):
    """every CT chunk a stack of these cells can touch, fetched in ONE call (P [.., 3] xyz, offsets klo..khi vox)."""
    pts = P[v][:, [2, 1, 0]]; nn = N[v][:, [2, 1, 0]]
    if not len(pts):
        return 0
    st = Store(); n = np.asarray(st.n, np.int64)
    u = _prefetch_keys(pts, nn, klo, khi, n)
    c = np.stack([u // ((n[1] + 2) * (n[2] + 2)), (u // (n[2] + 2)) % (n[1] + 2), u % (n[2] + 2)], 1) - 1
    c = c[((c >= 0) & (c < n)).all(1)]
    return st.fetch([tuple(int(t) for t in r) for r in c])


def _prefetch_keys(pts, nn, klo, khi, n):
    """prefetch's chunks as sorted unique integer keys (offset by 1: >= 0): the chunks of the 8 corners of +-4 vox around
    every point at 9 depths klo - 1 .. khi + 1 (float64 on the device, as the NumPy form)."""
    p = torch.as_tensor(np.asarray(pts, np.float64), device=DEV); m = torch.as_tensor(np.asarray(nn, np.float64), device=DEV)
    corners = torch.tensor([[dz, dy, dx] for dz in (-4.0, 4.0) for dy in (-4.0, 4.0) for dx in (-4.0, 4.0)], dtype=torch.float64, device=DEV)
    keys = []
    for k in np.linspace(klo - 1.0, khi + 1.0, 9):
        c = torch.floor((p + float(k) * m)[:, None, :].add(corners[None]) / 128.0).long().reshape(-1, 3) + 1
        keys.append(torch.unique((c[:, 0] * int(n[1] + 2) + c[:, 1]) * int(n[2] + 2) + c[:, 2]))
    return torch.unique(torch.cat(keys)).cpu().numpy()


def prefetch_async(fn):
    """fn() (which fetches chunks) in a background thread, while the GPU works: the downloads overlap the solve. Only
    files of the disk cache are written, so the results do not depend on it; a failure is logged and the solve's own
    fetch does the work. Returns the thread."""
    def run():
        try:
            fn()
        except Exception as e:
            print(f"ct prefetch: {type(e).__name__}: {str(e)[:200]} (the solve fetches what it needs)", file=sys.stderr, flush=True)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def render_layers(P, v, N, offsets, up=5, rows_per_band=256, kgroup=24, as_torch=False):
    """the flattened render of the surface P (the grid bilinearly upsampled `up` px per cell) at EVERY normal offset,
    as one stack [len(offsets), Ho, Wo] float32 + the valid mask, in one CT sampling pass per row band and group of
    kgroup offsets (numpy; as_torch: the GPU tensors)."""
    H, Wd = v.shape
    Ho, Wo = int((H - 1) * up) + 1, int((Wd - 1) * up) + 1
    K = len(offsets)
    img_t = torch.zeros((K, Ho, Wo), dtype=torch.float32, device=DEV); ok_t = torch.zeros((Ho, Wo), dtype=torch.bool, device=DEV)
    Pt = torch.as_tensor(np.where(v[..., None], P, 0), device=DEV); Nt = torch.as_tensor(np.where(v[..., None], N, 0), device=DEV)
    vt = torch.as_tensor(v, device=DEV)
    cols = torch.arange(Wo, device=DEV, dtype=torch.float64) / up
    j0 = cols.floor().long().clamp(0, Wd - 2); b = (cols - j0)[None, :, None]
    offs = torch.as_tensor(np.asarray(offsets, np.float64), device=DEV)
    for y0 in range(0, Ho, rows_per_band):
        y1 = min(Ho, y0 + rows_per_band)
        rows = torch.arange(y0, y1, device=DEV, dtype=torch.float64) / up
        i0 = rows.floor().long().clamp(0, H - 2); a = (rows - i0)[:, None, None]
        I0, J0 = i0[:, None], j0[None, :]
        okb = vt[I0, J0] & vt[I0 + 1, J0] & vt[I0, J0 + 1] & vt[I0 + 1, J0 + 1]
        if not bool(okb.any()):
            continue
        p = (1 - a) * (1 - b) * Pt[I0, J0] + a * (1 - b) * Pt[I0 + 1, J0] + (1 - a) * b * Pt[I0, J0 + 1] + a * b * Pt[I0 + 1, J0 + 1]
        n = (1 - a) * (1 - b) * Nt[I0, J0] + a * (1 - b) * Nt[I0 + 1, J0] + (1 - a) * b * Nt[I0, J0 + 1] + a * b * Nt[I0 + 1, J0 + 1]
        n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
        pp, nn = p[okb][:, [2, 1, 0]], n[okb][:, [2, 1, 0]]
        idx = okb.reshape(-1).nonzero()[:, 0]
        for k0 in range(0, K, kgroup):
            ok_ = offs[k0:k0 + kgroup]
            pts = lambda ok_=ok_: (pp[None] + ok_[:, None, None] * nn[None]).reshape(-1, 3)
            if pp.is_cuda and KN.available():                      # the kernel reads the chunk cache directly (the same
                vol = volume()                                      # values as GPUVolume.sample), the chunks its samples
                ko = ok_.cpu().numpy()                              # read made resident first
                vol.ensure_samples(pp, nn, ko)

                def fast(ko=ko):
                    o = torch.zeros((len(ko), (y1 - y0) * Wo), dtype=torch.float32, device=DEV)
                    return KN.slab_sample(vol.vol, vol.table, vol.st.shape, pp.contiguous(), nn.contiguous(), ko, o, idx)[:, idx]
                vals = KN.run("render_sample", fast, lambda pts=pts: Sampler(pts())(pts()).reshape(len(ok_), -1), 0.0)
            else:
                vals = Sampler(pts())(pts()).reshape(len(ok_), -1)
            img_t[k0:k0 + len(ok_), y0:y1][:, okb] = vals
        ok_t[y0:y1] = okb
    return (img_t, ok_t) if as_torch else (img_t.cpu().numpy(), ok_t.cpu().numpy())


def stack(P, v, N, region, klo=-16.0, khi=14.0, step=0.5, up=5, pad_rows=0, as_torch=False):
    """the stack of a region (r0, r1, c0, c1) of cells: the CT on the surface moved along its normals by klo..khi vox
    (`step`), `up` px per cell (1 px ~ 1 vox): S [D, Y, X] float32, ok [Y, X], offs [D] (as_torch: S, ok on the GPU)."""
    r0, r1, c0, c1 = region
    ra, rb = max(r0 - pad_rows, 0), min(r1 + pad_rows, v.shape[0])
    offs = np.arange(klo, khi + 1e-6, step)
    prefetch(P[ra:rb, c0:c1], v[ra:rb, c0:c1], N[ra:rb, c0:c1], klo, khi)
    S, ok = render_layers(P[ra:rb, c0:c1], v[ra:rb, c0:c1], N[ra:rb, c0:c1], offs, up=up, as_torch=as_torch)
    return S, ok, offs
