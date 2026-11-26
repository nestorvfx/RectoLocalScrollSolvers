"""tifxyz I/O: a surface as a grid of 3-D points (x.tif, y.tif, z.tif, float32; invalid cells -1) + meta.json.

Positions are in the voxel frame of the volume the mesh is registered to (`<id>-on-<volume>-<pitch>um.tifxyz`),
x / y / z order as written by VC3D. `scale` in meta.json is the grid step relative to that volume's voxels (0.05 =
one grid cell per 20 voxels)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read(d: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> xyz [H, W, 3] float64, valid [H, W] bool, meta dict."""
    import tifffile
    d = Path(d)
    x, y, z = (tifffile.imread(d / f"{a}.tif").astype(np.float64) for a in "xyz")
    v = (x > 0) & (y > 0) & (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return np.stack([x, y, z], -1), v, meta


def write(d: str | Path, xyz: np.ndarray, valid: np.ndarray, meta: dict | None = None, overwrite: bool = False) -> None:
    import tifffile
    d = Path(d)
    if d.exists() and any(d.iterdir()) and not overwrite:
        raise FileExistsError(f"{d} exists: every mesh version gets a new folder")
    d.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate("xyz"):
        tifffile.imwrite(d / f"{a}.tif", np.where(valid, xyz[..., i], -1).astype(np.float32))
    meta = dict(meta or {})
    meta.setdefault("format", "tifxyz")
    meta.setdefault("type", "seg")
    pv = xyz[valid]
    if len(pv):
        meta["bbox"] = [pv.min(0).tolist(), pv.max(0).tolist()]
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def quad_area(xyz: np.ndarray, valid: np.ndarray) -> float:
    """Surface area of the valid quads, in squared voxels."""
    a, b, c, e = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    t = 0.5 * (np.linalg.norm(np.cross(b - a, c - a), axis=-1) + np.linalg.norm(np.cross(b - e, c - e), axis=-1))
    return float(np.where(q, t, 0).sum())
