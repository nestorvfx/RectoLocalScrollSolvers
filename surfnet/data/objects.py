"""Mesh -> object: geometry on the common cell grid + per-cell maps, written as a BandStore skeleton (bands are built
later by `bands.py`).

- Positions are converted to the processing level of the volume (`level_coords`: level-k voxel i covers level-0
  voxels 2^k i .. 2^k i + 2^k - 1, so q_k = (q_0 - (2^k - 1) / 2) / 2^k; checked against the official pyramid on the
  box by `surfnet bands --check-levels`).
- The grid is resampled (Catmull-Rom where the 4 x 4 stencil is valid, bilinear otherwise) so that one cell is
  `cell_um` (47.46 um, ~5 voxels at 9.4 um) along both grid directions.
- Normals: masked-Gaussian-smoothed grid (sigma 2 cells), oriented toward the scroll axis (recto).
- Maps (tier, split, winding id, rejected cells) follow the resampling by nearest neighbour."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..common.geometry import Axis, as_t, grid_normals, grid_steps, orient_to_axis, resample_grid
from .store import BandSpec, BandStore


def level_coords(P_l0: np.ndarray, level: int) -> np.ndarray:
    f = 2 ** level
    return (P_l0 - (f - 1) / 2.0) / f


def _nearest_map(M: np.ndarray, H: int, W: int, sr: float, sc: float, fill=0) -> np.ndarray:
    r = np.clip(np.rint(np.arange(H) * sr).astype(int), 0, M.shape[-2] - 1)
    c = np.clip(np.rint(np.arange(W) * sc).astype(int), 0, M.shape[-1] - 1)
    return M[..., r[:, None], c[None, :]]


def build_object(root: str | Path, oid: str, P: np.ndarray, V: np.ndarray, *, axis: Axis | None, cell_vox: float,
                 spec: BandSpec, meta: dict, tier=2, split=0, winding=-1, neg: np.ndarray | None = None,
                 device="cuda", resample: bool = True, overwrite: bool = False, known: np.ndarray | None = None
                 ) -> BandStore:
    """P [H, W, 3] xyz at the processing level, V [H, W]. tier / split / winding: scalar or [H, W] map. The grid step is
    measured on the `known` cells (`geometry.grid_steps`; every valid cell if None: a ground-truth mesh)."""
    out = Path(root) / "objects" / oid
    if BandStore.exists(out) and not overwrite:
        return BandStore(out, "r+")
    H0, W0 = V.shape
    maps = {}
    for name, val, dt in (("tier", tier, np.uint8), ("split", split, np.uint8), ("winding", winding, np.int16)):
        maps[name] = np.broadcast_to(np.asarray(val, dt), (H0, W0)).copy()
    if neg is not None:
        maps["neg"] = neg.astype(bool)
    Pt, Vt = as_t(P, device, torch.float64), as_t(V, device, torch.bool)
    dr, dc = grid_steps(P, V, V if known is None else known)
    sr, sc = cell_vox / dr, cell_vox / dc
    if resample and (abs(sr - 1) > 0.02 or abs(sc - 1) > 0.02):
        Pt, Vt = resample_grid(Pt, Vt, sr, sc)
        H, W = Vt.shape
        maps = {k: _nearest_map(v, H, W, sr, sc) for k, v in maps.items()}
    N, ok = grid_normals(Pt, Vt)
    N = orient_to_axis(N, Pt, ok, axis)
    geom = {"P": Pt.float().cpu().numpy(), "V": ok.cpu().numpy(), "N": N.float().cpu().numpy(), **maps}
    m = dict(meta)
    m.update(id=oid, cell_vox=float(cell_vox), cell_um=float(meta.get("cell_um", cell_vox * meta["pitch_um"])),
             grid_step_in=[float(dr), float(dc)])
    m.setdefault("axis_trusted", False)
    return BandStore.create(out, geom, spec, m)
