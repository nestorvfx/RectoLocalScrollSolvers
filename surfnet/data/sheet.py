"""Your own sheet as a training object (for fine-tuning): a tifxyz surface + a mask of the cells you verified.

- verified cells are the targets (tier 1, sigma 0.5 vox); every other valid cell is context only (no target);
- rejected cells (optional) are cells you know to be wrong: inputs only, never targets;
- held-out cells (optional) are kept for validation (split HELDOUT) and never enter a training crop.
Masks are images or .npy arrays on the tifxyz grid (non-zero = set). The surface is resampled to the model's cell
(47.46 um; its step measured on the verified cells) at the processing level of the volume, as the official training
objects are (`objects.build_object`).
Normals: toward the scroll axis when `axis` is given, else the grid's own orientation (d/drow x d/dcol, which points
to the recto for VC3D meshes); `flip` reverses them. The sheet's surface convention is taken as is (face offset 0)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..common import tifxyz
from ..common.geometry import Axis
from .objects import build_object, level_coords
from .store import SPLIT_HELDOUT, SPLIT_TRAIN, BandSpec, BandStore

CELL_UM = 47.46


def read_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if p.suffix == ".npy":
        m = np.load(p)
    else:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        m = np.asarray(Image.open(p))
        if m.ndim == 3:
            m = m[..., :3].max(-1)
    m = np.asarray(m) != 0
    if m.shape != tuple(shape):
        raise ValueError(f"{p}: mask {m.shape} is not on the sheet's grid {tuple(shape)}")
    return m


def processing_level(pitch_um: float) -> int:
    """The pyramid level closest to the ~9.4 um the model was trained at (9.36 um -> 0, 2.4 um -> 2)."""
    return max(0, int(round(math.log2(9.4 / pitch_um))))


def add_sheet(root: str | Path, oid: str, sheet: str | Path, verified: str | Path, volume: str, pitch_um: float, *,
              rejected=None, heldout=None, level: int | None = None, axis: str | Path | None = None,
              flip: bool = False, spacing_vox: float = 13.0, spec: BandSpec, device="cuda",
              overwrite: bool = False) -> BandStore:
    P0, V, meta_in = tifxyz.read(sheet)
    ver = read_mask(verified, V.shape) & V
    rej = read_mask(rejected, V.shape)
    held = read_mask(heldout, V.shape)
    if not ver.any():
        raise ValueError("no verified cell on the sheet")
    level = processing_level(pitch_um) if level is None else int(level)
    tier = ver.astype(np.uint8)
    split = np.full(V.shape, SPLIT_TRAIN, np.uint8)
    if held is not None:
        split[held & V] = SPLIT_HELDOUT
    ax = None
    if axis:                                                 # level-0 voxels -> processing-level voxels
        a0 = Axis.from_json(axis)
        ax = Axis(*(level_coords(v, level) for v in (a0.z, a0.y, a0.x)))
    cell_vox = CELL_UM / (pitch_um * 2 ** level)
    meta = {"scroll": meta_in.get("scroll", oid), "volume": str(volume), "volume_url": str(volume), "level": level,
            "pitch_um": float(pitch_um), "cell_um": CELL_UM, "mesh": str(sheet), "source": "verified sheet",
            "verified_cells_in": int(ver.sum()), "spacing_vox": float(spacing_vox), "spacing_fixed": True,
            "face_offset_vox": 0.0, "axis_trusted": ax is not None}
    s = build_object(root, oid, level_coords(P0, level), V, axis=ax, cell_vox=cell_vox, spec=spec, meta=meta,
                     tier=tier, split=split, winding=-1, neg=None if rej is None else rej & V, device=device,
                     overwrite=overwrite, known=ver)
    if flip:
        s.update_geom(N=-s.geom["N"])
        s.update_meta(flipped=True)
    return s
