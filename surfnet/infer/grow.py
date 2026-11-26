"""Grow an area of a sheet with GrowPatch before SurfNet corrects it (`surfnet grow`).

GrowPatch (VC3D `vc_grow_seg_from_seed`, the Vesuvius Challenge's tracer) grows a surface along the m7 surface
prediction from a seed: the known cells on its 20-voxel grid, grown for as many generations as the farthest cell of the
area needs (+3), within that grid (no grid expansion; its positions are not bounded). The seed's grid is the sheet's
own grid resampled to GrowPatch's step, measured on the known cells (with the area bridged from the known cells,
`surface.thin_plate_fill`, so that the grid reaches the area's edge). GrowPatch keeps that grid and grows it at the
same step, so its surface goes back onto the sheet's grid cell for cell: the sheet's masks apply to it unchanged. Area
cells GrowPatch did not reach are bridged; where GrowPatch gives no surface at all (it failed, timed out, or no known
cell lies on its seed grid) the whole area is, with a warning, and the report says so. Then

  surfnet correct --sheet OUT/surface --known K.png --solve A.png --mode fast --keep-draft ...

Outputs: OUT/growpatch (GrowPatch's own surface), OUT/surface (on the sheet's grid), OUT/report.json."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from ..common import tifxyz
from ..common.geometry import as_t, bilinear, grid_steps, resample_grid, valid_at
from ..data.sheet import read_mask
from .surface import thin_plate_fill

STEP = 20.0                      # vox: GrowPatch's grid step (that of the published normal grids)


def to_grid(P, V, masks: dict, steps, origin=(0.0, 0.0)):
    """P on its grid resampled with `steps` (sr, sc): output cell (i, j) lies at the input grid's (origin[0] + i * sr,
    origin[1] + j * sc); the masks by nearest. -> P, V, masks."""
    sr, sc = steps
    Pt, Vt = resample_grid(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), sr, sc, origin=origin)
    H, W = Vt.shape
    r = np.clip(np.rint(origin[0] + np.arange(H) * sr).astype(int), 0, V.shape[0] - 1)
    c = np.clip(np.rint(origin[1] + np.arange(W) * sc).astype(int), 0, V.shape[1] - 1)
    return Pt.numpy(), Vt.numpy(), {k: m[r[:, None], c[None, :]] for k, m in masks.items()}


def write_seed(d: Path, P, known, solve):
    """GrowPatch's seed (tifxyz + generations.tif): the known cells and the bridged area resampled together to its
    20-vox grid, the cells nearest a known cell kept. The steps come from the known cells (`geometry.grid_steps`), so
    the seed lies at GrowPatch's own step: a bridge extrapolated from a curved strip is shorter than the sheet (by up
    to 7.5 % on the benchmark's one-sided areas), and a grid laid on it would map the area that much short. The grid
    has a node on the first known row and column (its origin, within one step of the sheet's), so a known strip at
    least two cells wide is always in the seed. -> the generations needed to reach every area cell (+3), the seed
    grid's steps in sheet cells, its origin, and the number of seed cells."""
    import tifffile
    B = thin_plate_fill(P, known, solve)
    dr, dc = grid_steps(B, known | solve, known)
    steps = (STEP / dr, STEP / dc)
    kr, kc = np.nonzero(known)
    origin = (float(kr.min()) % steps[0], float(kc.min()) % steps[1])
    P20, V20, m = to_grid(B, known | solve, {"K": known, "S": solve}, steps, origin)
    K20 = m["K"] & V20
    tifxyz.write(d, np.where(K20[..., None], P20, -1.0), K20, {"scale": [1 / STEP] * 2}, overwrite=True)
    tifffile.imwrite(d / "generations.tif", K20.astype(np.uint16))
    reach = ndimage.distance_transform_edt(~K20)[m["S"] & V20]
    gens = int(1 + np.ceil(reach.max()) + 3) if reach.size else 4
    return gens, steps, origin, int(K20.sum())


def place(P, known, solve, Q, VQ, steps, origin, offset=(0, 0)):
    """GrowPatch's surface on the sheet's grid. GrowPatch keeps its seed's grid, padded by `offset` cells (its meta's
    grid_offset), and grows it at the seed's step; the seed's grid is the sheet's resampled with `steps` from
    `origin`: sheet cell (r, c) lies at GrowPatch's ((r - origin[0]) / sr + offset[0], (c - origin[1]) / sc +
    offset[1]). Area cells take GrowPatch's position there (bilinear) where it has one; the others are bridged from the
    known cells and those. -> P, covered."""
    H, W = known.shape
    r, c = np.meshgrid((np.arange(H) - origin[0]) / steps[0] + offset[0], (np.arange(W) - origin[1]) / steps[1] + offset[1],
                       indexing="ij")
    r, c = torch.as_tensor(r), torch.as_tensor(c)
    G = bilinear(as_t(Q, "cpu", torch.float64), r, c).numpy()
    cov = solve & valid_at(as_t(VQ, "cpu", torch.bool), r, c).numpy()
    D = np.where(cov[..., None], G, P)
    if (solve & ~cov).any():
        D = thin_plate_fill(D, known | cov, solve & ~cov)
    return D, cov


def fetch_grids(src: str, dst: Path) -> Path:
    if not src.startswith("s3://"):
        return Path(src)
    if not (dst / "metadata.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["s5cmd", "--no-sign-request", "sync", f"{src.rstrip('/')}/*", f"{dst}/"], check=True,
                       capture_output=True)
    return dst


def grow_sheet(sheet: str, known_mask: str, solve_mask: str, m7: str, normal_grids: str, pitch_um: float, vc3d: str,
               out: str, threads: int | None = None, timeout: float = 3600.0, log=print) -> dict:
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    P, V, meta = tifxyz.read(sheet)
    known = read_mask(known_mask, V.shape) & V
    solve = read_mask(solve_mask, V.shape) & ~known
    if not known.any() or not solve.any():
        raise SystemExit(f"known cells {int(known.sum())}, solve cells {int(solve.sum())}: both must be non-empty")
    generations, steps, origin, seeds = write_seed(outd / "seed", P, known, solve)
    params = {"voxelsize": pitch_um, "min_area_cm": 0, "generations": generations, "cache_size": 8e9,
              "thread_limit": threads or os.cpu_count(), "disable_grid_expansion": True,
              "normal_grid_path": str(fetch_grids(normal_grids, outd / "normal_grids"))}
    (outd / "growpatch_params.json").write_text(json.dumps(params, indent=1))
    log(f"GrowPatch: {int(known.sum())} known cells, {int(solve.sum())} to grow over, {generations} generations")
    shutil.rmtree(outd / "growpatch", ignore_errors=True)
    t0 = time.time()
    if not seeds:                                                # nothing for GrowPatch to start from
        failed = "no known cell lies on GrowPatch's 20-vox seed grid (the known cells are too narrow)"
        (outd / "growpatch.log").write_text(failed + "\n")
    else:
        try:
            r = subprocess.run([vc3d, "vc_grow_seg_from_seed", "-v", m7, "-t", str(outd / "growpatch"), "-p",
                                str(outd / "growpatch_params.json"), "--resume", str(outd / "seed"), "--segment-name",
                                "growpatch", "--resume-opt", "skip", "--rng-seed", "0"],
                               capture_output=True, text=True, timeout=timeout)
            (outd / "growpatch.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
            failed = None if r.returncode == 0 and (outd / "growpatch" / "x.tif").exists() else f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            failed = f"no result in {timeout:.0f} s"
    sec = time.time() - t0
    if failed:
        log(f"WARNING: GrowPatch gave no surface ({failed}; see {outd / 'growpatch.log'}): the area is bridged from "
            "the known cells")
        D, cov = thin_plate_fill(P, known, solve), np.zeros_like(solve)
    else:
        Q, VQ, gmeta = tifxyz.read(outd / "growpatch")
        D, cov = place(P, known, solve, Q, VQ, steps, origin, tuple(gmeta.get("grid_offset", (0, 0))))
    tifxyz.write(outd / "surface", np.where((V | solve)[..., None], D, -1.0), V | solve,
                 {k: v for k, v in meta.items() if k != "bbox"}, overwrite=True)
    rep = {"sheet": str(sheet), "m7": m7, "generations": generations, "seconds": round(sec, 1), "growpatch_failed": failed,
           "known_cells": int(known.sum()), "seed_cells": seeds, "solve_cells": int(solve.sum()),
           "covered": round(float(cov[solve].mean()), 4)}
    (outd / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"GrowPatch done in {sec:.0f} s; it covered {rep['covered']:.0%} of the area; written {outd / 'surface'}")
    return rep
