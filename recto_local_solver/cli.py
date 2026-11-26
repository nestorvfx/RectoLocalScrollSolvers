"""The recto local solver's command line (`recto-local-solver`, or `python -m recto_local_solver`).

  recto-local-solver solve --sheet S.tifxyz --known K.png --solve A.png --volume s3://.../volume.zarr/0 --axis AXIS.json
                           --out DIR [--frame-only] [--focus 1.0,2.0] [--context 80] [--cache DIR] [--gpu-cache-gb 8]

  --sheet   tifxyz directory (x.tif, y.tif, z.tif in the volume's level-0 voxels; invalid cells -1)
  --known   mask on the sheet's grid: the cells known to be correct (their positions are the only ones read)
  --solve   mask on the sheet's grid: the area to solve (invalid cells of the sheet may be in it)
  --volume  the CT's level 0: an OME-Zarr level of uint8, 128^3 raw chunks, '/' separator (every open-data scroll
            volume), as an s3:// URL (read with s5cmd, cached) or a local directory (read in place)
  --axis    the scroll axis, json {"z": [...], "y": [...], "x": [...]} in level-0 voxels (the normals point toward it)
-> DIR/frame.tifxyz (the frame), DIR/frame_fibre_step.tifxyz (the frame + the fibre step), DIR/solve.json; a rerun
   into the same DIR replaces them.

The solver's grid: the scroll's height along the rows, one cell per 5.07 level-0 voxels (47.46 um on a 9.362 um scan).
A sheet whose step, measured on its known cells, differs from that by more than 10 % (e.g. GrowPatch's 20-voxel grid) is
resampled to it first, as `surfnet correct` resamples (positions bicubic, masks nearest), and the results are written
on that grid."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

CELL_VOX = 5.06943      # the solver's grid step (vox), as sheetgrow.CELLV
STEP_TOL = 0.10         # a sheet within 10 % of it is solved on its own grid (the benchmark's: 4.89-5.15 vox)


def read_tifxyz(d):
    import tifffile
    d = Path(d)
    x, y, z = (tifffile.imread(d / f"{a}.tif").astype(np.float64) for a in "xyz")
    v = (x > 0) & (y > 0) & (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return np.stack([x, y, z], -1), v, meta


def write_tifxyz(d, P, v, meta):
    import tifffile
    d = Path(d)
    if d.exists():                                     # a rerun replaces the earlier result
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, a in enumerate("xyz"):
        tifffile.imwrite(d / f"{a}.tif", np.where(v, P[..., i], -1).astype(np.float32))
    pv = P[v]
    (d / "meta.json").write_text(json.dumps(dict(meta, bbox=[pv.min(0).tolist(), pv.max(0).tolist()]), indent=1))


def read_mask(path, shape):
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


def solver_grid(P, V, known, area, meta, log=print):
    """the sheet on the solver's grid: the scroll's height along the rows (checked), and one cell per CELL_VOX voxels
    (resampled when the step measured on the known cells differs by more than STEP_TOL). -> P, V, known, area, meta,
    the resampling's steps (in the sheet's cells; (1, 1): none)."""
    from surfnet.common.geometry import grid_steps
    from surfnet.infer.correct import _to_model_grid
    if known.sum() < 2:
        raise SystemExit(f"known cells: {int(known.sum())}; the solver needs the approved cells around the area")
    dr, dc = grid_steps(P, V, known)
    steps = (1.0, 1.0)
    if abs(dr / CELL_VOX - 1) > STEP_TOL or abs(dc / CELL_VOX - 1) > STEP_TOL:
        P, V, m, steps = _to_model_grid(P, V, {"known": known, "area": area}, CELL_VOX, known, tol=0.0)
        known, area = m["known"] & V, m["area"] & ~(m["known"] & V)
        meta = dict(meta, scale=[1.0 / CELL_VOX] * 2)
        log(f"sheet step {dr:.2f} x {dc:.2f} vox (rows x columns): resampled to {CELL_VOX:.2f} vox, {V.shape[0]} x "
            f"{V.shape[1]} cells; the results are on that grid")
    zr = np.abs(np.diff(P[..., 2], axis=0))[known[1:] & known[:-1]]
    zc = np.abs(np.diff(P[..., 2], axis=1))[known[:, 1:] & known[:, :-1]]
    if zr.size and zc.size and np.median(zr) < np.median(zc):
        raise SystemExit("the sheet's rows do not run along the scroll's height (z changes along its columns): the "
                         "solver needs the horizontal fibres along the columns, as VC3D flattens sheets; transpose "
                         "the sheet and the masks")
    return P, V, known, area, meta, steps


def cmd_solve(a):
    from . import volume
    from .solve import solve_area
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    volume.configure(a.volume, a.cache or out / "cache", a.axis, a.gpu_cache_gb)
    P, V, meta = read_tifxyz(a.sheet)
    known = read_mask(a.known, V.shape) & V
    area = read_mask(a.solve, V.shape) & ~known
    P, V, known, area, meta, steps = solver_grid(P, V, known, area, {k: v for k, v in meta.items() if k != "bbox"})
    focus = tuple(float(x) for x in a.focus.split(","))
    r = solve_area(P, known, area, out / "work", focus=focus, ctx=a.context, fibre_step=not a.frame_only)
    keep = known | r["solved"]
    write_tifxyz(out / "frame.tifxyz", np.where(known[..., None], P, r["frame"]), keep, meta)
    if r["fibre_step"] is not None:
        write_tifxyz(out / "frame_fibre_step.tifxyz", np.where(known[..., None], P, r["fibre_step"]), keep, meta)
    elif (out / "frame_fibre_step.tifxyz").exists():      # (an earlier run's, not this one's)
        shutil.rmtree(out / "frame_fibre_step.tifxyz")
    (out / "solve.json").write_text(json.dumps(dict(r["record"], focus=focus, context=a.context, volume=a.volume,
                                                    grid=list(V.shape), resampled=list(steps)), indent=1, default=str))
    print(f"done: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recto-local-solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("solve", help="an area of a sheet solved from its known cells")
    p.add_argument("--sheet", required=True)
    p.add_argument("--known", required=True)
    p.add_argument("--solve", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--axis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame-only", action="store_true", help="the frame alone (no fibre step)")
    p.add_argument("--focus", default="1.0,2.0", help="the scroll's fibre-focus offsets foc_kH,foc_kV (vox; docs/GUIDE.md)")
    p.add_argument("--context", type=int, default=80, help="known columns read beside the area by the fibre step")
    p.add_argument("--cache", default=None, help="the CT chunk cache (default OUT/cache)")
    p.add_argument("--gpu-cache-gb", type=float, default=None)
    a = ap.parse_args(argv)
    {"solve": cmd_solve}[a.cmd](a)


if __name__ == "__main__":
    main()
