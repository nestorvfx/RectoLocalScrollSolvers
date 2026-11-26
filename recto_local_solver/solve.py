"""THE RECTO LOCAL SOLVER: an area of a sheet solved from its known cells (the command line and the benchmark call
this).

  1. calibrate: the frame's matched filter from the known cells (sheetsolve.calibrate: the [vertical ply, horizontal
     ply, intensity] depth profile at the known surface against the other depths);
  2. the FRAME (hsolve.frame_strip): the lamination-following growth over the whole area from the known cells, with
     the scroll's fibre-focus offsets (`focus`);
  3. the FIBRE STEP (hsolve.fibre_field_id) on the frame: the known cells' thick horizontal fibres decide the recto
     surface near the frame that continues them, and the surface is moved through their centres; faded in from the
     known cells over `ramp` cells. Its window: the area's rows, its columns + `ctx` columns of the known cells.
Two results: the frame (no horizontal fibres) and the frame + the fibre step.

The grid: one cell per 5.07 level-0 voxels (47.46 um on a 9.362 um scan), the horizontal fibres along the grid's
columns (the scroll's height along its rows), as the sheets VC3D flattens. Call volume.configure first."""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

FOCUS = (1.0, 2.0)          # foc_kH, foc_kV: the depths (vox, along the normals) at which the known cells' horizontal /
                            # vertical fibres run on best, PHerc0139's (docs/GUIDE.md); PHerc0211's: (1.0, 4.3)


def calibrate(P, known, out):
    """the frame's matched filter on the known cells of P, saved to `out` (npz); returns its record."""
    import torch
    from . import sheetsolve as SS
    from . import volume as qc
    N, _ = qc.normals(P, known)
    rr, cc = np.nonzero(known)
    reg = (int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1)
    dk = ndimage.distance_transform_edt(known)
    inner = (dk >= (3 if (dk >= 3).sum() >= 200 else 1))[reg[0]:reg[1], reg[2]:reg[3]]   # away from the known edge
    cfg = dict(SS.CFG, J=16)
    ks = SS.slab_ks(cfg, cfg["kmax"])
    Fd = SS.region_feats(P, known, N, reg, ks, cfg)
    Fd["okc"] = Fd["okc"] & torch.as_tensor(inner, device=SS.DEV)
    cal = SS.calibrate([Fd], ks, cfg)
    np.savez(out, **{k: np.asarray(x) for k, x in cal.items()})
    return dict(n_pos=cal["n_pos"], n_neg=cal["n_neg"], dprime=round(cal["dprime"], 3))


def solve_area(P, known, area, work, focus=FOCUS, ctx=80, fibre_step=True, log=print):
    """P [H, W, 3] xyz (level-0 voxels; only the known cells' positions are read), known / area [H, W] bool, work a
    folder (the matched filter's npz). Returns dict(frame, fibre_step (None without it), solved, record)."""
    from pathlib import Path
    from . import hsolve as HS
    from . import volume as qc
    t0 = time.time()
    P = np.where(known[..., None], np.asarray(P, np.float64), 0.0)
    _, nok = qc.normals(P, known)
    known = known & nok
    zone = area & ~known
    # 1. the frame's matched filter from the known cells
    Path(work).mkdir(parents=True, exist_ok=True)
    calib = Path(work) / "calib.npz"
    rec = dict(calib=calibrate(P, known, calib))
    t1 = time.time()
    rec["s_calib"] = round(t1 - t0, 1)
    gcfg = dict(HS.GEN_CFG)
    gscfg = dict(HS.GEN_SCFG)
    gscfg.update(calib=str(calib), foc_kH=float(focus[0]), foc_kV=float(focus[1]))
    c = dict(HS.HCFG, ctx=int(ctx))
    target = HS.HW.TARGETS[c["target"]]
    # 2. the frame
    Fw, gotw, bridged, npat = HS.frame_strip(P, known, zone, gcfg, gscfg)
    solved = gotw & zone
    frame = np.where(solved[..., None], Fw, P)
    t2 = time.time()
    rec.update(cells=int(zone.sum()), solved=int(solved.sum()), bridged=int(bridged.sum()), patches=npat, s_frame=round(t2 - t1, 1))
    log(f"frame: {rec['solved']} of {rec['cells']} cells ({rec['bridged']} bridged), {rec['s_frame']} s")
    if not fibre_step or not solved.any():
        return dict(frame=frame, fibre_step=None, solved=solved, record=rec)
    # 3. the fibre step on the frame, in the window: the area's rows, its columns + ctx known columns
    dist = ndimage.distance_transform_cdt(~known, metric="chessboard")
    rr, cc = np.nonzero(solved)
    R0, R1 = int(rr.min()), int(rr.max()) + 1
    C0 = max(int(cc.min()) - c["ctx"], 0); C1 = min(int(cc.max()) + 1 + c["ctx"], known.shape[1])
    colsv = (known | solved)[R0:R1].any(0); idx = np.nonzero(colsv[C0:C1])[0]; C0, C1 = C0 + int(idx.min()), C0 + int(idx.max()) + 1
    pad = c["rpad"]; ga = R0 - pad; vv = known | solved
    sub = lambda m: np.pad(m[max(ga, 0):R1 + pad, C0:C1], ((max(ga, 0) - ga, max(R1 + pad - m.shape[0], 0)), (0, 0)))
    a0 = max(ga, 0); n_ = min(R1 + pad, known.shape[0]) - a0
    Nb, _ = qc.normals(frame, vv)
    ff = HS.fibre_field_id(frame, vv, (R0, R1), (C0, C1), target, sub(known), sub(solved), c)
    G = np.zeros(known.shape); G[a0:a0 + n_, C0:C1] = ff["C"][a0 - ga:a0 - ga + n_]
    rmp = np.clip(dist / float(c["ramp"]), 0.0, 1.0) if c["ramp"] > 0 else np.ones(known.shape)
    Q = np.where(solved[..., None], frame + (G * rmp)[..., None] * np.nan_to_num(Nb), P)
    rec.update(s_fibre_step=round(time.time() - t2, 1), window=[R0, R1, C0, C1], fibres=len(ff["fibres"]),
               fibres_used=len(ff["ident"]) + len(ff["extra"]), decision=ff["follow"], guarded=ff["guarded"],
               moved_p50=round(float(np.median(np.abs(G * rmp)[solved])), 2))
    log(f"fibre step: {rec['fibres_used']} fibres used, median move {rec['moved_p50']} vox, {rec['s_fibre_step']} s")
    return dict(frame=frame, fibre_step=Q, solved=solved, record=rec)
