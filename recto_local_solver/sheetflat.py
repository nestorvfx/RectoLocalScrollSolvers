"""SHEETFLAT: the whole-patch generator (the frame of the recto local solver).

The sheet is the surface through the known mesh that FOLLOWS THE LAMINATIONS everywhere (seismic "flattening without
picking", Lomask et al. 2006; least-squares horizons with local slopes, Wu & Fomel 2018), anchored at every cell by the
known cells' ply signature:

  per patch (`depth` cells beyond the mesh, the whole width of the front, solved at once):
    reference R = a geometric extrapolation of the mesh (only a coordinate frame: errors of many voxels are allowed)
    slab V(u, v, k) along R's normals, k in +-kmax (several windings: the solve is not tied to R)
    unknown h(u, v); energy (sheetsolve.solve_patch: coarse bins -> fine cells -> continuous):
      unary    the matched filter of the known cells' [vertical ply, horizontal ply, intensity] depth profile + the
               fibre focus (the horizontal and vertical fibres run on in the layer)
      pairwise dip_w * (label change - the change the laminations' 3-D orientation predicts)^2
      pins     the mesh band at h = 0
      then outer re-samplings along the solved normals
    receding horizon: accept the inner `accept` cells, extend from the new edge; JOINT: every patch solve includes all
    the cells grown so far as free variables; the SLANT GATE keeps out cells whose surface crosses the laminations (a
    winding ramp) and everything beyond them.
library: flatgrow(P, v, allowed, cfg, scfg, log) -> P, v, step_map, info"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import ndimage

from . import fclib as L
from . import sheetgrow as SG
from . import sheetsolve as SS
from . import volume as qc

FCFG = dict(depth=16, accept=12, pin_band=10, ref_w=2, ref_rad=8, ref_sig=4.0, ref_min_nb=12, relax_iters=8,
            max_patches=12, lookahead=True,   # lookahead: the next patch's reference = this patch's SOLVED, not accepted part
            slant_max=0.0, slant_wmin=0.3, slant_sig=2.0,   # SLANT GATE (0 = off): cells whose solved surface crosses the
                                              # laminations at > slant_max deg (a winding ramp) and everything beyond them
                                              # (not connected to the pins through flat cells) are neither accepted nor
                                              # used as the next patch's reference (holes, re-tried next patch)
            joint=False,                      # JOINT: every patch solve includes the cells grown so far as free variables
            stall_patches=3, stall_frac=0.0, stall_min=8,   # STALL STOP of the patch loop: after stall_patches patches in a
                                              # row that accept <= stall_frac of their cells (0: nothing), or at once when a
                                              # patch of <= stall_min cells accepts nothing. A larger patch that accepts
                                              # nothing IS retried
            crop=True, crop_margin=64)        # the growth on the allowed cells' bounding box + crop_margin cells
FLAT_SCFG = dict(kmax=16.0, dip_w=1.0, P1c=0.02, P1f=0.04, Dc=4.0, Df=2.0, beta=1.0, dip_cont=2.0, outer=1, outer_k=4.0)


def reference(P, v, patch, cfg, prev=None):
    """the coordinate frame: the patch filled ring by ring (ref_w cells) by the first-order isometric predictor of
    sheetgrow (geometry only, no data); cells already SOLVED by the previous patch (prev = (P, mask), not accepted)
    are taken from there (data-driven) and only the rest is extrapolated."""
    pc = dict(rad=cfg["ref_rad"], sig=cfg["ref_sig"], min_nb=cfg["ref_min_nb"])
    R = P.copy(); vr = v.copy()
    if prev is not None:
        Pq, mq = prev
        mq = mq & patch
        R[mq] = Pq[mq]; vr |= mq
    d = ndimage.distance_transform_cdt(~v, metric="chessboard")
    for dd in range(cfg["ref_w"], int(d[patch].max()) + cfg["ref_w"] + 1, cfg["ref_w"]):
        ring = patch & (d > dd - cfg["ref_w"]) & (d <= dd) & ~vr
        if not ring.any():
            continue
        Pp, ok = SG.predict(R, vr, ring, pc)
        ring &= ok
        R[ring] = Pp[ring]; vr |= ring
    return R, vr


def _normals_local(R, vr, cells, margin):
    """the normals for `cells` and their surroundings only: the valid cells within `margin` of the cells' bounding box
    (margin >= the pin band + the 8 cells of the normals' own smoothing: the same normals there as from the whole
    frame)."""
    rr, cc = np.nonzero(cells)
    if len(rr) == 0:
        return np.zeros(R.shape), np.zeros(vr.shape, bool)
    vl = np.zeros_like(vr)
    a0, a1 = max(int(rr.min()) - margin, 0), min(int(rr.max()) + margin + 1, vr.shape[0])
    b0, b1 = max(int(cc.min()) - margin, 0), min(int(cc.max()) + margin + 1, vr.shape[1])
    vl[a0:a1, b0:b1] = vr[a0:a1, b0:b1]
    return L.normals_crop(R, vl, 0)


def flatgrow(P, v, allowed, cfg=None, scfg=None, log=print):
    """the growth on a CROP of the frame (exact): the bounding box of the allowed cells + crop_margin cells (> the pin
    band + the feature tiles' padding + the normals' smoothing: nothing outside it can influence the result). Returns
    full-frame arrays (P, v, step_map, info)."""
    c_ = dict(FCFG, **(cfg or {}))
    rr, cc = np.nonzero(allowed)
    if not c_["crop"] or len(rr) == 0:
        return _flatgrow(P, v, allowed, cfg, scfg, log)
    mg = int(c_["crop_margin"])
    H, W = v.shape
    a0, a1 = max(int(rr.min()) - mg, 0), min(int(rr.max()) + mg + 1, H)
    b0, b1 = max(int(cc.min()) - mg, 0), min(int(cc.max()) + mg + 1, W)
    sl = (slice(a0, a1), slice(b0, b1))
    Pc, vc, smc, info = _flatgrow(P[sl].copy(), v[sl].copy(), allowed[sl].copy(), cfg, scfg, log)
    Po = P.copy(); Po[sl] = Pc
    vo = v.copy(); vo[sl] = vc
    sm = np.zeros(v.shape, smc.dtype); sm[sl] = smc
    return Po, vo, sm, info


def _flatgrow(P, v, allowed, cfg=None, scfg=None, log=print):
    cfg = dict(FCFG, **(cfg or {}))
    sc = dict(SS.CFG, **FLAT_SCFG); sc.update(scfg or {})
    P = P.copy(); v = v.copy()
    step_map = np.zeros(v.shape, np.int16)
    info = []
    cdt = lambda m: ndimage.distance_transform_cdt(~m, metric="chessboard")
    prev = None
    known0 = v.copy()
    stall = 0
    for it in range(cfg["max_patches"]):
        t0 = time.time()
        d = cdt(v)
        patch = (d >= 1) & (d <= cfg["depth"]) & allowed & ~v
        if not patch.any():
            break
        R, vr = reference(P, v, patch, cfg, prev if cfg["lookahead"] else None)
        patch &= vr
        nxt = (d > cfg["depth"]) & (d <= cfg["depth"] + cfg["accept"]) & allowed & ~v
        if nxt.any():                                                 # the next patch's CT downloaded while this one is solved
            qc.prefetch_async(lambda R=R, vr=vr, nxt=nxt: _prefetch_next(R, vr, nxt, cfg, sc))
        grown = v & ~known0 if cfg["joint"] else np.zeros_like(v)     # the grown cells re-solved with the new patch
        N, nok = _normals_local(R, vr, patch | grown, cfg["pin_band"] + 12)
        patch &= nok
        free = patch | (grown & nok)
        pins = v & ~free & (cdt(free) <= cfg["pin_band"])
        roi = free | pins
        rr, cc = np.nonzero(roi)
        region = (int(rr.min()), int(rr.max() + 1), int(cc.min()), int(cc.max() + 1))
        R0, R1, C0, C1 = region
        vs = np.zeros_like(v); vs[R0:R1, C0:C1] = roi[R0:R1, C0:C1]
        res = SS.solve_patch(R, vs, N, region, pins[R0:R1, C0:C1], np.zeros((R1 - R0, C1 - C0)), sc, verbose=False)
        Pn = res["P"]
        acc = patch & (d <= cfg["accept"])
        okc = np.zeros_like(v); okc[R0:R1, C0:C1] = True
        acc &= okc
        good = _slant_gate(res, v.shape, region, free, roi, pins, cfg)
        acc &= good
        prev = (Pn, patch & okc & ~acc & good)
        upd = acc | (free & ~patch & okc & good) if cfg["joint"] else acc
        P = np.where(upd[..., None], Pn, P)
        v = v | acc
        if cfg["relax_iters"]:
            Nn, _ = _normals_local(P, v, upd, 14)
            P = SG.relax(P, v, upd, Nn, cfg["relax_iters"])
        step_map[acc] = it + 1
        rec = {"step": it + 1, "patch": int(patch.sum()), "grown": int(acc.sum()), "seconds": round(time.time() - t0, 2),
               "timing": res["timing"], "gated": int((patch & ~good).sum())}
        info.append(rec)
        log(json.dumps(rec, default=float))
        if int(cfg["stall_patches"]) > 0:                             # the stall stop (FCFG)
            na, npc = int(acc.sum()), int(patch.sum())
            stall = stall + 1 if na <= float(cfg["stall_frac"]) * npc else 0
            if stall >= int(cfg["stall_patches"]) or (na == 0 and npc <= int(cfg["stall_min"])):
                break
    return P, v, step_map, info              # (a prefetch still running only fills the disk cache: not waited for)


def _prefetch_next(R, vr, nxt, cfg, sc):
    """the CT chunks of the next patch's cells (beyond this patch's reference R, extrapolated the same way) within the
    slab's depth range (+ 8 vox for the refinement passes' re-sampling): a prediction, errors only cost downloads."""
    R2, vr2 = reference(R, vr, nxt, cfg)
    nxt = nxt & vr2
    if not nxt.any():
        return
    N2, ok = _normals_local(R2, vr2, nxt, 10)
    k = float(sc["kmax"]) + 8.0
    qc.prefetch(R2, nxt & ok, np.nan_to_num(N2), -k, k)


def _slant_gate(res, shape, region, free, roi, pins, cfg):
    """the SLANT GATE on a solve's result: the cells connected to the pins through cells whose solved surface does not
    cross the laminations at more than slant_max deg (the per-cell slant smoothed nan-aware over slant_sig cells); all
    cells when the gate is off."""
    R0, R1, C0, C1 = region
    sl = res["conf"].get("slant")
    if sl is None or cfg["slant_max"] <= 0:
        return np.ones(shape, bool)
    slf = np.full(shape, np.nan); slf[R0:R0 + sl.shape[0], C0:C0 + sl.shape[1]] = sl
    swf = np.zeros(shape); swf[R0:R0 + sl.shape[0], C0:C0 + sl.shape[1]] = res["conf"]["slant_w"]
    fin = np.isfinite(slf) & (swf > cfg["slant_wmin"])               # (the per-cell slant is noisy: ~3 deg on a traced sheet)
    num = ndimage.gaussian_filter(np.where(fin, slf, 0.0), cfg["slant_sig"])
    den = ndimage.gaussian_filter(fin.astype(float), cfg["slant_sig"])
    sms = np.where(den > 0.2, num / np.maximum(den, 1e-6), np.inf)
    bad = free & (sms > cfg["slant_max"])
    lab, _ = ndimage.label(roi & ~bad)
    keep = np.unique(lab[pins & (lab > 0)])
    return np.isin(lab, keep[keep > 0])
