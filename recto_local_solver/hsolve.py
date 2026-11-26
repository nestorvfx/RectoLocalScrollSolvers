"""THE RECTO LOCAL SOLVER'S TWO STEPS.

  1. THE FRAME (frame_strip): the lamination-following growth (sheetflat.flatgrow with the frame's energy, GEN_CFG /
     GEN_SCFG) places the area from the known cells; cells it does not reach are grown again without the slant gate,
     the rest bridged (harmonic fill).
  2. THE FIBRE STEP (fibre_field_id): the known cells' thick horizontal fibres decide which recto surface near the
     frame continues them, and the surface is moved onto their centres:
       the decision (render_follow): the area and the known cells are read +-22 vox around the frame (hfibwin.detect's
         stack). A candidate surface = the frame moved by a label (+-18 vox), judged on ITS OWN render (the read's
         layers o-1..o+1) with the detector's own mechanics: the given fibres are the ones the detector
         (render_detect) finds VISIBLE on the known cells' render; each is followed on every candidate as a row path
         (its own width's evidence, held on its known rows), and per column the detector's visibility test says whether
         it shows there; credit = visible x clarity relative to the known cells x local continuity; ONE joint
         labelling over the area (semi-global, the known cells pinned, the dark gap, a slight frame prior);
       the bump: the fibres tracked in 3-D only within -3..+4 vox of the decided surface; the surface goes through their
         centres (target A), minus its own bumps; the void guard: never onto the dark gap.
     solve.solve_area fades the move in from the known cells over `ramp` cells.
Normal convention: + = along the normals toward the scroll axis."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import hfibwin as HW
from . import sheetflat as SF
from . import sheetsolve as SS
from . import volume as qc

HCFG = dict(target="A", ctx=80, tile_rows=48, rpad=12, ramp=8,
            sr=1.0, sc=2.0, cmax=20.0, bump_sig=3.0, void_frac=0.55,
            follow=True,                                              # the decision (render_follow); off: the bump only
            fl_rlam=0.5, fl_minctx=5, fl_R=4, fl_avg=6, fl_omax=18.0, fl_wv=1.0, fl_wf=0.02, fl_D=2, fl_P1=0.05,
            ident=dict(DSEL=(-20.0, 20.0), KLO=-22.0, KHI=22.0, RK=2, KPATH=2.0, MAXF=24, PLYLIM=(-1.5, 2.5)),
                                                                      # PLYLIM: the known cells are the truth, their fibres
                                                                      # lie on top of them
            idc=5, idz=3, zone_tol=2.5,
            place=dict(DSEL=(-3.0, 4.0), KLO=-9.0, KHI=10.0))        # the fine placement: ONLY close to the decided surface
                                                                      # (ctx: the given columns read beside the area)
GEN_CFG = {"depth": 16, "accept": 8, "joint": True, "slant_max": 6.0, "slant_sig": 2.0, "max_patches": 14,
           "stall_patches": 1, "stall_frac": 0.05}
                                                                      # joint: every patch re-solves all the cells grown so far.
                                                                      # The stall stop: a patch accepting <= 5 % (the slant gate
                                                                      # holding the same cells back: the next one would re-solve
                                                                      # them all to accept a few; the growth without the gate
                                                                      # places them)
GEN_SCFG = {"kmax": 24.0, "Dc": 8.0, "Df": 2.5, "dip_cap": 25.0, "outer": 5, "outer_k": 8.0, "outer_B": 5, "outer_Dc": 4.0,
            "scan_np": 26, "outer_scan_pmax": 0.4, "outer_scan_np": 17, "dip_method": "shear", "dip_gain": 1.0,
            "outer_dip_method": "tensor", "outer_dip_gain": 1.28, "slant_scan": True, "shear_iters": 1, "slant_method": "shear",
            "dip_tol": 0.25, "outer_tol": 0.5, "foc_w": 8.0, "foc_v": 0.5, "foc_kH": 1.0, "foc_kV": 4.3,
            "feat_tile": [192, 192], "bar_tile": 192}
                                                                      # the frame's energy (with the fibre focus
                                                                      # term); calib, foc_kH, foc_kV are set per scroll / sheet.
                                                                      # The features of a region on ONE canvas up to 192 cells
                                                                      # (sheetsolve.region_feats)


def harmonic_fill(Pc, valid, holes, iters=400):
    Pc = Pc.copy(); valid = valid.copy(); holes = holes & ~valid
    for _ in range(iters):
        if not holes.any():
            break
        cnt = np.zeros(valid.shape); acc = np.zeros(Pc.shape)
        for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            vs = np.roll(valid, (oy, ox), (0, 1)); ps = np.roll(np.where(valid[..., None], Pc, 0.0), (oy, ox), (0, 1))
            cnt += vs; acc += ps
        ready = holes & (cnt >= 2)
        if not ready.any():
            ready = holes & (cnt >= 1)
            if not ready.any():
                break
        Pc[ready] = acc[ready] / cnt[ready][:, None]
        valid |= ready; holes &= ~ready
    return Pc, valid


def frame_strip(P, v, allowed, gcfg, gscfg):
    """step 1: the growth into `allowed` from the valid cells v (max_patches patches of `accept` cells: 14 x 8 = 112
    cells deep); the cells it did not reach grown again without the slant gate (up to 32 patches more: 256 cells, the far
    side of a 320-cell extension), the rest bridged (harmonic). Returns P, the cells placed, the cells bridged, the
    patches of the two growths."""
    Pg, vg, _, info = SF.flatgrow(np.where(v[..., None], P, 0.0), v.copy(), allowed, gcfg, gscfg, log=lambda s: None)
    got = vg & allowed & ~v
    rem = allowed & ~got
    npat = [len(info), 0]
    if rem.any():
        Pg2, vg2, _, info2 = SF.flatgrow(np.where((v | got)[..., None], np.where(got[..., None], Pg, P), 0.0), v | got, rem,
                                         dict(gcfg, slant_max=0.0, max_patches=32), gscfg, log=lambda s: None)
        npat[1] = len(info2)
        g2 = vg2 & rem & ~(v | got)
        Pg = np.where(g2[..., None], Pg2, Pg); got |= g2
    rem = allowed & ~got
    br = np.zeros_like(v)
    if rem.any():
        Pf, vf = harmonic_fill(np.where((v | got)[..., None], np.where(got[..., None], Pg, P), 0.0), v | got, rem)
        br = vf & rem
        Pg = np.where(br[..., None], Pf, Pg); got |= br
    return np.where(got[..., None], Pg, P), got, br, npat


def _follow_core(Cst, Rst, om, fib, okb, okm, Z, lv, cfg, S):
    """THE GIVEN FIBRES FOLLOWED ON EVERY CANDIDATE RENDER with the detector's own mechanics (render_follow's steps
    2a-2c): Cst / Rst [NL, Y, X] the candidates' renders (cleaned / as read, straightened), om the reference candidate
    (the evidence's normalisation), fib the given fibres (w, y, pin, ...), okb [Xs] / okm [Y, X] the valid blocks /
    pixels, Z [Y, Xs] the zone blocks, lv [NL] the valid candidates. Each fibre's row path on every candidate (its
    width +- 1 step of the two-edge filter, held on its pinned rows), along it the visibility test per column.
    Returns vis [nF, NL, Xs], clarity [nF, NL, Xs], rows [nF, NL, Xs], Rb, zon, okbt, Ev (torch)."""
    import torch
    import torch.nn.functional as F
    from . import hfib
    RC = dict(HW.DCFG); dev = Cst.device
    NL, Y, X = Cst.shape; Xs = X // S; nF = len(fib); WID = np.asarray(RC["WIDTHS"], float)
    # 2a. the detector's filter on every candidate's render, normalised as the detector does (at label 0)
    Ibs = Cst[..., :Xs * S].reshape(NL, Y, Xs, S).mean(-1)
    Ev, _ = hfib.tow_evidence(Ibs, list(RC["WIDTHS"]), kcore=0, kfl=tuple(RC["KFL"]), fw=RC["FW"], beta=0.0, mode="koller",
                              gamma=RC["GAMMA"], etol=RC["ETOL"], esig=RC["ESIG"])
    okbt = torch.as_tensor(okb, device=dev)
    ref = Ev[len(WID) // 2, om][:, okbt]; med = ref.median(); sig = 1.4826 * (ref - med).abs().median()
    Ev = (Ev - med) / sig.clamp_min(1e-9)
    # 2b. each fibre's row path on every candidate (its width +- 1 step), held on its own rows in the context
    R = int(cfg["fl_R"]); R2 = 2 * R + 1
    Q = torch.full((nF, NL, R2, Xs), -1.0, device=dev)
    for i, f in enumerate(fib):
        wi = int(np.argmin(np.abs(WID - f["w"]))); w0, w1 = max(wi - 1, 0), min(wi + 1, len(WID) - 1)
        y0 = f["y"] - R; a0, a1 = max(y0, 0), min(f["y"] + R + 1, Y)
        Q[i, :, a0 - y0:a1 - y0] = Ev[w0:w1 + 1, :, a0:a1].amax(0).clamp(-1.0, 3.0)
        Q[i][:, :, ~okbt] = 0.0
        for x_, yr in f["pin"].items():
            if 0 <= x_ < Xs:
                ri = int(round(yr)) - y0
                Q[i, :, :, x_] = -5.0
                if 0 <= ri < R2:
                    Q[i, :, max(ri - 1, 0):ri + 2, x_] = 3.0
    lam = float(cfg["fl_rlam"])

    def paths_torch():
        dys = (-1, 0, 1); dyt = torch.as_tensor(dys, device=dev)
        D = Q[..., 0].clone(); bp = torch.zeros((nF, NL, Xs, R2), dtype=torch.int8, device=dev)
        for x in range(1, Xs):
            cs = []
            for dy in dys:
                if dy == 0:
                    sh = D
                else:
                    sh = torch.full_like(D, -1e9)
                    if dy > 0:
                        sh[..., dy:] = D[..., :-dy]
                    else:
                        sh[..., :dy] = D[..., -dy:]
                cs.append(sh - lam * abs(dy))
            best, arg = torch.stack(cs, 0).max(0)
            D = best + Q[..., x]; bp[..., x, :] = arg.to(torch.int8)
        ix = D.argmax(-1); path = torch.zeros((nF, NL, Xs), dtype=torch.long, device=dev)
        for x in range(Xs - 1, -1, -1):
            path[..., x] = ix
            if x > 0:
                a = bp[..., x, :].gather(-1, ix[..., None])[..., 0].long()
                ix = (ix - dyt[a]).clamp(0, R2 - 1)
        return path
    from . import kernels as KN
    path = (KN.run("follow_dp", lambda: KN.follow_dp(Q.contiguous(), lam), paths_torch, 0.0) if Q.is_cuda and KN.available()
            else paths_torch())
    yrow = torch.as_tensor([f["y"] - R for f in fib], device=dev)[:, None, None] + path      # [nF, NL, Xs]
    # 2c. the detector's visibility test (visible_nodes), on every candidate's render as read, along each fibre
    Rb = Rst[..., :Xs * S].reshape(NL, Y, Xs, S).mean(-1)                  # [NL, Y, Xs]
    # (the same normalisations as render_measures: over the VALID pixels only)
    okmb = okm[:, :Xs * S].reshape(Y, Xs, S).mean(-1) > 0.5
    pair = torch.as_tensor(okmb[1:] & okmb[:-1] & okb[None], device=dev)
    noise = (1.4826 * (Rb[:, 1:] - Rb[:, :-1]).abs()[:, pair].median(1).values / np.sqrt(2.0)).clamp_min(1e-6)
    Gx = torch.zeros_like(Rst); Gx[..., 1:-1] = 0.5 * (Rst[..., 2:] - Rst[..., :-2]).abs()
    gm = okm.copy(); gm[:, 1:-1] &= okm[:, 2:] & okm[:, :-2]; gm[:, [0, -1]] = False
    gx_typ = Gx[:, torch.as_tensor(gm, device=dev)].median(1).values.clamp_min(1e-6)
    csx = F.pad(Gx.cumsum(-1), (1, 0))
    xb = np.arange(Xs); ca = torch.as_tensor(np.clip(xb * S - 2, 0, X), device=dev); cb = torch.as_tensor(np.clip((xb + 1) * S + 2, 0, X), device=dev)
    Hb = (csx[..., cb] - csx[..., ca]) / (cb - ca).float()                 # [NL, Y, Xs] mean |d/dx| over the block +-2 px
    cR = F.pad(Rb.cumsum(1), (0, 0, 1, 0)); cH = F.pad(Hb.cumsum(1), (0, 0, 1, 0))
    VFL, VSM = int(RC["VFL"]), int(RC["VSM"])
    lvt = torch.as_tensor(lv, device=dev)
    zon = torch.as_tensor(np.stack([Z[f["y"]] for f in fib]), device=dev)
    vis_all = torch.zeros((nF, NL, Xs), device=dev); cl_all = torch.zeros((nF, NL, Xs), device=dev)
    for i, f in enumerate(fib):
        yr = yrow[i].float()
        t0 = (yr - f["w"] / 2).floor().long().clamp(0, Y - 1); t1 = (yr + f["w"] / 2).ceil().long().clamp(0, Y - 1)
        t1 = torch.maximum(t1, t0)
        g = lambda T, r: T.gather(1, r[:, None, :])[:, 0]                  # T [NL, Y(+1), Xs], r [NL, Xs]
        nrow = (t1 - t0 + 1).float()
        band = (g(cR, t1 + 1) - g(cR, t0)) / nrow
        upm = torch.stack([g(Rb, (t0 - j).clamp(0, Y - 1)) for j in range(VFL + 1)], 0).min(0).values
        dnm = torch.stack([g(Rb, (t1 + j).clamp(0, Y - 1)) for j in range(VFL + 1)], 0).min(0).values
        cl = torch.minimum(band - upm, band - dnm) / noise[:, None]
        cl = torch.where(okbt[None], cl, torch.zeros_like(cl))
        hx = (g(cH, t1 + 1) - g(cH, t0)) / nrow / gx_typ[:, None]
        sm = lambda T: F.pad(T[:, None], (VSM, VSM), mode="replicate")[:, 0].unfold(-1, 2 * VSM + 1, 1).median(-1).values
        cls, hxs = sm(cl), sm(hx)
        vis_all[i] = ((cls >= RC["VTAU"]) & (hxs <= RC["HXT"]) & lvt[:, None]).float()
        cl_all[i] = cls
    return vis_all, cl_all, yrow, Rb, zon, okbt, Ev


def render_follow(det, cfg):
    """THE DECISION, with the DETECTOR'S OWN MECHANICS (hfibwin.render_detect). det = the anchored detect of the
    window (the frame's straightened stack +-22 vox, cleaned Ism_t and as read Ist_t). A candidate surface = the frame
    moved by o vox (label o, +- fl_omax); ITS render = the stack's layers o - 1, o, o + 1 (what the surface would show).
      1. THE GIVEN FIBRES = the detector on the window as it is (label 0: the known cells on their own surface):
         render_detect, the fibres with >= fl_minctx nodes in the known columns and visible there in >= VMIN of them;
         each with its thickness, its rows in the known cells (held there) and its clarity there.
      2. EACH OF THEM CONTINUED ON EVERY CANDIDATE SURFACE: its row path on that render (the detector's two-edge filter
         for ITS width +- 1 step, one optimal path over the rows, <= 1 px per block, fl_rlam per px, held on its own rows
         in the known columns); along it, column by column, the detector's VISIBILITY TEST on that render as read
         (visible_nodes: the weaker edge contrast / noise >= VTAU and the along-fibre variation <= HXT, median over
         +-VSM columns).
      3. CREDIT where it is visible: its clarity relative to its own clarity in the known cells (<= 1) x the share of
         visible columns around it (+- fl_avg blocks: continuity); ONE joint labelling over the area (boxes of 5 rows x
         column blocks; every box a fibre covers at that label gets - its credit; the dark gap, a slight frame prior;
         the known cells pinned at 0; sheetsolve.sgm): the surface on which the given fibres stay visible, cleanly and
         continuously.
    Returns the offset (vox) per straightened pixel and a record."""
    import torch
    import torch.nn.functional as F
    if "Ism_t" not in det:
        return None, dict(reason="no stack")
    dev = SS.DEV
    Ism = det["Ism_t"].to(dev).float(); Ist = det["Ist_t"].to(dev).float()
    offs = np.asarray(det["offs"], float); step = float(offs[1] - offs[0]); K, Y, X = Ism.shape; S = int(det["S"]); Xs = X // S
    blk = lambda m: m[:, :Xs * S].reshape(m.shape[0], Xs, S).mean(-1)
    A = blk(det["anc_st"].astype(float)) > 0.5; OKb = blk(det["ok_st"].astype(float)) > 0.5; Z = OKb & ~A
    okcol = det["ok_st"].mean(0) > 0.5; okb = okcol[:Xs * S].reshape(Xs, S).mean(1) > 0.5
    if A.sum() < 20 or Z.sum() < 20:
        return None, dict(reason="context / zone too small", ctx_blocks=int(A.sum()), zone_blocks=int(Z.sum()))
    RC = dict(HW.DCFG)
    k0 = int(np.argmin(np.abs(offs))); h1 = int(round(1.0 / step))
    om = int(round(float(cfg["fl_omax"]) / step)); labs = np.arange(-om, om + 1); NL = len(labs)
    kl = k0 + labs; lv = (kl - h1 >= 0) & (kl + h1 < K)
    idx = torch.as_tensor(np.clip(kl, h1, K - 1 - h1), device=dev)
    Cst = (Ism[idx - h1] + Ism[idx] + Ism[idx + h1]) / 3.0                # [NL, Y, X] every candidate's render, cleaned
    Rst = (Ist[idx - h1] + Ist[idx] + Ist[idx + h1]) / 3.0                # ... and as read
    # 1. the given fibres: the detector itself at label 0
    okm = np.asarray(det["ok_st"], bool)
    tr0 = HW.render_detect(Cst[om], Rst[om], S, okb, RC, okm=okm)
    zc = np.nonzero(Z.any(0))[0]
    fib = []; why = dict(tracks=len(tr0), with_ctx=0, visible=0)
    for tr in tr0:
        nodes = sorted(tr, key=lambda n: n["x"])
        cn = [n for n in nodes if A[int(np.clip(round(n["yc"]), 0, Y - 1)), min(int(n["x"]), Xs - 1)]]
        if len(cn) < cfg["fl_minctx"] or len(zc) == 0:
            continue
        why["with_ctx"] += 1
        vis = float(np.mean([n["vis_on"] for n in cn]))
        if vis < RC["VMIN"]:
            continue
        why["visible"] += 1
        dz = np.array([np.min(np.abs(zc - n["x"])) for n in cn]); near = np.argsort(dz)[:10]
        cv = [n["clar"] for n in cn if n["vis_on"]]
        fib.append(dict(w=float(np.median([n["w"] for n in cn])), y=int(round(np.median([cn[j]["yc"] for j in near]))),
                        pin={int(n["x"]): float(n["yc"]) for n in cn}, clar=max(float(np.median(cv)) if cv else RC["VTAU"], RC["VTAU"]),
                        vis_ctx=round(vis, 2)))
    if not fib:
        return None, dict(reason="no given-block fibre", **why)
    nF = len(fib)
    vis_all, cl_all, yrow, Rb, zon, okbt, Ev = _follow_core(Cst, Rst, om, fib, okb, okm, Z, lv, cfg, S)
    lvt = torch.as_tensor(lv, device=dev)
    # 3. credit and the joint decision
    a_ = int(cfg["fl_avg"])
    loc = F.avg_pool1d(vis_all.reshape(-1, 1, Xs), 2 * a_ + 1, 1, a_, count_include_pad=False).reshape(nF, NL, Xs)
    rel = torch.stack([(cl_all[i] / f["clar"]).clamp(0, 1) for i, f in enumerate(fib)], 0)
    cr = vis_all * rel * loc * zon[:, None, :].float()
    nr = Y // 5
    U = torch.zeros((NL, nr, Xs), device=dev)
    for i, f in enumerate(fib):
        yr = yrow[i].float()
        lo = ((yr - f["w"] / 2) / 5).floor().long().clamp(0, nr - 1); hi = ((yr + f["w"] / 2) / 5).floor().long().clamp(0, nr - 1)
        for b in range(int((hi - lo).max()) + 1):
            U.scatter_add_(1, (lo + b).clamp(max=nr - 1)[:, None, :], (-cr[i] * ((lo + b) <= hi).float())[:, None, :])
    Bc = Rb[:, :nr * 5].reshape(NL, nr, 5, Xs).mean(2)
    Abx = torch.as_tensor(A[:nr * 5].reshape(nr, 5, Xs).mean(1) >= 0.8, device=dev)
    Okx = torch.as_tensor(OKb[:nr * 5].reshape(nr, 5, Xs).mean(1) >= 0.5, device=dev)
    if Abx.any():
        thr = cfg["void_frac"] * float(Bc[om][Abx].median())
        U += cfg["fl_wv"] * (Bc < thr).float()
    U += cfg["fl_wf"] * torch.as_tensor(np.abs(labs) * step, device=dev, dtype=torch.float32)[:, None, None]
    U[~lvt] += 5.0
    U = torch.where(Okx[None], U, torch.zeros_like(U))
    lab0 = torch.full((nr, Xs), int(om), device=dev, dtype=torch.long)
    Sg = SS.sgm(SS._pin(U, Abx, lab0), int(cfg["fl_D"]), float(cfg["fl_P1"]))
    o_box = SS._argmin_sub(Sg, labs.astype(float) * step)
    o_px = F.interpolate(o_box[None, None], size=(nr * 5, Xs * S), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    out = np.zeros((Y, X)); out[:nr * 5, :Xs * S] = o_px
    out = np.where(det["anc_st"], 0.0, out)
    # the record: per given fibre, the share of the zone's columns where it is visible on the decided surface
    ob = o_box.cpu().numpy(); vn = vis_all.cpu().numpy(); fol = []
    for i, f in enumerate(fib):
        zcol = Z[f["y"]]
        if zcol.sum() >= 5:
            li = np.clip(np.round(ob[min(f["y"] // 5, nr - 1)] / step).astype(int) + om, 0, NL - 1)
            fol.append(float(np.mean(vn[i, li, np.arange(Xs)][zcol])))
    Zx = torch.as_tensor(Z[:nr * 5].reshape(nr, 5, Xs).mean(1) >= 0.5, device=dev)
    return out, dict(fibres=nF, widths=[int(f["w"]) for f in fib], vis_ctx=[f["vis_ctx"] for f in fib],
                     visible_in_zone=[round(x, 2) for x in fol],
                     o_zone_p50=round(float(o_box[Zx].median()), 2) if Zx.any() else None,
                     o_zone_abs_p90=round(float(torch.quantile(o_box[Zx].abs(), 0.9)), 2) if Zx.any() else None)


def _px2cells(o, det, ga, Hh, Ww, cfg):
    """a straightened-px field of a read -> frame cells of the grid (rows ga .., the read's columns): the mean of
    the px in each cell (px 0 IS cell 0, align corners). Returns (grid, px count)."""
    Y_, X_ = o.shape; b_ = det["b"]; ra_ = det["ra"]
    yy, xx = np.meshgrid(np.arange(Y_), np.arange(X_), indexing="ij")
    q_ = float(cfg.get("px2cell", 0.4))
    rcell = np.round(ra_ - ga + (yy + b_ * (xx - X_ / 2.0)) / 5.0 - q_).astype(int); ccell = np.round(xx / 5.0 - q_).astype(int)
    okm = (rcell >= 0) & (rcell < Hh) & (ccell >= 0) & (ccell < Ww)
    lin = rcell[okm] * Ww + ccell[okm]                                 # (summed in the px order, as np.add.at)
    acc = np.bincount(lin, weights=o[okm], minlength=Hh * Ww).reshape(Hh, Ww)
    cnt = np.bincount(lin, minlength=Hh * Ww).reshape(Hh, Ww).astype(np.float64)
    return acc / np.maximum(cnt, 1e-9), cnt


def fibre_field_id(P, v, rows, cols, target, ctxg, zoneg, cfg):
    """step 2 on the window rows x cols (grid rows r0 - rpad .. r1 + rpad): ctxg / zoneg the known / area cells on the
    grid. Returns dict(C = the move per grid cell (vox, along the normals), ...)."""
    r0, r1 = rows; c0, c1 = cols
    pad = cfg["rpad"]; ga = r0 - pad
    Hh, Ww = (r1 + pad) - ga, c1 - c0
    tr = int(cfg["tile_rows"])
    Opat = np.zeros((Hh, Ww)); npat = 0; idinfo = None
    if cfg.get("follow"):                                                 # THE DECISION: the surface the known cells' fibres go on
        N0, nk0 = qc.normals(P, v)
        for a in range(r0, r1, tr):
            b = min(a + tr, r1)
            det = HW.detect(P, v, (a, b, c0, c1), ANCHOR=ctxg[(a - pad) - ga:(b + pad) - ga], **dict(cfg["ident"], MAXF=1))
            if "ist" not in det:
                idinfo = dict(reason="the identity read returned no stack", note=det.get("summary", {}).get("note"))
                continue
            o, idrec = render_follow(det, cfg)
            if o is None:
                idinfo = idrec
                continue
            idinfo = idrec
            # straightened px -> frame cells: row px y (straightened) at column x is the frame row ra + (y + b (x - X/2)) / 5
            go_, cnt = _px2cells(o, det, ga, Hh, Ww, cfg)
            Opat = np.where(cnt > 0, go_, Opat); npat += 1
        Opat = np.where(zoneg, Opat, 0.0)
        Of = np.zeros(v.shape); a0 = max(ga, 0); n_ = min(r1 + pad, v.shape[0]) - a0
        Of[a0:a0 + n_, c0:c1] = Opat[a0 - ga:a0 - ga + n_]
        P = np.where(Of[..., None] != 0, P + Of[..., None] * np.nan_to_num(N0), P)
    fibres, stacks = [], []
    for a in range(r0, r1, tr):
        b = min(a + tr, r1)
        anc = ctxg[(a - pad) - ga:(b + pad) - ga]
        det = HW.detect(P, v, (a, b, c0, c1), ANCHOR=anc, **dict(cfg["ident"], **(cfg.get("place") or {})))
        fibres += det["fibres"]
        if "stack" in det:
            stacks.append(det)
    N_, _ = qc.normals(P, v)
    bump = HW.surface_bumps(P, v, N_, (ga, r1 + pad), (c0, c1), cfg["bump_sig"]) if cfg["bump_sig"] > 0 else np.zeros((Hh, Ww))
    # per fibre: its nodes on the grid, known / area membership, heights over the smoothed frame
    for f in fibres:
        ri = np.clip(np.round(0.5 * (f["top"] + f["bot"])).astype(int) - ga, 0, Hh - 1); ci = np.clip(np.round(f["cols"]).astype(int) - c0, 0, Ww - 1)
        f["_ri"], f["_ci"] = ri, ci
        f["_h"] = f["depth"] - target + bump[ri, ci]
        f["_nctx"] = int(ctxg[ri, ci].sum()); f["_nzone"] = int(zoneg[ri, ci].sum())
    ident = [f for f in fibres if f["_nctx"] >= cfg["idc"] and f["_nzone"] >= cfg["idz"]]

    def vote(fs):
        SW = np.zeros((Hh, Ww)); SD = np.zeros((Hh, Ww))
        cb = np.median([f["cbest_med"] for f in fs]) if fs else 1.0
        for f in fs:
            wf = float(np.clip(f["cbest_med"] / max(cb, 1e-9), 0.2, 2.0))
            for c_, tp, bt, h in zip(f["cols"], f["top"], f["bot"], f["_h"]):
                ci = int(round(c_ - c0))
                if not (0 <= ci < Ww):
                    continue
                a_, b_ = max(int(np.ceil(tp)) - ga, 0), min(int(np.floor(bt)) - ga, Hh - 1)
                if b_ >= a_:
                    SW[a_:b_ + 1, ci] += wf; SD[a_:b_ + 1, ci] += wf * h
        return SW, SD
    SWi, SDi = vote(ident)
    extra = []
    if SWi.sum() > 0:                                                     # the identity layer (wide), for area-only fibres
        sw4 = (4 * cfg["sr"], 4 * cfg["sc"])
        Zw = ndimage.gaussian_filter(SWi, sw4)
        Zi = ndimage.gaussian_filter(SDi, sw4) / np.maximum(Zw, 1e-9)
        idset = {id(f) for f in ident}
        for f in fibres:
            if id(f) in idset or f["_nzone"] < cfg["idz"]:
                continue
            m = Zw[f["_ri"], f["_ci"]] > 1e-3
            if m.mean() > 0.5 and abs(float(np.median((f["_h"] - Zi[f["_ri"], f["_ci"]])[m]))) <= cfg["zone_tol"]:
                extra.append(f)
    kept = ident + extra
    SW, SD = vote(kept)
    Z, Swt = HW.correction(SW, SD, 0.0, sr=cfg["sr"], sc=cfg["sc"])
    C = np.clip(Z - bump, -cfg["cmax"], cfg["cmax"])
    # the VOID GUARD: never onto the dark gap
    nguard = 0
    for det in stacks:
        Bc, offs, ra = det["stack_cells"], det["offs"], det["ra"]        # the stack per cell, per layer
        K, nr, nc = Bc.shape
        Bc = ndimage.uniform_filter1d(Bc, size=max(1, int(round(2.0 / (offs[1] - offs[0]))) + 1), axis=0, mode="nearest")
        g0 = ra - ga                                                      # grid row of the stack's first cell row
        nr = min(nr, Hh - g0); nc = min(nc, Ww)
        sl = (slice(g0, g0 + nr), slice(0, nc))
        k0 = int(np.argmin(np.abs(offs)))
        cg = ctxg[sl]
        if cg.sum() < 10:
            continue
        thr = cfg["void_frac"] * float(np.median(Bc[k0, :nr, :nc][cg]))
        zg = zoneg[sl]; Cw = C[sl].copy()
        step = offs[1] - offs[0]
        ii, jj = np.nonzero(zg)                                           # every zone cell at once: on the void at
        o = Cw[ii, jj]                                                    # its move, it is pulled back toward 0 to
        kk = np.clip(np.round((o - offs[0]) / step), 0, K - 1).astype(int)   # the first step that is not (0.5 vox
        bad = Bc[kk, ii, jj] < thr                                           # steps, np.linspace's own values)
        if bad.any():
            ii, jj, o = ii[bad], jj[bad], o[bad]
            m = np.maximum((np.abs(o) / 0.5).astype(int), 1) + 1
            J = np.arange(int(m.max()))[None, :]
            o2 = J * ((0.0 - o) / (m - 1))[:, None] + o[:, None]
            o2 = np.where(J == (m - 1)[:, None], 0.0, o2)
            k2 = np.clip(np.round((o2 - offs[0]) / step), 0, K - 1).astype(int)
            hit = (Bc[k2, ii[:, None], jj[:, None]] >= thr) & (J < m[:, None])
            got = hit.any(1)
            first = hit.argmax(1)
            Cw[ii[got], jj[got]] = o2[got, first[got]]
            nguard += int(got.sum())
        C[sl] = Cw
    C = C + Opat                                                          # the move relative to the frame given
    return dict(C=C, Opat=Opat, Swt=Swt, fibres=fibres, ident=ident, extra=extra, ga=ga, bump=bump, guarded=nguard,
                follow=dict(tiles=npat, rec=idinfo, p50=round(float(np.median(Opat[zoneg])), 2) if zoneg.any() else None,
                            abs_p90=round(float(np.percentile(np.abs(Opat[zoneg]), 90)), 2) if zoneg.any() else None))
