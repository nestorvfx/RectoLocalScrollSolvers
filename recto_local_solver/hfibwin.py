"""THE THICK HORIZONTAL RECTO FIBRES OF A WINDOW: the detector and the stack read.
detect(P, v, region, **cfg)  ->  the stack read (the CT along P's normals, KLO..KHI vox, straightened along the fibres'
    common drift and cleaned along them in 3-D) and the 3-D tow tracker (fibres over row, depth and
    width; the recto ply from the known cells' ANCHOR): the fibre step reads its decision from the stack and PLACES the
    bump with the tracker (the fibres within -3..+4 vox of the decided surface).
render_detect / render_detect_many / render_measures / visible_nodes  ->  the fibres VISIBLE on a render (its layers
    -1, 0, +1 vox): the weaker-edge tow filter, whole-window Viterbi paths, both edges refined, overlaps resolved,
    shared boundaries; per column, VISIBLE = both edges clear of the darkest rows within VFL px (CLARITY >= VTAU noise)
    and the band smooth along the fibre (mean |d/dx| <= HXT the typical), median over +-VSM columns: a vertical fibre
    on top, a break or an artefact is not visible there. A fibre counts if visible in >= VMIN of its columns. The
    decision (hsolve.render_follow) runs exactly these mechanics on every candidate surface at once.
correction(SW, SD, target, ...)  ->  the move along the normals (vox, per cell) that puts the fibres' centres at
    `target` (A = 0.0: the surface THROUGH the fibre centres), from the fibres' votes. surface_bumps: the surface's own
    bumps (subtracted before the move)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import hfib
from . import volume as qc

TARGETS = {"A": 0.0}                                                # the surface through the fibre centres
DCFG = dict(PAD=12, KLO=-10.0, KHI=10.0, STEP=0.5, UP=5, S=4, DSEL=(-6.0, 6.0), DREL=(-3.5, 2.5),
            FLK=30, FST=20.0, FSMAX=0.08, WIDTHS=(7, 9, 11, 13, 15, 19, 23, 27, 31), KCORE=2, KFL=(3, 6),
            FW=4, BETA=0.5, GAMMA=0.5, ETOL=1, ESIG=1.0, LAMS=(0.3, 0.3, 0.5), RY=1, XB=4, CLIP=1.0, TAU=0.5, KEEP=1.0,
            SMB=9, GAPB=12, MINB=30, MAXF=16, MARGIN=1, KWIN=1.0, DSM=1, EPAD=4.0, LAM=0.5, DLAM=0.15, DMS=4,
            JOINT=True, TOUCH=2.0, JMIN=6, MINLEN=100, THICK=7.0, RK=1, KPATH=None, NORM_BAND=4.0,
            PLYLIM=(-6.0, 6.0),                                       # the context's ply searched only here
            VFL=3, VTAU=1.5, HXT=1.25, VSM=2, VMIN=0.35)              # per-column visibility (render_detect)


def detect(P, v, region, ANCHOR, log=None, **cfg):
    """the fibres of window `region` = (r0, r1, c0, c1) (cells) read from surface P [R, C, 3] (valid v).
    ANCHOR (bool grid, rows r0 - PAD .. r1 + PAD, cols c0 .. c1; the solver's context cells): the recto ply is found
    from the context's pixels only, and in the context's cells a fibre may lie only on that ply (DREL around it); in
    the other cells it may lie anywhere in DSEL (set wide, e.g. +-20 vox: the volume the solution can occupy), moving
    up to RK layers per knot: the context's fibres are FOLLOWED into the new area as one path each, whatever layer the
    frame happens to be on there. Also returned then: the stack (stack, offs, ra) for the solver's void guard."""
    import torch
    from .hfib import straighten
    c = dict(DCFG, **cfg)
    r0, r1, c0, c1 = region
    P = np.asarray(P, np.float64)
    N, nok = qc.normals(P, v); vv = v & nok
    S0, okt, offs = qc.stack(P, vv, N, region, c["KLO"], c["KHI"], c["STEP"], up=c["UP"], pad_rows=c["PAD"], as_torch=True)
    ok = okt.cpu().numpy()
    ra = max(r0 - c["PAD"], 0); S = c["S"]; UP = c["UP"]
    offs = np.asarray(offs, float)
    if ok.sum() < 100:
        return dict(fibres=[], summary=dict(region=list(region), fibres=0, note="no valid data"))
    lo, hi = _percentiles(S0[:, okt], (1.0, 99.5))                    # the stack normalised on the GPU (float64 as numpy)
    It = ((S0.double() - lo) / max(hi - lo, 1e-6)).clamp(0, 1).float()
    It[:, ~okt] = 0.0
    del S0
    dev = qc.DEV
    DSEL = list(c["DSEL"])
    A0 = np.asarray(ANCHOR, bool); o_ = ra - (r0 - c["PAD"])              # the context's cells -> stack pixels
    Ypx, Xpx = ok.shape
    ry_ = np.clip(o_ + np.arange(Ypx) // UP, 0, A0.shape[0] - 1); rx_ = np.clip(np.arange(Xpx) // UP, 0, A0.shape[1] - 1)
    anc = A0[ry_][:, rx_] & ok
    msk = torch.as_tensor(anc, device=dev) if anc.sum() > 200 else okt  # the recto ply: most horizontal-fibre energy
    hz = []                                                               # in the context, within PLYLIM (the context's
    for k_ in range(0, It.shape[0], 8):                                   # surface is right: its ply is near 0), 8 layers at once
        A_ = hfib.gauss_blur(It[k_:k_ + 8], 0.0, 8.0); Hy = torch.zeros_like(A_); Hy[:, 1:-1] = A_[:, 2:] - 2 * A_[:, 1:-1] + A_[:, :-2]
        hz.append(hfib.gauss_blur(Hy.abs(), 3.0, 0.0)[:, msk].mean(1))
    hz = torch.cat(hz).double().cpu().numpy()
    cand = np.nonzero((offs >= c["PLYLIM"][0]) & (offs <= c["PLYLIM"][1]))[0]
    KPLY = float(offs[cand[int(np.argmax(hz[cand]))]])
    DLO, DHI = KPLY - 3.0, KPLY
    near = torch.as_tensor(np.abs(offs - (KPLY - 1.0)) <= 1.0, device=dev)
    b = hfib.slope(It[near].mean(0))
    Ist = straighten(It, b)
    K, Y, X = Ist.shape; Xs = X // S
    Ib = Ist[..., :Xs * S].reshape(K, Y, Xs, S).mean(-1)
    sel = torch.as_tensor((offs >= DLO) & (offs <= DHI), device=dev)
    slab = Ist[sel].mean(0)
    sl, _ = hfib.flow_field(slab, st=c["FST"], smax=c["FSMAX"])
    okst = straighten(torch.as_tensor(ok.astype(np.float32), device=dev)[None], b)[0].cpu().numpy()
    okcol = okst.mean(0) > 0.5
    okb = okcol[:Xs * S].reshape(Xs, S).mean(1) > 0.5
    sk3 = hfib.depth_slope_field(Ist, st=20.0, smax=0.8)                 # the line integral along the fibres, in 3-D
    Ism = hfib.flow_median3d(Ist, sl, sk3, L=c["FLK"])
    nr, nc = Y // UP, X // UP                                            # the stack per cell (the void guard's input)
    cells_ = It[:, :nr * UP, :nc * UP].reshape(K, nr, UP, nc, UP).mean((2, 4)).cpu().numpy()
    stk = dict(stack_cells=cells_, offs=offs, ra=ra, ok=ok, ist=Ism.cpu().numpy(), b=b, kply=KPLY, S=S, Ism_t=Ism, Ist_t=Ist,
               ok_st=okst > 0.5, Ib=Ib)
    stk["anc_st"] = straighten(torch.as_tensor(anc.astype(np.float32), device=dev)[None], b)[0].cpu().numpy() > 0.5
    Ibs = Ism[..., :Xs * S].reshape(K, Y, Xs, S).mean(-1)
    WID = list(c["WIDTHS"])
    Ev, Rc = hfib.tow_evidence(Ibs, WID, kcore=c["KCORE"], kfl=tuple(c["KFL"]), fw=c["FW"], beta=c["BETA"], mode="koller",
                               gamma=c["GAMMA"], etol=c["ETOL"], esig=c["ESIG"])
    ksel = np.nonzero((offs >= DSEL[0]) & (offs <= DSEL[1]))[0]
    okbt = torch.as_tensor(okb, device=dev)
    if okb.sum() < 4 or len(ksel) < 3:
        return dict(fibres=[], summary=dict(region=list(region), fibres=0, note="too few valid columns"))
    knb = np.nonzero(np.abs(offs - KPLY) <= c["NORM_BAND"])[0]                                # normalised on the ply's band
    ref = Ev[len(WID) // 2][knb][:, :, okbt]
    med = ref.median(); sig = 1.4826 * (ref - med).abs().median()
    Ev = (Ev - med) / sig.clamp_min(1e-9)
    XB = c["XB"]; Xc = Xs // XB
    Evc = Ev[..., :Xc * XB].reshape(Ev.shape[0], Ev.shape[1], Ev.shape[2], Xc, XB).mean(-1) if XB > 1 else Ev
    okc = okb[:Xc * XB].reshape(Xc, XB).mean(1) > 0.5 if XB > 1 else okb
    At = straighten(torch.as_tensor(anc.astype(np.float32), device=dev)[None], b)[0]   # in the context's cells: the
    Ab = At[..., :Xs * S].reshape(Y, Xs, S).mean(-1)                                     # context's ply only
    Ac = (Ab[..., :Xc * XB].reshape(Y, Xc, XB).mean(-1) if XB > 1 else Ab) > 0.5          # [Y, Xc]
    okk = torch.as_tensor((offs[ksel] >= KPLY + c["DREL"][0]) & (offs[ksel] <= KPLY + c["DREL"][1]), device=dev)
    allow = (~Ac)[None] | okk[:, None, None]
    runs = hfib.tow_extract(Evc, WID, ksel, okc, lam=tuple(c["LAMS"]), clip=c["CLIP"], tau=c["TAU"], keep_tau=c["KEEP"],
                            smooth=max(c["SMB"] // XB, 1), gap=max(c["GAPB"] // XB, 0), min_len=max(c["MINB"] // XB, 1),
                            max_fibres=c["MAXF"], margin=c["MARGIN"], ry=c["RY"], rk=c["RK"], allow=allow)
    Ismn = stk["ist"]; tracks = []
    Wa = np.asarray(WID, float)
    for r in runs:
        if XB > 1:
            tc = np.arange(r["t0"], r["t1"] + 1) * XB + 0.5 * (XB - 1)
            tf = np.arange(r["t0"] * XB, min((r["t1"] + 1) * XB, Xs))
            r["y"] = np.interp(tf, tc, r["y"]); r["k"] = np.interp(tf, tc, r["k"]); r["w"] = np.interp(tf, tc, r["w"])
            r["t0"], r["t1"] = int(tf[0]), int(tf[-1])
        tt = np.arange(r["t0"], r["t1"] + 1)
        wi = np.array([int(np.argmin(np.abs(Wa - w))) for w in r["w"]]); yi = np.clip(np.round(r["y"]).astype(int), 0, Y - 1)
        ki = np.clip(np.round(r["k"]).astype(int), 0, K - 1)
        ix = [torch.as_tensor(a_, device=dev) for a_ in (wi, ki, yi, tt)]  # (read on the GPU: the volumes stay there)
        r["e"] = Ev[ix[0], ix[1], ix[2], ix[3]].cpu().numpy()
        r["rc"] = Rc[ix[0], :, ix[2], ix[3]].cpu().numpy()
    # every fibre's edges in one pass
    nds = hfib.tow_refine_batch(Ismn, offs, runs, S, Y, kwin=c["KWIN"], dsm=c["DSM"], edge_pad=c["EPAD"], lam=c["LAM"], dsel=DSEL,
                                dlam=c["DLAM"] if c["DLAM"] > 0 else None, dms=c["DMS"], kpath=c["KPATH"])
    for r, nd in zip(runs, nds):
        for q in nd:
            q["score"] = r["score"]; q["order"] = r["order"]
        if len(nd) >= 2:
            tracks.append(nd)
    tracks, _ = hfib.resolve_overlaps(tracks)
    if c["JOINT"]:
        tracks, _ = hfib.joint_boundaries(Ismn, offs, tracks, S, touch=c["TOUCH"], minrun=c["JMIN"], lam=c["LAM"])
    tracks = hfib.fibre_depth(Ib.cpu().numpy(), offs, tracks, dmin=DSEL[0], dmax=DSEL[1])
    tab = hfib.tracks_table(tracks, b, X, S, min_len_px=c["MINLEN"], half=False)
    cells = lambda y: ra + y / UP
    fib = []
    for f in tab:
        fib.append(dict(cols=(c0 + f["x"] / UP), top=cells(f["top"]), bot=cells(f["bot"]), depth=f["d"].copy(), width_px=f["w"].copy(),
                        vis=np.nan_to_num(f["vis"]), cbest=np.nan_to_num(f["cbest"]), length_cells=f["length"] / UP,
                        width_med_px=float(f["wmed"]), depth_med=float(f["dmed"]), score=f.get("score"), order=f.get("order"),
                        cbest_med=float(np.nanmedian(f["cbest"])), vis_med=float(np.nanmedian(f["vis"])),
                        thick=bool(f["wmed"] >= c["THICK"])))
    summ = dict(region=list(region), kply=KPLY, dsel=DSEL, slope=b, fibres=len(fib),
                depth_med=float(np.median(np.concatenate([f["depth"] for f in fib]))) if fib else None,
                vis_med=float(np.median(np.concatenate([f["vis"] for f in fib]))) if fib else None)
    if log:
        log("detect %s: %d fibres, ply %+.1f, depth med %s" % (region, len(fib), KPLY, None if summ["depth_med"] is None else round(summ["depth_med"], 2)))
    return dict(fibres=fib, summary=summ, **stk)


def _percentiles(x, qs):
    """np.percentile(x, qs) (method "linear") of a GPU tensor, the same values: the order statistics around each
    quantile from ONE sort, then numpy's own interpolation (_lerp: float32 neighbours, float64 weight)."""
    import torch
    x = x.reshape(-1)
    n = x.numel()
    xs = torch.sort(x).values
    hs = [(n - 1) * (q / 100.0) for q in qs]
    js = [min(int(np.floor(h)), n - 1) for h in hs]
    v = torch.stack([xs[j] for j in js] + [xs[min(j + 1, n - 1)] for j in js]).cpu().numpy().astype(np.float32)
    out = []
    for i, (h, j) in enumerate(zip(hs, js)):
        a, b = v[i], v[len(js) + i]
        g = np.float64(h - j)
        d = b - a
        out.append(float(b - d * (1 - g)) if g >= 0.5 else float(a + d * g))
    return out


def render_detect(Cm, Rr, S, okb, c, okm=None):
    """THE DETECTOR ON ONE RENDER (traces the fibres VISIBLE on the surface, side to side, and
    marks where they show): Cm [Y, X] the render cleaned along the fibres (the 3-D streamline median at the render's
    layers -1 / 0 / +1 vox), Rr [Y, X] the same render as read (torch; straightened, fibres along x), S px per column
    block, okb [Xs] valid blocks, c = DCFG. The tow tracker on the render only (the two-edge cross-section filter, one
    layer), whole-window paths over (row, width), the edges as smooth optimal lines, shared boundaries; then every
    node's VISIBILITY on Rr (visible_nodes). Returns the tracks (lists of nodes: x block, top / bot / yc rows, w, score,
    clar, hx, vis_on)."""
    return render_detect_many([(Cm, Rr, okb, okm)], S, c)[0]


def _rd_evidence(Cm, S, okb, c):
    """render_detect's evidence on one cleaned render: the two-edge filter, normalised, averaged to the knots."""
    import torch
    Y, X = Cm.shape; Xs = X // S
    Ibs = Cm[:, :Xs * S].reshape(Y, Xs, S).mean(-1)[None]
    WID = list(c["WIDTHS"])
    Ev, Rc = hfib.tow_evidence(Ibs, WID, kcore=0, kfl=tuple(c["KFL"]), fw=c["FW"], beta=0.0, mode="koller", gamma=c["GAMMA"],
                               etol=c["ETOL"], esig=c["ESIG"])
    okbt = torch.as_tensor(np.asarray(okb, bool), device=Cm.device)
    ref = Ev[len(WID) // 2][0][:, okbt]
    med = ref.median(); sig = 1.4826 * (ref - med).abs().median()
    Ev = (Ev - med) / sig.clamp_min(1e-9)
    XB = c["XB"]; Xc = Xs // XB
    Evc = Ev[..., :Xc * XB].reshape(Ev.shape[0], 1, Y, Xc, XB).mean(-1)
    okc = np.asarray(okb, bool)[:Xc * XB].reshape(Xc, XB).mean(1) > 0.5
    return dict(Evn=Ev.cpu().numpy(), Rcn=Rc.cpu().numpy(), Evc=Evc, okc=okc, Y=Y, Xs=Xs)


def render_detect_many(items, S, c):
    """render_detect for SEVERAL renders at once (items: list of (Cm, Rr, okb, okm), the same shape; e.g. the renders
    of many candidate surfaces): the evidence per render, the fibre extraction in batched
    Viterbi rounds (hfib.tow_extract_many), every fibre's edges of every render in one pass (hfib.tow_refine_groups),
    the rest per render. Render for render the same tracks as render_detect alone (one code path for both)."""
    WID = list(c["WIDTHS"]); Wa = np.asarray(WID, float); XB = c["XB"]; o0 = np.array([0.0])
    evs = [_rd_evidence(Cm, S, okb, c) for (Cm, Rr, okb, okm) in items]
    runs_all = hfib.tow_extract_many([e["Evc"] for e in evs], WID, np.array([0]), [e["okc"] for e in evs], lam=tuple(c["LAMS"]),
                                     clip=c["CLIP"], tau=c["TAU"], keep_tau=c["KEEP"], smooth=max(c["SMB"] // XB, 1),
                                     gap=max(c["GAPB"] // XB, 0), min_len=max(c["MINB"] // XB, 1), max_fibres=c["MAXF"],
                                     margin=c["MARGIN"], ry=c["RY"])
    groups, Cns = [], []
    for (Cm, Rr, okb, okm), e, runs in zip(items, evs, runs_all):
        Y, Xs, Evn, Rcn = e["Y"], e["Xs"], e["Evn"], e["Rcn"]
        for r in runs:
            if XB > 1:
                tc = np.arange(r["t0"], r["t1"] + 1) * XB + 0.5 * (XB - 1)
                tf = np.arange(r["t0"] * XB, min((r["t1"] + 1) * XB, Xs))
                r["y"] = np.interp(tf, tc, r["y"]); r["k"] = np.interp(tf, tc, r["k"]); r["w"] = np.interp(tf, tc, r["w"])
                r["t0"], r["t1"] = int(tf[0]), int(tf[-1])
            tt = np.arange(r["t0"], r["t1"] + 1)
            wi = np.array([int(np.argmin(np.abs(Wa - w))) for w in r["w"]]); yi = np.clip(np.round(r["y"]).astype(int), 0, Y - 1)
            r["e"] = Evn[wi, 0, yi, tt]
            r["rc"] = np.stack([Rcn[a_, :, cc_, t] for a_, cc_, t in zip(wi, yi, tt)])
        Cn = Cm[None].cpu().numpy(); Cns.append(Cn)
        groups.append((Cn, o0, runs, S, Y))
    # every fibre's edges of every render in one pass
    ekw = dict(kwin=c["KWIN"], dsm=c["DSM"], lam=c["LAM"], dsel=[-0.5, 0.5], dlam=None, dms=c["DMS"], gsy=1.0, gsx=2.0)
    nds_all = hfib.tow_refine_groups(groups, edge_pad=c["EPAD"], **ekw)
    out = []
    for (Cm, Rr, okb, okm), Cn, runs, nds in zip(items, Cns, runs_all, nds_all):
        tracks = []
        for r, nd in zip(runs, nds):
            for q in nd:
                q["score"] = r["score"]; q["order"] = r["order"]
            if len(nd) >= 2:
                tracks.append(nd)
        tracks, _ = hfib.resolve_overlaps(tracks)
        if c["JOINT"]:
            tracks, _ = hfib.joint_boundaries(Cn, o0, tracks, S, touch=c["TOUCH"], minrun=c["JMIN"], lam=c["LAM"])
        visible_nodes(tracks, Rr.cpu().numpy(), S, okb, c, okm=okm)
        out.append(tracks)
    return out


def render_measures(Rr, S, okb, okm=None):
    """the per-render quantities of the visibility test: Rb [Y, Xs] the render block-averaged, its noise (row-to-row MAD
    / sqrt 2), Gx [Y, X] the |d/dx| along the fibres, gx_typ its median, both over the VALID pixels only (okm [Y, X]
    bool, the straightened data mask; None: the valid column blocks; empty rows must not enter the normalisation)."""
    Y, X = Rr.shape; Xs = X // S
    okb = np.asarray(okb, bool)
    Rb = Rr[:, :Xs * S].reshape(Y, Xs, S).mean(-1)
    if okm is None:
        okm = np.repeat(np.repeat(okb[None], Y, 0), S, 1)
        okm = np.pad(okm, ((0, 0), (0, X - okm.shape[1])))
    okm = np.asarray(okm, bool)
    okmb = okm[:, :Xs * S].reshape(Y, Xs, S).mean(-1) > 0.5
    pair = okmb[1:] & okmb[:-1] & okb[None]
    dif = np.abs(np.diff(Rb, axis=0))[pair]
    noise = max(1.4826 * float(np.median(dif)) / np.sqrt(2.0), 1e-6) if dif.size else 1e-6
    Gx = np.zeros_like(Rr); Gx[:, 1:-1] = 0.5 * np.abs(Rr[:, 2:] - Rr[:, :-2])
    gm = okm.copy(); gm[:, 1:-1] &= okm[:, 2:] & okm[:, :-2]; gm[:, [0, -1]] = False
    gx_typ = max(float(np.median(Gx[gm])), 1e-6) if gm.any() else 1e-6
    return Rb, noise, Gx, gx_typ


def visible_nodes(tracks, Rr, S, okb, c, okm=None):
    """VISIBLE where the fibre shows on the render as read (Rr [Y, X] numpy): per node (column block) CLARITY = the
    weaker of its two edge contrasts (the band's mean minus the darkest row just outside that edge, VFL rows, the edge
    row included) / the render's noise, and HX = the band's mean |d/dx| over the block +-2 px / the window's typical;
    both median-smoothed over +-VSM nodes along the fibre; visible if CLARITY >= VTAU and HX <= HXT. Writes clar, hx,
    vis_on into the nodes."""
    from scipy.ndimage import median_filter
    Y, X = Rr.shape; Xs = X // S
    Rb, noise, Gx, gx_typ = render_measures(Rr, S, okb, okm)
    okb = np.asarray(okb, bool)
    for tr in tracks:
        for n in tr:
            x_ = min(int(n["x"]), Xs - 1); t0, t1 = max(int(np.floor(n["top"])), 0), min(int(np.ceil(n["bot"])), Y - 1)
            band = Rb[t0:t1 + 1, x_].mean() if t1 >= t0 else np.nan
            up_ = Rb[max(t0 - c["VFL"], 0):t0 + 1, x_]; dn_ = Rb[t1:min(t1 + c["VFL"] + 1, Y), x_]
            cl = min(band - up_.min(), band - dn_.min()) / noise if (len(up_) and len(dn_) and np.isfinite(band)) else 0.0
            n["clar"] = float(cl) if okb[x_] else 0.0
            ca, cb = max(x_ * S - 2, 0), min((x_ + 1) * S + 2, X)
            n["hx"] = float(Gx[t0:t1 + 1, ca:cb].mean() / gx_typ) if t1 >= t0 else 9.0
        o_ = np.argsort([n["x"] for n in tr])
        cl_ = median_filter(np.array([tr[i]["clar"] for i in o_]), size=2 * c["VSM"] + 1, mode="nearest")
        hx_ = median_filter(np.array([tr[i]["hx"] for i in o_]), size=2 * c["VSM"] + 1, mode="nearest")
        for j, i in enumerate(o_):
            tr[i]["vis_on"] = float(cl_[j] >= c["VTAU"] and hx_[j] <= c["HXT"])
    return tracks


def correction(SW, SD, target, sr=2.0, sc=4.0):
    """the move t = depth - target per cell from the fibres' votes (SW the weights, SD the weighted depths per cell): a
    normalised Gaussian fill (sr x sc cells), and a 3x wider one where the support is thin. Returns (C, SW)."""
    T0 = np.where(SW > 0, SD / np.maximum(SW, 1e-9), 0.0) - target
    ST = T0 * SW
    num = ndimage.gaussian_filter(ST, (sr, sc)); den = ndimage.gaussian_filter(SW, (sr, sc))
    num3 = ndimage.gaussian_filter(ST, (3 * sr, 3 * sc)); den3 = ndimage.gaussian_filter(SW, (3 * sr, 3 * sc))
    a_ = np.clip(den / max(np.percentile(den[den > 0], 50), 1e-9), 0, 1) if (den > 0).any() else np.zeros_like(den)
    C = a_ * num / np.maximum(den, 1e-9) + (1 - a_) * num3 / np.maximum(den3, 1e-9)
    return np.nan_to_num(C), SW


def surface_bumps(P, v, N, rows, cols, sig=3.0):
    """the surface's deviation from its Gaussian-smoothed self (sig cells, valid cells only) along its normals, on the
    grid rows x cols (rows may start above 0: padded with 0)."""
    r0, r1 = rows; c0, c1 = cols
    a, b = max(r0, 0), min(r1, v.shape[0])
    Pw = P[a:b, c0:c1]; vw = v[a:b, c0:c1].astype(float)
    den = ndimage.gaussian_filter(vw, sig)
    Ps = np.stack([ndimage.gaussian_filter(np.where(vw > 0, Pw[..., k], 0.0), sig) for k in range(3)], -1) / np.maximum(den, 1e-6)[..., None]
    h = np.where(vw > 0, ((Pw - Ps) * np.nan_to_num(N[a:b, c0:c1])).sum(-1), 0.0)
    out = np.zeros((r1 - r0, c1 - c0)); out[a - r0:a - r0 + (b - a)] = h
    return out
