"""SHEETSOLVE: the patch solver of the frame (no ink, no ML model; the known cells never move).

Unknown: the depth h(cell) along the normals of a reference surface R on the grid. The objective is the sheet's own 3-D
physics, measured in ONE deep slab V(k, cell) sampled along R's normals:
  DATA   the calibrated matched filter (calibrate: Fisher LDA of the known cells' [vertical-ply, horizontal-ply,
         intensity] depth profiles against other depths) + the FIBRE FOCUS (fibre_focus: the layer where the horizontal
         and vertical fibres run on, read at the scroll's offsets foc_kH / foc_kV)
  SMOOTH coarse bins: global damped min-sum message passing (bp) with the lamination-following pairwise term
         (dip_cost: the label change the CT's own 3-D layer orientation predicts); fine cells: semi-global aggregation
         (sgm) with a hard bound on the depth change per cell and the coarse solution's slope as the expected one
  PINS   the known cells at their depth (Dirichlet): they carry the winding into the patch
Coarse-to-fine: B x B-cell bins over the full label range (the winding decision) -> per cell within +-fine_rng of the
coarse -> continuous sub-voxel refinement (per-cell parabola on the data, bending, pins; sparse LS) -> OUTER
iterations: re-sample the slab along the solved surface's own normals and solve the residual; the final surface's slant
against the laminations (the frame's slant gate).
library: solve_patch(P, v, N, region, pins, pin_val, cfg) -> dict(P, k, conf, timing); calibrate(feats, ks, cfg)"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import fclib as L
from . import kernels as KN

DEV = L.DEV
UP = 3                                      # the slab's px per cell in-plane (1.69 vox per px). The px-unit settings
PXS = UP / 5.0                              # (ridge, s_sm, dip_*_px, scan_D / dd / sp, bar_bw / along / hp / lag, the 20 px
                                            # normalisation, the 12 px high-pass) are given at 5 px per cell and scaled by
                                            # PXS: the same physical sizes in vox; the scan's slopes (vox per px) by 1 / PXS


def _sync():
    """(the stage timings: wait for the GPU)"""
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()
BIG = 1e4

CFG = dict(
    step=0.5,            # label / slab step (vox)
    kmax=16.0,           # label range +- (vox) at the coarse level (> L: the other windings are inside the search)
    ridge=(2.5, 10.0),   # elongated ridge filter (across, along) px (1 px = 1 vox at up = 5)
    s_sm=3.0,            # in-plane smoothing of the energies (px) before cell pooling
    B=5, Dc=2.0, P1c=0.08,                    # coarse: bin size (cells), hard bound (vox / bin), linear penalty / label
    Df=1.0, P1f=0.15, fine_rng=3.0,           # fine: hard bound (vox / cell), penalty, range around the coarse (vox)
    uscale=1.0,                               # unary scale
    beta=16.0, alpha=0.2, wdata=1.0,          # continuous: bending, first-order, data weight
    outer=1, outer_k=4.0,                     # outer re-sampling iterations, their label range
    L=10.5,                                   # the winding period (vox) for the margin
    calib=None,                               # the matched filter (calibrate's npz, from the sheet's known cells)
    J=16,                                     # profile half-length (steps) of the matched filter (8 vox)
    bp_iters=0, bp_damp=0.5,                  # the coarse level's belief propagation (iters 0 = 2 x size), damping
    dip_w=0.0,                                # LAMINATION FOLLOWING (0 = off): pairwise cost of a label change vs the change the
    dip_tol=0.5, dip_cap=4.0,                 # CT's own 3-D layer orientation predicts (structure tensor of the slab: the layer
    dip_sd=0.75, dip_si=2.0, dip_si_px=15.0,  # through the neighbour continues here at the measured dip); tolerance vox / cell,
    dip_gain=1.0,                             # slope gain correction (1.25-1.3 measured)
    dip_sd_px=8.0,                            # in-plane derivative scale ABOVE the fibre scale
    dip_cont=0.0,                             # cap; derivative / integration scales (vox, px); dip_cont: weight of the
)                                             # dip-steered gradient in the continuous refinement (0 = plain alpha |grad h|^2)


# ------------------------------------------------------------------------------------------------ features
def tile_feats(P, v, N, win, ks, cfg, core=None):
    """ply energies EV, EH and intensity I per (k, cell) of one window: torch [K, Hc, Wc] (normalised per cell over
    k), okc [Hc, Wc]. core = (a0, a1, b0, b1) (cells of the window): the cells the caller reads; EV / EH / I are then
    computed on the core + feat_pad cells only (the window's wide margin exists for the fibre descriptors' 300 px
    lines; the ply energies' own support is 9 cells, + 14 for the local normalisation) and are 0 outside it."""
    V, ok = L.slab(P, v, N, win, ks, up=UP)
    if float(ok.float().mean()) < 0.02:
        return None
    Vt = V; okf = ok.float()
    m = okf.bool()
    mu, sd = Vt[:, m].mean(), Vt[:, m].std().clamp(min=1e-6)
    K = Vt.shape[0]
    EV, EH, II = [], [], []
    sa, sl = cfg["ridge"][0] * PXS, cfg["ridge"][1] * PXS
    if cfg.get("dips_only"):                                          # the frame flattening needs the slopes only
        out = DIP_FN[cfg.get("dip_method", "tensor")](Vt, okf, float(ks[1] - ks[0]), cfg)
        out["okc"] = F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99
        del Vt
        return out
    Hp_, Wp_ = Vt.shape[1:]
    Hc_, Wc_ = Hp_ // UP, Wp_ // UP
    fa0, fa1, fb0, fb1 = 0, Hc_, 0, Wc_                               # the sub-window of the ply energies (cells)
    fp = int(cfg.get("feat_pad", 16))
    if core is not None and fp >= 0:
        fa0, fa1 = max(int(core[0]) - fp, 0), min(int(core[1]) + fp, Hc_)
        fb0, fb1 = max(int(core[2]) - fp, 0), min(int(core[3]) + fp, Wc_)
    sub = (slice(fa0 * UP, fa1 * UP), slice(fb0 * UP, fb1 * UP))
    okf_s = okf[sub]
    den_c = L.filt(F.avg_pool2d(okf_s[None, None], UP)[0], (20.0 * PXS / UP,) * 2)   # the 20 px local mean / rms at the CELL
                                                                      # rate (_msmooth_coarse): its denominator once per tile
    s = cfg["s_sm"] * PXS
    den_s = L.filt(okf_s, (s, s)).clamp(min=1e-4)                     # the mask's own smoothing: once per tile
    for a in range(0, K, 48):                                         # (48 depths per pass: the same values per depth)
        X = Vt[a:a + 48, sub[0], sub[1]]
        mu_l = _msmooth_coarse(X, okf_s, den_c, 20.0 * PXS)
        hp = (X - mu_l) * okf_s
        Z = hp / torch.sqrt(_msmooth_coarse(hp ** 2, okf_s, den_c, 20.0 * PXS).clamp(min=1e-6))
        eh = torch.clamp(-L.filt(Z, (sa, sl), (2, 0)), min=0)
        ev = torch.clamp(-L.filt(Z, (sl, sa), (0, 2)), min=0)
        eh = L.filt(eh * okf_s, (s, s)) / den_s; ev = L.filt(ev * okf_s, (s, s)) / den_s
        ii = L.filt((X - mu) / sd * okf_s, (s, s)) / den_s
        for lst, t in ((EV, ev), (EH, eh), (II, ii)):
            lst.append(F.avg_pool2d(t[:, None], UP)[:, 0])
    okc = F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99
    out = {}
    want_bar = cfg.get("foc_w", 0) > 0 and "_bar" in cfg
    if cfg.get("dip_w", 0) > 0:
        out.update(DIP_FN[cfg.get("dip_method", "tensor")](Vt, okf, float(ks[1] - ks[0]), dict(cfg, shear_keep=want_bar)))
    if want_bar:
        shear = None
        if "DY" in out:                                                 # the lamination height field of this tile
            shear = out.pop("SHEAR", None)
            if shear is None:                                           # (other slope estimators: integrate their slopes)
                okc_c = F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99
                shear = _dip_field(out, out["DW"], okc_c, float(ks[1] - ks[0]), float(cfg.get("shear_sig", 2.0))).float()
        out.pop("SHEAR", None)
        out["bar"] = bar_descriptors(Vt, okf, float(ks[1] - ks[0]), win, cfg, shear)
    for n, lst in (("EV", EV), ("EH", EH), ("I", II)):
        X = torch.cat(lst, 0)
        X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)
        if (fa0, fa1, fb0, fb1) != (0, Hc_, 0, Wc_):                   # back on the window's cell grid (0 outside the sub-window)
            X = F.pad(X, (fb0, Wc_ - fb1, fa0, Hc_ - fa1))
        out[n] = X
    out["okc"] = okc
    del Vt
    return out


def _msmooth_coarse(X, okf, den_c, sig):
    """the masked (normalised) Gaussian smoothing of X for a WIDE sigma (20 px = 4 cells), computed at the cell rate:
    the masked signal pooled to cells (UP x UP box), the Gaussian of sig / UP cells, divided by the (pooled, smoothed) mask, and
    upsampled bilinearly to the px grid. The pooling + interpolation add ~6 px^2 to the kernel's 400 px^2 (sigma 20 ->
    20.15 px); the field is a slowly varying local mean / rms (exact masked smoothing: 141 taps per axis at the px
    rate, ~50x the cost)."""
    num = L.filt(F.avg_pool2d((X * okf[None])[:, None], UP)[:, 0], (sig / UP, sig / UP))
    r = num / den_c.clamp(min=1e-4)
    return F.interpolate(r[:, None], size=X.shape[1:], mode="bilinear", align_corners=False)[:, 0]


def lamination_dips(Vt, okf, step, cfg):
    """the lamination dips: the structure tensor at the cell rate (_lamination_dips_fast)."""
    return _lamination_dips_fast(Vt, okf, step, cfg)


def _lamination_dips_fast(Vt, okf, step, cfg):
    """the structure tensor of the slab at the cell rate: with an in-plane derivative scale of 8 px the gradient
    fields are band-limited far below the cell rate (5 px), so (1) the slab is smoothed in-plane ONCE
    (sigma dip_sd_px), (2) the three gradients are taken at the CELL CENTRES only (in-plane: central differences over
    +-2 px of the smoothed slab = the Gaussian derivative to 1 %; along depth: the Gaussian derivative itself), (3) the
    six tensor products are formed and integrated on the cell grid (sigma dip_si_px / 5 cells; the products' band
    limit, twice the gradients', is still below the cell Nyquist rate)."""
    vpx = L.CELL / UP
    m = okf.bool()
    X = torch.where(m[None], Vt, Vt[:, m].mean() if m.any() else torch.zeros((), device=DEV))
    sdk, sik = cfg.get("dip_sd", 0.75) / step, cfg.get("dip_si", 1.5) / step
    sip, sdp = cfg.get("dip_si_px", 4.0) * PXS, cfg.get("dip_sd_px", 1.0) * PXS
    K, Hp, Wp = X.shape
    Hc, Wc = Hp // UP, Wp // UP
    c = UP // 2
    dh = max(1, int(round(2 * PXS)))                                  # the central differences' half step (2 px at UP 5)
    # the smoothed slab is read at FIVE px per cell only (the centre and +-2 px along each axis): the separable
    # smoothing evaluated just there (columns of phase 0 / 2 / 4, then rows): 0.35 of a full filter, the same values
    g = L.gkern(max(sdp, 0.5))[None]; rg = (g.shape[1] - 1) // 2
    X = F.pad(X[:, None], (rg, rg, 0, 0), mode="replicate")[:, 0]   # (padded once for the three phases)
    cols = [L.strided_corr(X, g, torch.arange(Wc) * UP + ph + rg, 2)[0] for ph in (c - dh, c, c + dh)]   # [K, Hp, Wc] each
    del X
    rows = lambda T_, ph: L.strided_corr(T_, g, torch.arange(Hc) * UP + ph, 1)[0]                     # [K, Hc, Wc]
    Sc = rows(cols[1], c)
    gy = (rows(cols[1], c + dh) - rows(cols[1], c - dh)) / (2.0 * dh * vpx)
    gx = (rows(cols[2], c) - rows(cols[0], c)) / (2.0 * dh * vpx)
    del cols
    gk = L.filt(Sc, (sdk, 0, 0), (1, 0, 0)) / step
    gy = L.filt(gy, (sdk, 0, 0)); gx = L.filt(gx, (sdk, 0, 0))
    del Sc
    sic = sip / UP
    comps = [L.filt(a_ * b_, (sik, sic, sic)) for a_, b_ in ((gk, gk), (gy, gy), (gx, gx), (gk, gy), (gk, gx), (gy, gx))]
    del gk, gy, gx
    return _dips_from_tensor(comps, okf, step, cfg)


def _dips_from_tensor(comps, okf, step, cfg):
    """(Jkk, Jyy, Jxx, Jky, Jkx, Jyx) [K, Hc, Wc] -> DY, DX (labels per cell), DW (planarity)."""
    f = L.CELL / step * cfg.get("dip_gain", 1.0)
    okc = (F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99).float()
    if comps[0].is_cuda and all(c_.dtype == torch.float32 for c_ in comps) and KN.available():
        out = KN.run("tensor_dips", lambda: KN.tensor_dips(comps, okc, f),
                     lambda: tuple(_dips_from_tensor_torch(comps, okc, f).values()), 1e-6)
        return dict(zip(("DY", "DX", "DW"), out))
    return _dips_from_tensor_torch(comps, okc, f)


def _dips_from_tensor_torch(comps, okc, f):
    Jkk, Jyy, Jxx, Jky, Jkx, Jyx = comps
    # the same 12 power iterations from the slab normal, on the six components directly and in float32, every
    # tensor scaled by its trace first (the direction does not depend on the scale; the batched 3 x 3 float64
    # products were the tensor's own time)
    a, b_, c_, d, e, g_ = (torch.nan_to_num(t) for t in (Jkk, Jyy, Jxx, Jky, Jkx, Jyx))
    tr = (a + b_ + c_)
    sc_ = 1.0 / tr.clamp(min=1e-30)
    a, b_, c_, d, e, g_ = a * sc_, b_ * sc_, c_ * sc_, d * sc_, e * sc_, g_ * sc_
    nk_, ny_, nx_ = torch.ones_like(a), torch.zeros_like(a), torch.zeros_like(a)
    for _ in range(12):
        u = a * nk_ + d * ny_ + e * nx_; v_ = d * nk_ + b_ * ny_ + g_ * nx_; w_ = e * nk_ + g_ * ny_ + c_ * nx_
        inv = 1.0 / torch.sqrt(u * u + v_ * v_ + w_ * w_).clamp(min=1e-30)
        nk_, ny_, nx_ = u * inv, v_ * inv, w_ * inv
    l1 = nk_ * (a * nk_ + d * ny_ + e * nx_) + ny_ * (d * nk_ + b_ * ny_ + g_ * nx_) + nx_ * (e * nk_ + g_ * ny_ + c_ * nx_)
    pl = (2 * l1 - (a + b_ + c_)) / l1.clamp(min=1e-12)             # (the scaled trace: 1, or 0 for an empty tensor)
    sgn = torch.sign(nk_ + 1e-12)
    nk_, ny_, nx_ = nk_ * sgn, ny_ * sgn, nx_ * sgn
    nkc = nk_.clamp(min=0.3)
    dy = (-ny_ / nkc).clamp(-1.5, 1.5); dx = (-nx_ / nkc).clamp(-1.5, 1.5)
    return {"DY": (dy * f).float(), "DX": (dx * f).float(), "DW": (pl.float().clamp(0, 1) * okc[None])}


def lamination_dips_scan_light(Vt, okf, step, cfg):
    """the semblance slope scan for the SHEAR FIELD only (lamination_dips_shear averages the scan's slopes over the
    slab depth and smooths them over 2 cells before integrating them): a semblance stack over slopes and lags,
    evaluated at ONE px position per cell along the lag axis (neighbouring positions share 90 % of their traces), at
    every px across it (those are independent traces: pooled), and at every `shear_kstride`-th depth (the scan's output
    is smoothed 2 vox along depth and then averaged over it). ~10x fewer voxels; outputs on the SUB-SAMPLED depths
    [K / kstride, Hc, Wc]: DY, DX (labels of the ORIGINAL step per cell), DW."""
    vpx = L.CELL / UP
    m = okf.bool()
    X = torch.where(m[None], Vt, Vt[:, m].mean() if m.any() else torch.zeros((), device=DEV))
    X = X - L.filt(X, (cfg.get("scan_hp", 3.0) / step, 0, 0))
    K, Hp, Wp = X.shape
    Hc, Wc = Hp // UP, Wp // UP
    c = UP // 2
    kst = max(1, int(cfg.get("shear_kstride", 4)))
    D, dd = int(round(cfg.get("scan_D", 10) * PXS)), max(1, int(round(cfg.get("scan_dd", 2) * PXS)))
    pmax, npv = cfg.get("scan_pmax", 1.0) / PXS, int(cfg.get("scan_np", 51))
    ps = np.linspace(-pmax, pmax, npv)
    lags = list(range(-D, D + 1, dd))
    gw = {d: math.exp(-0.5 * (d / (0.6 * D)) ** 2) for d in lags}
    padk = int(math.ceil(pmax * D / step)) + 2
    Xp = F.pad(X[None, None], (D, D, D, D, padk, padk), mode="replicate")[0, 0]
    Op = F.pad(okf[None, None], (D, D, D, D), value=0.0)[0, 0]
    del X
    sk, sp = cfg.get("dip_si", 2.0) / step / kst, cfg.get("scan_sp", 4.0) * PXS
    Ks = len(range(0, K, kst))
    out = {}
    for ax in ("y", "x"):
        # output positions: along the lag axis the cell centres (stride UP), across it every px
        if ax == "y":
            ys0, ysn, yst, xs0, xsn, xst = c, Hc, UP, 0, Wc * UP, 1
        else:
            ys0, ysn, yst, xs0, xsn, xst = 0, Hc * UP, 1, c, Wc, UP
        sl = lambda oy, ox: (slice(D + oy + ys0, D + oy + ys0 + (ysn - 1) * yst + 1, yst), slice(D + ox + xs0, D + ox + xs0 + (xsn - 1) * xst + 1, xst))
        wsl = {}
        for d in lags:
            oy, ox = (d, 0) if ax == "y" else (0, d)
            yy, xx = sl(oy, ox)
            wsl[d] = gw[d] * Op[yy, xx]
        wsum = sum(wsl.values())
        pool = (lambda T_: F.avg_pool2d(T_[:, None], (1, UP))[:, 0]) if ax == "y" else (lambda T_: F.avg_pool2d(T_[:, None], (UP, 1))[:, 0])
        # the same stack with less memory traffic (the loop is bound by it): per lag the traces at the output positions are laid out ONCE, contiguous and weighted by
        # sqrt(w) (w = the lag's weight x the trace's validity, constant along depth), so that per (slope, lag)
        # ys = sqrt(w) x(shifted) is one interpolation between two depth-strided views, and the stack's sum and
        # energy are two in-place updates: st += sqrt(w) ys (= w x), en += ys^2 (= w x^2)
        def ref(ax=ax, wsl=wsl, wsum=wsum, sl=sl, pool=pool):
            Sm = []; Dm = []
            sq = {d: torch.sqrt(wsl[d]) for d in lags}
            Yd = {}
            for d in lags:
                oy, ox = (d, 0) if ax == "y" else (0, d)
                yy, xx = sl(oy, ox)
                Yd[d] = Xp[:, yy, xx] * sq[d][None]
            shp_ = (Ks,) + tuple(wsum.shape)
            st_ = torch.empty(shp_, device=DEV); en = torch.empty(shp_, device=DEV); buf = torch.empty(shp_, device=DEV)
            for p in ps:
                st_.zero_(); en.zero_()
                for d in lags:
                    f = p * d / step
                    i0 = int(math.floor(f)); fr = float(f - i0)
                    A_ = Yd[d][padk + i0:padk + i0 + K:kst]
                    ys = A_ if fr == 0.0 else torch.lerp(A_, Yd[d][padk + i0 + 1:padk + i0 + 1 + K:kst], fr, out=buf)
                    st_.addcmul_(ys, sq[d][None]); en.addcmul_(ys, ys)
                Sm.append(pool(st_ * st_)); Dm.append(pool(en * wsum[None]))
            del Yd, st_, en, buf
            return torch.stack(Sm, 0), torch.stack(Dm, 0)
        if Vt.is_cuda and KN.available():
            Smt, Dmt = KN.run("scan", lambda ax=ax: KN.scan_stack(Xp.contiguous(), Op.contiguous(), ps, lags, [gw[d] for d in lags], D,
                                                                   padk, kst, K, Hc, Wc, UP, ax, step), ref, 1e-4)
        else:
            Smt, Dmt = ref()
        # (the slopes' maps smoothed in ONE batch each: the same per-slope filter, 2 calls instead of 2 per slope)
        S = L.filt(Smt, (sk, sp / UP, sp / UP)) / L.filt(Dmt, (sk, sp / UP, sp / UP)).clamp(min=1e-9)
        ib = S.argmax(0); ibc = ib.clamp(1, npv - 2)
        l_ = S.gather(0, (ibc - 1)[None])[0]; c_ = S.gather(0, ibc[None])[0]; r_ = S.gather(0, (ibc + 1)[None])[0]
        den = l_ - 2 * c_ + r_
        off = torch.where(den < -1e-9, 0.5 * (l_ - r_) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        off = torch.where(ib == ibc, off, torch.zeros_like(off))
        pb = ps[0] + (ib.float() + off) * float(ps[1] - ps[0])
        out["D" + ax.upper()] = (pb / vpx) * L.CELL / step * cfg.get("dip_gain", 1.0)
        out["S" + ax] = S.max(0).values
        del S, Smt, Dmt
    okc = (F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99).float()
    return {"DY": out["DY"].float(), "DX": out["DX"].float(),
            "DW": (torch.minimum(out["Sy"], out["Sx"]).clamp(0, 1) * okc[None]).float()}


_DCT = {}


def _dct_mat(N, dev):
    """the orthonormal DCT-II matrix [N, N] (scipy.fft.dct type 2, norm "ortho"), float64, cached."""
    key = (int(N), str(dev))
    if key not in _DCT:
        k = torch.arange(N, dtype=torch.float64, device=dev)[:, None]; n = torch.arange(N, dtype=torch.float64, device=dev)[None]
        M = 2.0 * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        M[0] *= math.sqrt(1.0 / (4 * N)); M[1:] *= math.sqrt(1.0 / (2 * N))
        _DCT[key] = M
    return _DCT[key]


def _poisson_neumann(gy, gx, w):
    """the height h [H, W] (same units as gy * cell) whose gradient best fits (gy per +1 row, gx per +1 col), cells
    weighted by w only through the masking of the slopes (w <= 0: slope 0): div g = lap h with Neumann boundaries,
    solved by the DCT (Simchony, Chellappa & Shao 1990). Mean 0. Torch float64 (the DCTs as matrix products)."""
    H, W = gy.shape
    gy = torch.where(w > 0, gy, torch.zeros_like(gy)).double(); gx = torch.where(w > 0, gx, torch.zeros_like(gx)).double()
    # forward-difference gradients on the staggered grid: g_y[i] ~ h[i+1] - h[i]; divergence by backward differences
    div = torch.zeros((H, W), dtype=torch.float64, device=gy.device)
    div[:-1] += gy[:-1]; div[1:] -= gy[:-1]
    div[:, :-1] += gx[:, :-1]; div[:, 1:] -= gx[:, :-1]
    Mh, Mw = _dct_mat(H, gy.device), _dct_mat(W, gy.device)
    D = Mh @ div @ Mw.T
    i = torch.arange(H, dtype=torch.float64, device=gy.device)[:, None]; j = torch.arange(W, dtype=torch.float64, device=gy.device)[None]
    den = (2 * torch.cos(math.pi * i / H) - 2) + (2 * torch.cos(math.pi * j / W) - 2)
    den[0, 0] = 1.0
    Hh = D / den; Hh[0, 0] = 0.0
    return Mh.T @ Hh @ Mw


def _gauss_nd(x, sg):
    """scipy.ndimage.gaussian_filter(x, sg) of each float32 [H, W] image of x [..., H, W] (torch): the same kernel
    (truncate 4), the same 'reflect' (half-sample symmetric) border, each axis summed in float64 and rounded to
    float32 as SciPy does."""
    r = int(4.0 * sg + 0.5)
    xs = torch.arange(-r, r + 1, dtype=torch.float64, device=x.device)
    k = torch.exp(-0.5 / (sg * sg) * xs ** 2); k = k / k.sum()
    y = x.float()
    for ax in (y.dim() - 2, y.dim() - 1):
        n = y.shape[ax]
        idx = torch.arange(-r, n + r, device=x.device) % (2 * n)
        idx = torch.where(idx >= n, 2 * n - 1 - idx, idx)
        z = y.index_select(ax, idx).double()
        z = z.unfold(ax, 2 * r + 1, 1)                                  # [.., n, .., 2r + 1]
        y = (z @ k).float()
    return y


def _dip_field(D, conf, okc, step, sg):
    """the HEIGHT FIELD of the laminations over a tile (vox, cell grid, median 0; torch float64): the confidence-weighted
    k-mean of the slopes D (labels per cell) -> smoothed sg cells -> integrated (DCT Poisson). okc [Hc, Wc] bool."""
    okc = torch.as_tensor(okc, device=conf.device)
    den = conf.sum(0).clamp(min=1e-6)
    dy = ((D["DY"] * conf).sum(0) / den) * step
    dx = ((D["DX"] * conf).sum(0) / den) * step
    wm = conf.mean(0) * okc
    nrm, dy, dx = _gauss_nd(torch.stack([wm, dy * wm, dx * wm]), sg)    # (the three smoothings in one batch)
    dy = dy / nrm.clamp(min=1e-6)
    dx = dx / nrm.clamp(min=1e-6)
    h = _poisson_neumann(dy, dx, nrm > 0.02)
    return h - (torch.quantile(h[okc], 0.5) if bool(okc.any()) else 0.0)


def lamination_dips_shear(Vt, okf, step, cfg):
    """SHEARED TENSOR: unbiased at any tilt (as the scan) with the tensor's low noise near zero tilt. The scan's slopes
    (confidence-weighted over the slab depths), smoothed shear_sig cells, are integrated into a height field s(y, x) (DCT Poisson); the slab is
    re-sampled SHEARED, V'(k', y, x) = V(k' + s, y, x), so the layers are nearly flat in it; the structure tensor reads
    the RESIDUAL slope there. ITERATED (shear_iters): the residual is integrated into s again, until the sheared layers
    read flat; the fixed point does not depend on the tensor's gain (on the real CT it reads small tilts at 0.35-0.8:
    a single pass keeps that share of the scan's noise). Total slope = grad s + the last residual, mapped back to the
    original depth index k = k' + s. Same outputs as lamination_dips (DY, DX labels per cell, DW)."""
    K, Hp, Wp = Vt.shape
    okc_c = F.avg_pool2d(okf[None, None], UP)[0, 0] > 0.99
    sg = float(cfg.get("shear_sig", 2.0))

    def field(D, conf):
        return _dip_field(D, conf, okc_c, step, sg)

    m = okf.bool()
    X = torch.where(m[None], Vt, Vt[:, m].mean() if m.any() else torch.zeros((), device=DEV))
    kk = torch.arange(K, device=DEV, dtype=torch.float32)[:, None, None]
    sc = lamination_dips_scan_light(Vt, okf, step, dict(cfg, dip_gain=1.0))
    h = field(sc, sc["DW"])
    del sc
    tcfg = dict(cfg, dip_gain=float(cfg.get("shear_tgain", 1.28)))
    nit = max(1, int(cfg.get("shear_iters", 1)))   # 1: gabl7 (real CT) one pass beats 3 (0.7 vs 1.3 % > 5 vox)
    for it in range(nit):
        hp = F.interpolate(h.float()[None, None], size=(Hp, Wp), mode="bilinear",
                           align_corners=False)[0, 0] / step              # slab index units, px grid
        Xs = _shift_depth(X, hp)                                          # V'(k') = V(k' + s)
        td = lamination_dips(Xs, okf, step, tcfg)
        del Xs
        if it < nit - 1:                                                  # the residual into the shear field
            h = h + field(td, td["DW"])
            del td
    del X
    sc_c = (h / step).float()                                            # labels, cell grid
    srcc = kk - sc_c[None]                                               # at (k, cell) the sheared index is k - s
    okk = (srcc >= 0) & (srcc <= K - 1)
    srcc = srcc.clamp(0, K - 1); j0 = srcc.floor().long().clamp(0, K - 2); f2 = srcc - j0.float()
    back = lambda Y: torch.lerp(Y.gather(0, j0), Y.gather(0, j0 + 1), f2)
    gyl = torch.zeros_like(sc_c); gxl = torch.zeros_like(sc_c)          # grad s, labels per cell
    gyl[1:-1] = 0.5 * (sc_c[2:] - sc_c[:-2]); gxl[:, 1:-1] = 0.5 * (sc_c[:, 2:] - sc_c[:, :-2])
    gyl[0] = sc_c[1] - sc_c[0]; gyl[-1] = sc_c[-1] - sc_c[-2]; gxl[:, 0] = sc_c[:, 1] - sc_c[:, 0]; gxl[:, -1] = sc_c[:, -1] - sc_c[:, -2]
    g = float(cfg.get("dip_gain", 1.0))
    out = {"DY": ((back(td["DY"]) + gyl[None]) * g).float(), "DX": ((back(td["DX"]) + gxl[None]) * g).float(),
           "DW": (back(td["DW"]) * okk.float()).float()}
    if cfg.get("shear_keep"):                                           # the laminations' height field itself (vox, cell
        hf = h + field(td, td["DW"])                                     # grid, tile-local constant): the shear + the
        out["SHEAR"] = hf.float()                                          # residual; the fibre descriptors are
    del td                                                               # extracted along it (flat layers at any tilt)
    return out


def _shift_depth(X, s, valid=False):
    """X [K, H, W] at depth k + s [H, W] (slab index units; linear, clamped), with valid: also where 0 <= k + s <= K - 1."""
    if X.is_cuda and KN.available():
        return KN.run("depth_shift", lambda: KN.depth_shift(X, s, valid), lambda: _shift_depth_torch(X, s, valid), 1e-6)
    return _shift_depth_torch(X, s, valid)


def _shift_depth_torch(X, s, valid=False):
    K, Hp, Wp = X.shape
    out = torch.empty_like(X)
    vk = torch.empty((K, Hp, Wp), dtype=torch.bool, device=X.device) if valid else None
    kk = torch.arange(K, device=X.device, dtype=torch.float32)[:, None, None]
    for a in range(0, Hp, 256):                                        # (in row bands: the index tensors' memory)
        src = kk + s[None, a:a + 256]
        if valid:
            vk[:, a:a + 256] = (src >= 0) & (src <= K - 1)
        src = src.clamp(0, K - 1); i0 = src.floor().long().clamp(0, K - 2); fr = src - i0.float()
        out[:, a:a + 256] = torch.lerp(X[:, a:a + 256].gather(0, i0), X[:, a:a + 256].gather(0, i0 + 1), fr)
    return (out, vk) if valid else out


DIP_FN = {}


def bar_descriptors(Vt, okf, step, win, cfg, shear=None):
    """the fibre constellation's cross-sections, one line per 5-cell bin and family, computed ON THE LINES ONLY. The
    full-slab form filters the whole px-resolution slab five times per family (12 px along the fibres, 12 x 12 and 40 px high-passes,
    two depth filters; up to 131 x 720 x 720 voxels) and then reads ONE line per 5-cell bin. All of these filters are
    linear and separable, so: the two along-fibre kernels (the 12 px smoothing, and the same composed with the 12 px of
    the 2-D high-pass) are evaluated only at the bins' centre columns (H family) / rows (V family) as strided
    convolutions, and the across-fibre and depth filters run on those few lines (the tile's margin is 160 px)."""
    R0, C0, B, hb, wb = cfg["_bar"]
    bw = _bar_bw(cfg)
    m = okf.bool()
    Z = torch.where(m[None], Vt, Vt[:, m].mean() if m.any() else torch.zeros((), device=DEV))
    K, Hp, Wp = Z.shape
    vk = None; sp = None
    if shear is not None:
        sp = F.interpolate(shear[None, None], size=(Hp, Wp), mode="bilinear", align_corners=False)[0, 0] / step   # slab index
        Z, vk = _shift_depth(Z, sp, valid=True)                       # V'(k') = V(k' + s) and where k' + s is inside
    Z = (Z - Z[:, m].mean()) / Z[:, m].std().clamp(min=1e-6) if m.any() else Z
    s_al, hp, dsm = cfg.get("bar_along", 12.0) * PXS, cfg.get("bar_hp", 40.0) * PXS, cfg.get("bar_dsm", 1.5)
    w0, w1, w2, w3 = win
    bi = torch.arange(hb, device=DEV); bj = torch.arange(wb, device=DEV)
    py = (R0 + bi * B + B // 2 - w0) * UP + UP // 2                    # the bin centres in the tile's px
    px = (C0 + bj * B + B // 2 - w2) * UP + UP // 2
    oky = (py >= 0) & (py < Hp); okx = (px >= 0) & (px < Wp)
    out = {}
    if not (bool(oky.any()) and bool(okx.any())):
        e = torch.zeros(0, dtype=torch.long, device=DEV)
        return {fam: (e, e, torch.zeros((K, 0, 2 * bw), device=DEV), torch.zeros(0, dtype=torch.bool, device=DEV)) for fam in ("AH", "AV")}
    g_al = L.gkern(max(s_al, 0.5)); g12 = L.gkern(12.0 * PXS)
    k2 = L.kern_compose(g_al, g12)                                    # the along smoothing after the 2-D high-pass's own
    k1 = F.pad(g_al, ((len(k2) - len(g_al)) // 2,) * 2)               # the along smoothing alone, on the same support
    kern = torch.stack([k1, k2], 0)
    imap = torch.cumsum(oky.long(), 0) - 1; jmap = torch.cumsum(okx.long(), 0) - 1
    off = torch.arange(-bw, bw, device=DEV)
    ii, jj = torch.meshgrid(torch.nonzero(oky)[:, 0], torch.nonzero(okx)[:, 0], indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    for fam, al in (("AH", 2), ("AV", 1)):
        if al == 2:                                                   # H: smoothed along x, read at the bins' columns
            T2 = L.strided_corr(Z, kern, px[okx].cpu(), 2)            # [2, K, Hp, nx]
            S = T2[0] - L.filt(T2[1], (0, 12.0 * PXS, 0))             # (Z - G12y G12x Z) smoothed along x, at those columns
            del T2
            S = S - L.filt(S, (0, hp, 0))                             # high-pass ACROSS the fibres (rows)
        else:                                                         # V: smoothed along y, read at the bins' rows
            T2 = L.strided_corr(Z, kern, py[oky].cpu(), 1)            # [2, K, ny, Wp]
            S = T2[0] - L.filt(T2[1], (0, 0, 12.0 * PXS))
            del T2
            S = S - L.filt(S, (0, 0, hp))
        S = S - L.filt(S, (4.0 / step, 0, 0))
        S = L.filt(S, (dsm / step, 0, 0))
        shc = torch.round(sp[py[ii].clamp(0, Hp - 1), px[jj].clamp(0, Wp - 1)]).long() if sp is not None else None   # the
        minfrac = float(cfg.get("bar_minfrac", 0.4))                                                                  # FRAME depth

        def lines(S=S, al=al, shc=shc):
            if al == 2:                                               # the line runs along y (rows) at the centre column
                rows = py[ii][:, None] + off[None]; cols = px[jj][:, None].expand(-1, 2 * bw)
            else:
                rows = py[ii][:, None].expand(-1, 2 * bw); cols = px[jj][:, None] + off[None]
            ins = (rows >= 0) & (rows < Hp) & (cols >= 0) & (cols < Wp)   # PARTIAL lines: the valid part only
            rc, cc_ = rows.clamp(0, Hp - 1), cols.clamp(0, Wp - 1)
            mk = (okf[rc, cc_] > 0.5) & ins
            mk3 = mk[None] if vk is None else (mk[None] & vk[:, rc, cc_])
            if al == 2:
                A = S[:, rc, jmap[jj][:, None].expand(-1, 2 * bw)] * mk3
            else:
                A = S[:, imap[ii][:, None].expand(-1, 2 * bw), cc_] * mk3
            cnt = mk3.sum(2, keepdim=True).clamp(min=1)
            A = (A - A.sum(2, keepdim=True) / cnt) * mk3
            okb = mk.float().mean(1) >= minfrac
            A = torch.where(okb[None, :, None], A, torch.zeros_like(A))
            if shc is not None:                                       # index by the FRAME depth at the bin centre
                idx = torch.arange(K, device=DEV)[:, None] - shc[None, :]
                okd = (idx >= 0) & (idx < K)
                A = A.gather(0, idx.clamp(0, K - 1)[:, :, None].expand(-1, -1, A.shape[2])) * okd[:, :, None]
            return A, okb
        if S.is_cuda and KN.available():                              # (the lines' two passes in kernels)
            A, okb = KN.run("bar_lines", lambda S=S, al=al, shc=shc: KN.bar_lines(
                S.contiguous(), okf, vk, py[ii], px[jj], jmap[jj] if al == 2 else imap[ii],
                shc if shc is not None else torch.zeros_like(ii), bw, al == 2, minfrac), lines, 1e-5)
        else:
            A, okb = lines()
        out[fam] = (ii, jj, A, okb)
        del S
    return out


def _bar_bw(cfg):
    """the fibre lines' half length (px)."""
    return int(round(cfg.get("bar_bw", 150) * PXS))


def bar_cost(Fd, ks, labs, step, oy, ox, D, cfg):
    """THE FIBRE TIE between a bin and its neighbour at offset (oy, ox) (bins), as a cost table
    [2D+1, NL, hb, wb]: 1 - NCC of the fibre constellation of this bin at label l and of the other bin at label l - d,
    in the family whose fibres run ALONG that offset (H for a column offset, V for a row offset), over the family's own
    ply (H at +bar_kH vox, V at bar_kV; +-bar_th vox), the best of the across-fibre lags 0, +-bar_lag (px: the grid's
    rows / columns do not follow the fibres exactly). Not periodic in depth: the neighbouring winding carries other
    fibres. Bins without a valid cross-section: cost 1 for every d (no preference)."""
    A = Fd["BH" if oy == 0 else "BV"]                                # [K, n, hb, wb]
    K, n, hb, wb = A.shape
    o = int(round((cfg.get("bar_kH", 1.5) if oy == 0 else cfg.get("bar_kV", -2.5)) / step))
    th = int(round(cfg.get("bar_th", 1.5) / step))
    lag = max(1, int(round(cfg.get("bar_lag", 4) * PXS)))
    lags = (-lag, 0, lag)
    An = _shift_sp(A.reshape(K * n, hb, wb), oy, ox, fill=0.0).reshape(K, n, hb, wb)
    a = A[:, lag:n - lag]
    box = lambda X: F.avg_pool1d(F.pad(X.reshape(K, -1).t()[:, None], (th, th)), 2 * th + 1, 1)[:, 0].t().reshape(X.shape) * (2 * th + 1)
    e1 = box((a * a).sum(1))                                         # [K, hb, wb]
    centre = np.clip(np.round((np.asarray(labs) - float(ks[0])) / step).astype(int) + o, 0, K - 1)
    ci = torch.as_tensor(centre, device=DEV)
    kk = torch.arange(K, device=DEV)
    best = None
    # the depth shifts of a lag in ONE product (instead of a roll, a product, a box and a division per shift and lag)
    dv = torch.arange(-D, D + 1, device=DEV)
    nd_ = max(1, int(6e7 // max(a.numel(), 1)))                   # shifts per product (memory)
    boxd = lambda X: (F.avg_pool1d(F.pad(X.permute(0, 2, 3, 1).reshape(-1, 1, K), (th, th)), 2 * th + 1, 1) * (2 * th + 1)) \
        .reshape(X.shape[0], hb, wb, K).permute(0, 3, 1, 2)       # [nd, K, hb, wb]: the box along depth
    NUM = None
    if A.is_cuda and KN.available():                                 # every lag's and shift's products in one kernel
        def ref():
            si_ = (kk[None, :] - dv[:, None]) % K
            return torch.stack([torch.einsum("knhw,dknhw->dkhw", a, An[:, lag + g:n - lag + g][si_]) for g in lags])
        NUM = KN.run("shift_dot", lambda: KN.shift_dot(A, An, lag, lags, D), ref, 1e-4)
    for gi_, g in enumerate(lags):
        b = An[:, lag + g:n - lag + g]
        e2 = box((b * b).sum(1))
        if NUM is not None:
            src = kk[None, :] - dv[:, None]                            # [nd, K]: the other bin's depth index k - d
            okd = (src >= 0) & (src < K)
            ncc = boxd(NUM[gi_]) / torch.sqrt(e1[None] * e2[src % K]).clamp(min=1e-6)
            res = torch.where(okd[:, :, None, None], ncc, torch.zeros_like(ncc))[:, ci]
            best = res if best is None else torch.maximum(best, res)
            continue
        res = []
        for c0 in range(0, 2 * D + 1, nd_):
            dd = dv[c0:c0 + nd_]
            src = kk[None, :] - dd[:, None]                        # [nd, K]: the other bin's depth index k - d
            okd = (src >= 0) & (src < K)
            si = src % K                                           # (= the roll's wrap; masked below)
            num = boxd(torch.einsum("knhw,dknhw->dkhw", a, b[si]))
            ncc = num / torch.sqrt(e1[None] * e2[si]).clamp(min=1e-6)
            ncc = torch.where(okd[:, :, None, None], ncc, torch.zeros_like(ncc))
            res.append(ncc[:, ci])
        res = torch.cat(res, 0)
        best = res if best is None else torch.maximum(best, res)
        del res
    return (1.0 - best).clamp(0, 2)


def fibre_focus(Fd, ks, labs, step, cfg):
    """THE FIBRE FOCUS, a data term per bin and label (the ply-profile filter and the layer slopes alone do not follow
    the thick horizontal fibres).
    cost(bin, l) = 1 - the correlation of the family's across-fibre pattern of this bin, read for a surface at label l,
    with the pattern of the next bin ALONG the fibres read at the same depth (within +-foc_tol vox), both directions:
    low where a surface at l lies in the layer whose fibres RUN ON (the flattened view's continuous fibres, measured in
    the CT), high where it cuts through them or sits between plies. H (along the columns) has weight 1, V (along the
    rows) foc_v. The reading offsets foc_kH / foc_kV are calibrated so that approved surfaces read their minimum at
    l = 0. Per bin the minimum over the labels is subtracted (a bin without fibres says nothing)."""
    Dfl = max(int(round(cfg.get("foc_tol", 1.0) / step)), 0)
    c2 = dict(cfg, bar_kH=cfg.get("foc_kH", cfg.get("bar_kH", 1.5)), bar_kV=cfg.get("foc_kV", cfg.get("bar_kV", -2.5)))
    fam = {}
    for nm, offs in (("H", ((0, 1), (0, -1))), ("V", ((1, 0), (-1, 0)))):
        fam[nm] = sum(bar_cost(Fd, ks, labs, step, oy, ox, Dfl, c2).amin(0) for oy, ox in offs) / 2.0
    fv = float(cfg.get("foc_v", 0.0))
    Uf = (fam["H"] + fv * fam["V"]) / (1.0 + fv)
    return Uf - Uf.amin(0, keepdim=True)


DIP_FN.update(tensor=lambda *a: lamination_dips(*a), shear=lambda *a: lamination_dips_shear(*a))


def dip_cost(Fd, ks, labs, step, oy, ox, D, cfg):
    """[2D+1, NL, H, W]: the cost of the label change d = l(this) - l(neighbour at offset (oy, ox)) given the measured
    laminations: the layer through the neighbour continues here at e = -(DY oy + DX ox) labels (the two cells' dips
    averaged, read at label l); cost = conf * min(((d - e) step / dip_tol)^2, dip_cap)."""
    e, w = _dip_ew(Fd, ks, labs, step, oy, ox)
    tol, cap = cfg.get("dip_tol", 0.5), cfg.get("dip_cap", 4.0)
    dv = torch.arange(-D, D + 1, device=DEV, dtype=e.dtype).view(-1, 1, 1, 1)
    return w[None] * torch.clamp(((dv - e[None]) * (step / tol)) ** 2, max=cap)


def _dip_ew(Fd, ks, labs, step, oy, ox):
    """dip_cost's inputs [NL, H, W]: e the label change the laminations predict toward the neighbour at (oy, ox), w the
    pair's weight."""
    idx = torch.as_tensor(np.clip(np.round((np.asarray(labs) - float(ks[0])) / step).astype(int), 0, Fd["DY"].shape[0] - 1), device=DEV)
    DYl, DXl, Wl = Fd["DY"][idx], Fd["DX"][idx], Fd["DW"][idx]
    e = -0.5 * ((DYl + _shift_sp(DYl, oy, ox)) * oy + (DXl + _shift_sp(DXl, oy, ox)) * ox)
    w = 0.5 * (Wl + _shift_sp(Wl, oy, ox, fill=0.0))
    return e, w


def region_feats(P, v, N, region, ks, cfg, tile=(110, 110), pad=8):
    """features over a region (r0, r1, c0, c1) of cells, tiled with overlap -> torch [K, H, W] each + okc. cfg feat_tile /
    bar_tile: other tile sizes (cells; one tile = the whole region for sizes above it)."""
    R0, R1, C0, C1 = region
    tile = tuple(int(x) for x in cfg.get("feat_tile", tile))
    H, W = R1 - R0, C1 - C0
    K = len(ks)
    names = ("EV", "EH", "I") + (("DY", "DX", "DW") if cfg.get("dip_w", 0) > 0 else ())
    if cfg.get("dips_only"):
        names = ("DY", "DX", "DW")
    out = {n: torch.zeros((K, H, W), device=DEV) for n in names}
    okc = torch.zeros((H, W), dtype=torch.bool, device=DEV)
    bar = cfg.get("foc_w", 0) > 0 and cfg.get("B", 1) > 1 and not cfg.get("dips_only")
    if bar:                                                            # the fibre constellations on the coarse bin grid
        Bb = int(cfg["B"]); hb, wb = -(-H // Bb), -(-W // Bb); nbw = 2 * _bar_bw(cfg)
        pad = max(pad, _bar_bw(cfg) // UP + 2)                         # the across-fibre line must fit in the padded tile
        bt_ = int(cfg.get("bar_tile", 80))                            # the frame: 192 (hsolve.GEN_SCFG), one canvas for the
        tile = (min(tile[0], bt_), min(tile[1], bt_))                 # regions of the benchmark's 160-cell areas. Every tile
        # normalises its slab and anchors its lamination height field on its own, so the tile size is part of the solver: on
        # the checked cases one canvas solves the frame's patches as well as 80-cell tiles (-0.03 to +0.13 points of right
        # sheet) in 0.56-0.79x the time
        cfg = dict(cfg, _bar=(R0, C0, Bb, hb, wb))
        BAR = {f: torch.zeros((K, nbw, hb, wb), device=DEV) for f in ("H", "V")}
    mg = 2 * max(int(cfg.get("B", 1)), 1)        # a tile whose core has no cell of v within 2 bins is not computed: nothing
    for a in range(R0, R1, tile[0]):             # read from it reaches a cell of v (the unary and the fibre focus are masked
        for b in range(C0, C1, tile[1]):         # to the valid cells, the dips' weight is 0 off them, the fibre ties read
            w = (max(a - pad, 0), min(a + tile[0] + pad, v.shape[0]), max(b - pad, 0), min(b + tile[1] + pad, v.shape[1]))
            if not v[max(a - mg, 0):min(a + tile[0], R1) + mg, max(b - mg, 0):min(b + tile[1], C1) + mg].any():
                continue                         # one bin around): the same result
            t = tile_feats(P, v, N, w, ks, cfg, core=(a - w[0], min(a + tile[0], R1) - w[0], b - w[2], min(b + tile[1], C1) - w[2]))
            if t is None:
                continue
            ia, ib = a - w[0], b - w[2]
            ha = min(tile[0], R1 - a, t["okc"].shape[0] - ia); hb = min(tile[1], C1 - b, t["okc"].shape[1] - ib)
            for n in out:
                out[n][:, a - R0:a - R0 + ha, b - C0:b - C0 + hb] = t[n][:, ia:ia + ha, ib:ib + hb]
            okc[a - R0:a - R0 + ha, b - C0:b - C0 + hb] = t["okc"][ia:ia + ha, ib:ib + hb]
            if bar and "bar" in t:                                    # the bins whose centre lies in this tile's core
                for nm, dst in [("A" + k_, d_) for k_, d_ in BAR.items()]:
                    ii, jj, A_, okb = t["bar"][nm]
                    cr = R0 + ii * Bb + Bb // 2; cc_ = C0 + jj * Bb + Bb // 2
                    core = (cr >= a) & (cr < a + tile[0]) & (cc_ >= b) & (cc_ < b + tile[1])
                    dst[:, :, ii[core], jj[core]] = A_[:, core].permute(0, 2, 1)
    vr = torch.as_tensor(v[R0:R1, C0:C1], device=DEV)
    out["okc"] = okc & vr
    if bar:
        for k_, d_ in BAR.items():
            out["B" + k_] = d_
    return out


# ------------------------------------------------------------------------------------------------ data term


def _depth_corr(X, w, J):
    """X [K, H, W], w [2J+1] -> [K - 2J, H, W]: sum_j w_j X(i + j) (cross-correlation along depth)."""
    K, H, Wd = X.shape
    if X.is_cuda and X.dtype == torch.float32 and KN.available():      # (the valid positions of the depth filter)
        return KN.run("depth_corr", lambda: KN.sepconv(X, w, 0, start=J, count=K - 2 * J), lambda: _depth_corr_torch(X, w), 1e-5)
    return _depth_corr_torch(X, w)


def _depth_corr_torch(X, w):
    K, H, Wd = X.shape
    Y = F.conv1d(X.permute(1, 2, 0).reshape(-1, 1, K), w.view(1, 1, -1))
    return Y.reshape(H, Wd, -1).permute(2, 0, 1)


def unary_filter(Fd, ks, labs, wvec, J, step):
    """score(l) = Pearson-like: (sum_cj w_cj X_c(l + j) - mean_l * sum w) / std_l over the 3(2J+1)-dim profile at
    label l (the profile z-normalised, as in calibration). Returns score [NL, H, W]."""
    chans = ("EV", "EH", "I")
    D = 2 * J + 1
    wv = torch.as_tensor(wvec, device=DEV, dtype=torch.float32).view(3, D)
    ones = torch.ones(D, device=DEV)
    num = 0; s1 = 0; s2 = 0
    for ci, n in enumerate(chans):
        X = Fd[n]
        num = num + _depth_corr(X, wv[ci], J)
        s1 = s1 + _depth_corr(X, ones, J)
        s2 = s2 + _depth_corr(X * X, ones, J)
    nD = 3 * D
    mean = s1 / nD
    std = torch.sqrt((s2 / nD - mean ** 2).clamp(min=1e-6))
    sc = (num - mean * wv.sum()) / std                       # [K - 2J, H, W], index i <-> centre ks[i + J]
    centre = (np.asarray(labs) - float(ks[0])) / step - J
    idx = np.round(centre).astype(int)
    good = (idx >= 0) & (idx < sc.shape[0])
    out = sc[torch.as_tensor(np.clip(idx, 0, sc.shape[0] - 1), device=DEV)]
    return torch.where(torch.as_tensor(good, device=DEV)[:, None, None], out, torch.full_like(out, -1e3))


def profiles_at(Fd, ks, lab_vals, J, step, okc):
    """z-normalised profiles [n, 3(2J+1)] at the given labels (list) over the valid cells."""
    out = []
    for lv in lab_vals:
        ci = int(round((lv - float(ks[0])) / step))
        parts = [Fd[n][ci - J:ci + J + 1][:, okc].T for n in ("EV", "EH", "I")]
        X = torch.cat(parts, 1)
        X = (X - X.mean(1, keepdim=True)) / X.std(1, keepdim=True).clamp(min=1e-6)
        out.append(X)
    return out


def calibrate(feats, ks, cfg, neg_min=1.0, neg_max=None):
    """Fisher LDA matched filter + template from green feature volumes (truth = label 0 everywhere)."""
    J, step = cfg["J"], cfg["step"]
    neg_max = neg_max or cfg["kmax"]
    negl = [x for x in np.arange(-neg_max, neg_max + 1e-6, step) if abs(x) >= neg_min]
    Xp, Xn = [], []
    for Fd in feats:
        ok = Fd["okc"]
        Xp.append(profiles_at(Fd, ks, [0.0], J, step, ok)[0])
        rng = np.random.default_rng(0)
        for lv, X in zip(negl, profiles_at(Fd, ks, negl, J, step, ok)):
            sel = torch.as_tensor(rng.random(X.shape[0]) < 0.08, device=DEV)
            Xn.append(X[sel])
    Xp = torch.cat(Xp).double(); Xn = torch.cat(Xn).double()
    Sw = torch.cov(Xp.T) + torch.cov(Xn.T)
    Sw = Sw + 1e-2 * torch.trace(Sw) / Sw.shape[0] * torch.eye(Sw.shape[0], device=DEV, dtype=torch.float64)
    w = torch.linalg.solve(Sw, Xp.mean(0) - Xn.mean(0))
    w = (w / w.norm()).float()
    tmpl = Xp.median(0).values.float()
    sp = (Xp.float() @ w); sn = (Xn.float() @ w)
    return {"w": w.cpu().numpy(), "tmpl": tmpl.cpu().numpy(), "J": J, "step": step, "neg_min": neg_min,
            "scale": float(torch.cat([sp, sn]).std()), "n_pos": int(Xp.shape[0]), "n_neg": int(Xn.shape[0]),
            "dprime": float((sp.mean() - sn.mean()) / torch.sqrt(0.5 * (sp.var() + sn.var())))}


_CAL = {}


def unary(Fd, ks, labs, cfg):
    """U [NL, H, W] >= 0 and contrast [H, W]: the calibrated matched filter (cfg calib: calibrate's npz)."""
    if cfg["calib"] not in _CAL:
        _CAL[cfg["calib"]] = dict(np.load(cfg["calib"]))
    cal = _CAL[cfg["calib"]]
    J = int(cal["J"])
    sc = unary_filter(Fd, ks, labs, cal["w"], J, cfg["step"])
    scale = float(cal["scale"])
    smax = sc.max(0).values
    contrast = (smax - sc.median(0).values) / scale
    U = (smax[None] - sc) / scale * 0.5 * cfg["uscale"]
    return U, contrast


# ------------------------------------------------------------------------------------------------ layer continuity
def _shift_sp(X, oy, ox, fill=None):
    """out[..., i, j] = X[..., i + oy, j + ox] (edge: replicate, or `fill`)."""
    if oy == 0 and ox == 0:
        return X
    H, W = X.shape[-2:]
    pad = (max(-ox, 0), max(ox, 0), max(-oy, 0), max(oy, 0))
    Y = F.pad(X.reshape(-1, 1, H, W), pad, mode="replicate").reshape(*X.shape[:-2], H + abs(oy), W + abs(ox))
    y0 = max(oy, 0); x0 = max(ox, 0)
    out = Y[..., y0:y0 + H, x0:x0 + W]
    if fill is not None:
        m = torch.ones((H, W), dtype=torch.bool, device=X.device)
        if oy > 0: m[H - oy:] = False
        if oy < 0: m[:-oy] = False
        if ox > 0: m[:, W - ox:] = False
        if ox < 0: m[:, :-ox] = False
        out = torch.where(m, out, torch.full_like(out, fill))
    return out


OFFS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPP = [1, 0, 3, 2]


def _shift_stack(pre, De):
    """S [2 De + 1, NL, ...]: S[d + De, l] = pre[l - d] (BIG outside the label range): every label shift of a belief /
    path-cost volume at once (a view of the padded volume + one flip), for the min over the shifts in ONE operation."""
    NL = pre.shape[0]
    pad = torch.cat([torch.full((De,) + pre.shape[1:], BIG, device=pre.device, dtype=pre.dtype), pre,
                     torch.full((De,) + pre.shape[1:], BIG, device=pre.device, dtype=pre.dtype)], 0)
    S = pad.unfold(0, NL, 1)                                        # [2 De + 1, ..., NL]: [s, .., l] = pre[s + l - De]
    return S.flip(0).movedim(-1, 1)                                 # s = De - d  ->  index d + De


def bp(U, D, P1, pcs, lam, iters, damp=0.5, ew=None):
    """damped parallel min-sum belief propagation on the 4-connected grid; hard bound |dl| <= D, pairwise
    P1 |dl| + lam * pcs[dir][dl + D] (dir = the neighbour's offset; pcs = dw x dip_cost per edge, a list or a function
    returning it). Returns the beliefs [NL, H, W]: on the GPU kernels.bp_iterations, which computes that cost per step
    from its inputs ew = (E, Wt, sct, cap, dw) (_dip_ew; None when lam = 0), else the PyTorch loop (_bp_torch)."""
    pl = (lambda: pcs() if callable(pcs) else pcs)
    if U.is_cuda and KN.available():
        assert ew is not None or lam == 0

        def fast():
            NL, H, W = U.shape
            valid = torch.stack([_shift_sp(torch.ones((1, H, W), device=U.device), oy, ox, fill=0.0)[0] for oy, ox in OFFS])
            if lam > 0:
                E, Wt, sct, cap, dw = ew
            else:                                                   # (no dips term: P1 |dl| + 0)
                E = Wt = torch.zeros((4, NL, H, W), device=U.device); sct, cap, dw = 1.0, 1.0, 0.0
            return KN.bp_iterations(U.contiguous(), E, Wt, valid, iters, damp, P1, lam, sct, cap, dw, D)
        return KN.run("bp", fast, lambda: _bp_torch(U, D, P1, pl(), lam, iters, damp), 1e-5)
    return _bp_torch(U, D, P1, pl(), lam, iters, damp)


def _bp_torch(U, D, P1, pcs, lam, iters, damp=0.5):
    NL, H, W = U.shape
    edges = []
    for di, (oy, ox) in enumerate(OFFS):
        T_ = P1 * torch.arange(-D, D + 1, device=U.device, dtype=U.dtype).abs().view(-1, 1, 1, 1)
        if lam > 0:
            T_ = T_ + lam * pcs[di]
        edges.append((oy, ox, D, T_, OPP[di]))
    valid = [_shift_sp(torch.ones((1, H, W), device=U.device), oy, ox, fill=0.0) for oy, ox, _, _, _ in edges]
    M = [torch.zeros_like(U) for _ in edges]
    for it in range(iters):
        B = U + sum(M)
        new = []
        for di, (oy, ox, De, T_, opp) in enumerate(edges):
            pre = _shift_sp(B - M[opp], oy, ox, fill=0.0)              # the neighbour's belief without what it got from here
            best = (_shift_stack(pre, De) + T_).min(0).values
            best = (best - best.min(0, keepdim=True).values) * valid[di]
            new.append(damp * M[di] + (1 - damp) * best)
        M = new
    return U + sum(M)


# ------------------------------------------------------------------------------------------------ SGM(-P)
def _shift_lab(p, e):
    """p [NL, H] -> q[l, i] = p[l - e_i] (e float per row; linear between labels; outside -> BIG)."""
    NL, H = p.shape
    ei = torch.floor(e); f = (e - ei)[None]
    ei = ei.long()
    l = torch.arange(NL, device=p.device)[:, None]
    i0 = l - ei[None]; i1 = i0 - 1
    g0 = p.gather(0, i0.clamp(0, NL - 1)); g1 = p.gather(0, i1.clamp(0, NL - 1))
    g0 = torch.where((i0 >= 0) & (i0 < NL), g0, torch.full_like(g0, BIG))
    g1 = torch.where((i1 >= 0) & (i1 < NL), g1, torch.full_like(g1, BIG))
    return (1 - f) * g0 + f * g1


def _scan(C, D, P1, xdir, sy, ex, ey, pc=None, lam=0.0):
    """one SGM path family along the last axis; predecessor of (i, x) = (i - sy, x - xdir). ex, ey [H, W] expected
    label change per +1 col / +1 row (or None). The min over the label shifts is one tensor operation per column
    (`_shift_stack`; identical to the per-shift loop `_scan_ref`)."""
    NL, H, W = C.shape
    Lr = torch.empty_like(C)
    order = range(W) if xdir == 1 else range(W - 1, -1, -1)
    prev = None
    zero_row = torch.zeros((NL, 1), device=C.device)
    T_ = P1 * torch.arange(-D, D + 1, device=C.device, dtype=C.dtype).abs().view(-1, 1, 1)
    PC = (lam * pc + T_[..., None]) if pc is not None else None      # [2D+1, NL, H, W]
    for x in order:
        c = C[:, :, x]
        if prev is None:
            cur = c
        else:
            if sy == 1:
                p = torch.cat([zero_row, prev[:, :-1]], 1)
            elif sy == -1:
                p = torch.cat([prev[:, 1:], zero_row], 1)
            else:
                p = prev
            if PC is not None:
                best = (_shift_stack(p, D) + PC[:, :, :, x]).min(0).values
            else:
                if ex is not None:
                    e = ex[:, x] * xdir + ey[:, x] * sy
                    p = _shift_lab(p, e)
                best = (_shift_stack(p, D) + T_).min(0).values
            mn = best.min(0).values
            cur = c + best - mn[None]
        Lr[:, :, x] = cur
        prev = cur
    return Lr


def sgm(C, D, P1, ex=None, ey=None, pcf=None, lam=0.0, dip=None):
    """8-path semi-global aggregation of C [NL, H, W] with the hard bound |dl - e| <= D labels (linear penalty P1 per
    label) -> S [NL, H, W] (sum of the 8 path costs). pcf(oy, ox) -> the layer-continuity cost [2D+1, NL, H, W] of
    the predecessor at offset (oy, ox) (then the expected slopes ex / ey are not used: the pairs measure them).
    dip: the same cost's inputs (DY, DX, DW, lidx, sct, cap, dw) for the GPU kernel, which computes it per step
    (kernels.sgm_dip: one launch per column / row for all paths), else the PyTorch loops (_sgm_torch)."""
    if dip is not None and pcf is not None and C.is_cuda and KN.available():
        def fast():
            Lp = KN.sgm_dip(C.contiguous(), dip["DY"].contiguous(), dip["DX"].contiguous(), dip["DW"].contiguous(), dip["lidx"],
                            D, P1, dip["sct"], dip["cap"], dip["dw"], lam)
            S = torch.zeros_like(C)
            for q in range(8):
                S += Lp[q]
            return S
        return KN.run("sgm", fast, lambda: _sgm_torch(C, D, P1, ex, ey, pcf, lam), 1e-5)
    if dip is None and pcf is None and ex is None and ey is None and C.is_cuda and KN.available():
        def plain():                                                  # the plain linear penalty (the fibre step's
            NL, H, W = C.shape                                        # follow): the same kernel with no layer cost
            z = torch.zeros((1, H, W), device=C.device, dtype=torch.float32)
            Lp = KN.sgm_dip(C.contiguous(), z, z, z, torch.zeros(NL, dtype=torch.long, device=C.device), D, P1, 1.0, 1.0, 0.0, 0.0)
            S = torch.zeros_like(C)
            for q in range(8):
                S += Lp[q]
            return S
        return KN.run("sgm_plain", plain, lambda: _sgm_torch(C, D, P1, ex, ey, pcf, lam), 1e-5)
    return _sgm_torch(C, D, P1, ex, ey, pcf, lam)


def _sgm_torch(C, D, P1, ex=None, ey=None, pcf=None, lam=0.0):
    S = torch.zeros_like(C)
    for xdir in (1, -1):
        for sy in (0, 1, -1):
            pc = pcf(-sy, -xdir) if pcf is not None else None
            S += _scan(C, D, P1, xdir, sy, None if pc is not None else ex, None if pc is not None else ey, pc, lam)
            del pc
    Ct = C.transpose(1, 2).contiguous()
    exT = ey.t().contiguous() if ey is not None else None
    eyT = ex.t().contiguous() if ex is not None else None
    for ydir in (1, -1):
        pc = pcf(-ydir, 0).transpose(2, 3).contiguous() if pcf is not None else None
        S += _scan(Ct, D, P1, ydir, 0, None if pc is not None else exT, None if pc is not None else eyT, pc, lam).transpose(1, 2)
        del pc
    return S


def _argmin_sub(S, labs):
    NL = S.shape[0]
    kb = S.argmin(0)
    km = kb.clamp(1, NL - 2)
    l_ = S.gather(0, (km - 1)[None])[0]; c_ = S.gather(0, km[None])[0]; r_ = S.gather(0, (km + 1)[None])[0]
    den = l_ - 2 * c_ + r_
    off = torch.where(den > 1e-9, 0.5 * (l_ - r_) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    off = torch.where(kb == km, off, torch.zeros_like(off))
    step = float(labs[1] - labs[0])
    return float(labs[0]) + (kb.float() + off) * step


def _pin(C, pins_t, lab_idx_t):
    C = C.clone()
    C[:, pins_t] = 20.0
    ii, jj = pins_t.nonzero(as_tuple=True)
    C[lab_idx_t[pins_t], ii, jj] = 0.0
    return C


# ------------------------------------------------------------------------------------------------ continuous
_STENCIL, _SPD = {}, {"ok": None, "cache": {}}


def _stencil(H, W, al, beta):
    """the continuous refinement's fixed part on an H x W grid: al (Dx'Dx + Dy'Dy) + beta (Dxx'Dxx + Dyy'Dyy) as CSR with
    every diagonal entry stored, the positions of the diagonal in its data, Dx and Dy (cached per grid and weights)."""
    from scipy import sparse
    key = (H, W, float(al), float(beta))
    if key not in _STENCIL:
        n = H * W
        idx = np.arange(n).reshape(H, W)
        def add_diff(a, b, wt):
            m = len(a)
            r = np.arange(m)
            return sparse.csr_matrix((np.r_[np.full(m, wt), np.full(m, -wt)], (np.r_[r, r], np.r_[a, b])), shape=(m, n))
        Dx = add_diff(idx[:, 1:].ravel(), idx[:, :-1].ravel(), 1.0)
        Dy = add_diff(idx[1:, :].ravel(), idx[:-1, :].ravel(), 1.0)
        def d2(a, b, c_):                                         # second differences
            m = len(a); r = np.arange(m)
            return sparse.csr_matrix((np.r_[np.ones(m), -2 * np.ones(m), np.ones(m)], (np.r_[r, r, r], np.r_[a, b, c_])), shape=(m, n))
        Dxx = d2(idx[:, 2:].ravel(), idx[:, 1:-1].ravel(), idx[:, :-2].ravel())
        Dyy = d2(idx[2:, :].ravel(), idx[1:-1, :].ravel(), idx[:-2, :].ravel())
        Cm = (al * (Dx.T @ Dx + Dy.T @ Dy) + beta * (Dxx.T @ Dxx + Dyy.T @ Dyy)).tocsr()
        Cm.setdiag(Cm.diagonal()); Cm.sort_indices()
        rows = np.repeat(np.arange(n), np.diff(Cm.indptr))
        dpos = np.nonzero(Cm.indices == rows)[0]
        if len(_STENCIL) >= 6:
            _STENCIL.pop(next(iter(_STENCIL)))
        _STENCIL[key] = (Cm, dpos, Dx.tocsr(), Dy.tocsr())
    return _STENCIL[key]


def _solve_spd(key, Cm, dpos, dg, rhs):
    """(Cm + diag(dg)) h = rhs, symmetric positive definite, float64. On the GPU NVIDIA cuDSS (nvmath-python; one
    symbolic analysis per grid shape, then per solve the diagonal written in place, a numeric factorisation and the
    solve); else SciPy SuperLU with a symmetric ordering. The same solution to ~1e-12."""
    from scipy.sparse.linalg import splu

    def cpu():
        A = Cm.copy(); A.data[dpos] += dg
        return splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(rhs)
    if _SPD["ok"] is None:
        try:
            import importlib
            importlib.import_module("cupy"); importlib.import_module("nvmath.sparse.advanced")
            _SPD["ok"] = DEV.startswith("cuda") and KN.available()
        except Exception as e:
            KN.log(f"cudss: nvmath-python not usable ({type(e).__name__}); the continuous solve runs on the CPU")
            _SPD["ok"] = False
    if not _SPD["ok"]:
        return cpu()
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, DirectSolverOptions
        cache = _SPD["cache"]
        st = cache.get(key)
        if st is None:
            if len(cache) >= 3:                                   # (a few grid shapes: the patches' and the passes')
                old = cache.pop(next(iter(cache)))
                old["solver"].free()
            A = csp.csr_matrix((cp.asarray(Cm.data), cp.asarray(Cm.indices.astype(np.int32)), cp.asarray(Cm.indptr.astype(np.int32))),
                               shape=Cm.shape)
            b = cp.zeros(Cm.shape[0], dtype=cp.float64)
            st = dict(A=A, b=b, dpos=cp.asarray(dpos), cdiag=cp.asarray(Cm.data[dpos]))
            st["solver"] = DirectSolver(A, b, options=DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD))
            st["solver"].plan()
            cache[key] = st
        st["A"].data[st["dpos"]] = st["cdiag"] + cp.asarray(dg)
        st["b"][...] = cp.asarray(rhs)
        st["solver"].factorize()
        h = cp.asnumpy(st["solver"].solve()).reshape(-1)
    except Exception as e:
        KN.log(f"cudss: failed ({type(e).__name__}: {str(e)[:200]}); the continuous solve runs on the CPU")
        _SPD["ok"] = False
        return cpu()
    chk = os.environ.get("RECTO_KERNELS_CHECK", "0")
    if "cudss" not in _SPD and (chk == "1" or "cudss" in chk.split(",")):
        r = cpu()
        diff = float(np.abs(h - r).max() / max(np.abs(r).max(), 1e-12))
        _SPD["cudss"] = diff < 1e-6
        KN.log(f"cudss: against SuperLU max rel diff {diff:.2g} -> {'GPU solve used' if _SPD['cudss'] else 'GPU solve OFF'}")
        if not _SPD["cudss"]:
            _SPD["ok"] = False
            return r
    return h


def _data_fit(U, labs, k0, okc, rng, step):
    """the continuous refinement's data term per cell, on U's device: d (vox, float64) the sub-label minimum of U (a
    parabola through the three labels around the best one within +-rng of k0), w (float32) its curvature (0..4,
    0 off okc); numpy out, the arithmetic and dtypes of the NumPy form it replaces."""
    NL = U.shape[0]
    lab = torch.as_tensor(np.asarray(labs, np.float64), device=U.device)[:, None, None]
    near = (lab - torch.as_tensor(np.asarray(k0), device=U.device).double()[None]).abs() <= rng
    ib = torch.where(near, U, torch.full_like(U, float("inf"))).argmin(0)
    ibc = ib.clamp(1, NL - 2)
    l_, c_, r_ = (U.gather(0, (ibc + o)[None])[0] for o in (-1, 0, 1))
    den = l_ - 2 * c_ + r_
    off = torch.where(den > 1e-6, (0.5 * (l_ - r_) / den.clamp(min=1e-6)).clamp(-0.5, 0.5), torch.zeros_like(den))
    d = float(labs[0]) + (ibc.double() + off.double()) * step
    w = (den / step ** 2).clamp(0, 4.0) * torch.as_tensor(np.asarray(okc), device=U.device).float()
    return d.cpu().numpy(), w.cpu().numpy()


def refine_continuous(U, labs, k0, okc, pins, pin_val, cfg, rng=1.5, gexp=None):
    """per-cell sub-voxel data estimate d (parabola of U around k0, within +-rng) with weight w (curvature), then
    min sum w (h - d)^2 + alpha |grad h|^2 + beta |lap h|^2 + pins (sparse LS, _solve_spd). numpy in / out."""
    step = float(labs[1] - labs[0])
    NL, H, W = U.shape
    d, w = _data_fit(U, labs, k0, okc, rng, step)
    wd = cfg["wdata"] * w.ravel()
    pw = np.where(pins.ravel(), 1e4, 0.0)
    al = cfg["alpha"] if gexp is None else cfg["dip_cont"]
    Cm, dpos, Dx, Dy = _stencil(H, W, al, cfg["beta"])
    rhs = wd * d.ravel() + pw * pin_val.ravel()
    if gexp is not None:                                          # alpha |grad h - the laminations' slope|^2
        gy_, gx_ = gexp
        rhs = rhs + al * (Dy.T @ np.nan_to_num(gy_).ravel() + Dx.T @ np.nan_to_num(gx_).ravel())
    h = _solve_spd((H, W, float(al), float(cfg["beta"])), Cm, dpos, wd + pw + 1e-6, rhs).reshape(H, W)
    h = h.astype(np.float32).astype(np.float64)                   # (to 1e-6 vox: the GPU factorisation's last bits differ
                                                                  # from run to run; the rounding makes them the same)
    return h, d, w


# ------------------------------------------------------------------------------------------------ the patch solve
def solve_levels(Fd, ks, okc, pins, pin_val, cfg, timing):
    """coarse -> fine -> continuous on one feature volume. pins bool [H, W] numpy, pin_val vox [H, W] numpy.
    Returns k (vox, numpy), conf dict."""
    step = cfg["step"]
    labs = np.arange(-cfg["kmax"], cfg["kmax"] + 1e-6, step)
    NL = len(labs)
    t0 = time.time()
    U, contrast = unary(Fd, ks, labs, cfg)
    U = torch.where(okc[None], U, torch.full_like(U, 0.3 * cfg["uscale"]))
    fw = cfg.get("foc_w", 0.0)
    if fw > 0 and "BH" in Fd and cfg["B"] > 1:                       # THE FIBRE FOCUS (fibre_focus): measured per bin, added
        Uf = fibre_focus(Fd, ks, labs, step, cfg)                     # to every cell of the bin, so that the coarse, the fine
        Uf = F.interpolate(Uf[None], scale_factor=cfg["B"], mode="nearest")[0][:, :okc.shape[0], :okc.shape[1]]   # and the
        U = U + fw * cfg["uscale"] * torch.where(okc[None], Uf, torch.zeros_like(Uf))                              # continuous stage see it
    _sync(); timing["unary_s"] = round(time.time() - t0, 3)
    H, W = okc.shape
    pins_t = torch.as_tensor(pins, device=DEV)
    lab_idx = torch.as_tensor(np.clip(np.round((pin_val + cfg["kmax"]) / step), 0, NL - 1).astype(np.int64), device=DEV)
    # coarse
    t0 = time.time()
    B = cfg["B"]
    okf = okc.float()
    num = F.avg_pool2d((U * okf[None])[None], B, ceil_mode=True)[0]
    den = F.avg_pool2d(okf[None, None], B, ceil_mode=True)[0]
    Ub = torch.where(den > 0.3, num / den.clamp(min=1e-6), torch.full_like(num, 0.3 * cfg["uscale"]))
    hb, wb = Ub.shape[1:]
    pt = torch.zeros(hb * B, wb * B, device=DEV); pt[:H, :W] = pins_t.float()
    pv = torch.zeros(hb * B, wb * B, device=DEV); pv[:H, :W] = torch.as_tensor(np.where(pins, pin_val, 0.0), device=DEV).float()
    pc = F.avg_pool2d(pt[None, None], B)[0, 0]; ps = F.avg_pool2d(pv[None, None], B)[0, 0]
    pinb = pc > 0.3
    labb = torch.clamp(torch.round((ps / pc.clamp(min=1e-6) + cfg["kmax"]) / step), 0, NL - 1).long()
    Dcl = int(round(cfg["Dc"] / step))
    dw = cfg.get("dip_w", 0.0)
    lam = 1.0 if dw > 0 else 0.0
    dip_b = None
    if dw > 0:                                                    # dips per bin: confidence-weighted mean x B (labels / bin)
        wsum = F.avg_pool2d(Fd["DW"][None], B, ceil_mode=True)[0]
        dip_b = {"DW": wsum,
                 "DY": F.avg_pool2d((Fd["DY"] * Fd["DW"])[None], B, ceil_mode=True)[0] / wsum.clamp(min=1e-6) * B,
                 "DX": F.avg_pool2d((Fd["DX"] * Fd["DW"])[None], B, ceil_mode=True)[0] / wsum.clamp(min=1e-6) * B}

    # the coarse level by belief propagation, its pairwise term the lamination following (a bin step's tolerance:
    # dip_tol, i.e. B times stiffer per cell than the fine level)
    pcs = (lambda: [dw * dip_cost(dip_b, ks, labs, step, oy, ox, Dcl, cfg) for oy, ox in OFFS]) if lam > 0 else [None] * 4
    ew = None
    if lam > 0:                                                   # (the kernel computes the table from its inputs)
        ews = [_dip_ew(dip_b, ks, labs, step, oy, ox) for oy, ox in OFFS]
        ew = (torch.stack([e_ for e_, _ in ews]), torch.stack([w_ for _, w_ in ews]), step / cfg.get("dip_tol", 0.5),
              cfg.get("dip_cap", 4.0), dw)
        del ews
    iters = cfg.get("bp_iters", 0) or int(2 * max(hb, wb))
    Sb = bp(_pin(Ub, pinb, labb), Dcl, cfg["P1c"], pcs, lam, iters, cfg.get("bp_damp", 0.5), ew=ew) * 8.0
    del pcs, ew
    kA = _argmin_sub(Sb, labs)
    # margin (coarse, aggregated): best outside +-L/3 minus best
    kk = torch.as_tensor(labs, device=DEV, dtype=torch.float32)[:, None, None]
    far = (kk - kA[None]).abs() > cfg["L"] / 3
    mgb = (torch.where(far, Sb, torch.full_like(Sb, BIG)).min(0).values - Sb.min(0).values) / 8.0
    kA_c = F.interpolate(kA[None, None], scale_factor=B, mode="bilinear", align_corners=False)[0, 0][:H, :W]
    mg = F.interpolate(mgb[None, None], scale_factor=B, mode="nearest")[0, 0][:H, :W]
    _sync(); timing["coarse_s"] = round(time.time() - t0, 3)
    # fine (SGM-P: the expected slope = the coarse solution's gradient, in labels per cell)
    t0 = time.time()
    ks_s = F.avg_pool2d(F.pad(kA_c[None, None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]
    ey = torch.zeros_like(ks_s); ex = torch.zeros_like(ks_s)
    ey[1:-1] = 0.5 * (ks_s[2:] - ks_s[:-2]); ex[:, 1:-1] = 0.5 * (ks_s[:, 2:] - ks_s[:, :-2])
    ey, ex = ey / step, ex / step
    Cf = FD_blur(U, 0.7)
    lab = torch.as_tensor(labs, device=DEV, dtype=torch.float32)[:, None, None]
    Cf = torch.where((lab - kA_c[None]).abs() <= cfg["fine_rng"], Cf, torch.full_like(Cf, 5.0 * cfg["uscale"]))
    Dfl = int(round(cfg["Df"] / step))

    def pcf_f(oy, ox):                                            # the lamination following, per cell
        return dw * dip_cost(Fd, ks, labs, step, oy, ox, Dfl, cfg)
    lam_f = 1.0 if dw > 0 else 0.0
    dip = None
    if lam_f > 0:
        lidx = torch.as_tensor(np.clip(np.round((np.asarray(labs) - float(ks[0])) / step).astype(int), 0, Fd["DY"].shape[0] - 1), device=DEV)
        dip = dict(DY=Fd["DY"], DX=Fd["DX"], DW=Fd["DW"], lidx=lidx, sct=step / cfg.get("dip_tol", 0.5),
                   cap=cfg.get("dip_cap", 4.0), dw=dw)
    Sf = sgm(_pin(Cf, pins_t, lab_idx), Dfl, cfg["P1f"], ex, ey, pcf=pcf_f if lam_f > 0 else None, lam=lam_f, dip=dip)
    kB = _argmin_sub(Sf, labs)
    _sync(); timing["fine_s"] = round(time.time() - t0, 3)
    # continuous
    t0 = time.time()
    kBn = ndimage.median_filter(kB.cpu().numpy(), 3)
    gexp = None
    if dw > 0 and cfg.get("dip_cont", 0) > 0:                     # expected h differences (vox) from the dips at kB
        li = torch.as_tensor(np.clip(np.round((kBn - float(ks[0])) / step), 0, Fd["DY"].shape[0] - 1).astype(np.int64), device=DEV)
        dyc = Fd["DY"].gather(0, li[None])[0].cpu().numpy() * step; dxc = Fd["DX"].gather(0, li[None])[0].cpu().numpy() * step
        gexp = (0.5 * (dyc[1:] + dyc[:-1]), 0.5 * (dxc[:, 1:] + dxc[:, :-1]))
    kC, d, w = refine_continuous(U, labs, kBn, okc.cpu().numpy(), pins, pin_val, cfg, gexp=gexp)
    timing["cont_s"] = round(time.time() - t0, 3)
    conf = {"margin": mg.cpu().numpy(), "contrast": contrast.cpu().numpy(), "w": w, "kA": kA_c.cpu().numpy(), "kB": kBn}
    return kC, conf


def FD_blur(X, s):
    return L.filt(X, (s, s))


def slab_ks(cfg, kmax):
    """slab depths covering every label +- the data term's reach."""
    step = cfg["step"]
    reach_lo = reach_hi = cfg["J"] * step
    lo = -kmax - reach_lo - 0.5; hi = kmax + reach_hi + 0.5
    return np.arange(math.floor(lo / step) * step, hi + 1e-6, step)


def slant_map(Fd, ks, k, cfg):
    """the SLANT of the solved surface against the CT's laminations at the solved depth k [H, W] (label units of the
    slab Fd): the slab's normals are the surface's own (outer pass), so a surface parallel to its layer reads dips ~0;
    a surface that ramps across layers (a winding jump, "fibres going through fibres") reads the ramp's angle.
    -> (angle deg [H, W], planarity [H, W]) (nan where no dips)."""
    if "DY" not in Fd:
        return None, None
    step = float(cfg["step"])
    idx = np.clip(np.round((np.asarray(k) - float(ks[0])) / step).astype(int), 0, Fd["DY"].shape[0] - 1)
    I, J = np.indices(idx.shape)
    it = torch.as_tensor(idx, device=DEV); Ii = torch.as_tensor(I, device=DEV); Jj = torch.as_tensor(J, device=DEV)
    f = step / L.CELL / cfg.get("dip_gain", 1.0)
    dy = Fd["DY"][it, Ii, Jj] * f; dx = Fd["DX"][it, Ii, Jj] * f
    ang = torch.rad2deg(torch.atan(torch.sqrt(dy ** 2 + dx ** 2))).cpu().numpy()
    w = Fd["DW"][it, Ii, Jj].cpu().numpy()
    return np.where(w > 0, ang, np.nan), w


def solve_patch(P, v, N, region, pins, pin_val, cfg=None, verbose=True):
    """P, v, N: the reference surface (frame). region (r0, r1, c0, c1). pins bool [H, W] (region-relative) at depth
    pin_val. Returns the new positions over the region, k total, conf, timings."""
    cfg = dict(CFG, **(cfg or {}))
    R0, R1, C0, C1 = region
    timing = {}
    ks = slab_ks(cfg, cfg["kmax"])
    t0 = time.time()
    Fd = region_feats(P, v, N, region, ks, cfg)
    _sync(); timing["feats_s"] = round(time.time() - t0, 2)
    okc = Fd["okc"]
    k, conf = solve_levels(Fd, ks, okc, pins, pin_val, cfg, timing)
    del Fd
    vr = v[R0:R1, C0:C1]
    k = np.where(vr, k, 0.0)
    Pn = P.copy()
    Pn[R0:R1, C0:C1] = P[R0:R1, C0:C1] + k[..., None] * N[R0:R1, C0:C1]
    ktot = k.copy()
    # outer iterations: re-sample along the solved surface's own normals, solve the residual
    otol = float(cfg.get("outer_tol", 0.0))
    last_it = cfg["outer"] - 1
    for it in range(cfg["outer"]):
        Nn, _ = L.normals_crop(Pn, v)
        c2 = dict(cfg, kmax=cfg["outer_k"], B=cfg.get("outer_B", 1), Dc=cfg.get("outer_Dc", cfg["Df"]) if cfg.get("outer_B", 1) > 1 else cfg["Df"],
                  scan_pmax=cfg.get("outer_scan_pmax", cfg.get("scan_pmax", 1.0)), scan_np=cfg.get("outer_scan_np", cfg.get("scan_np", 51)))
        for k_ in ("dip_method", "dip_gain", "dip_w", "dip_tol", "dip_tol_c", "dip_cap", "dip_cont", "uscale", "P1c", "P1f", "bar_w"):
            if ("outer_" + k_) in cfg:                              # the re-centred passes may use other terms: once the
                c2[k_] = cfg["outer_" + k_]                         # frame is near the surface the residual slopes are ~0
                                                                    # (the scan's noise at zero tilt
                                                                    # makes blisters; the tensor is accurate there)
        ks2 = slab_ks(cfg, c2["kmax"])
        t0 = time.time()
        Fd = region_feats(Pn, v, Nn, region, ks2, c2)
        tm2 = {}
        pv2 = np.zeros_like(pin_val)                                                      # the pins stay where they are
        k2, conf2 = solve_levels(Fd, ks2, Fd["okc"], pins, pv2, c2, tm2)
        kabs = np.abs(np.where(vr, k2, 0.0)[vr])
        if otol > 0 and it >= int(cfg.get("outer_min", 2)) - 1 and kabs.size and float(np.percentile(kabs, 95)) < otol:
            last_it = it                                                                  # CONVERGED: the pass moved < outer_tol
        if it == last_it:                                                                 # the final surface's slant
            conf["slant"], conf["slant_w"] = slant_map(Fd, ks2, k2, c2)
        del Fd
        k2 = np.where(vr, k2, 0.0)
        Pn = Pn.copy()
        Pn[R0:R1, C0:C1] = Pn[R0:R1, C0:C1] + k2[..., None] * Nn[R0:R1, C0:C1]
        ktot = ktot + k2
        timing[f"outer{it}_s"] = round(time.time() - t0, 2)
        timing[f"outer{it}_k_abs_p50_p95"] = [round(float(np.percentile(np.abs(k2[vr]), 50)), 2), round(float(np.percentile(np.abs(k2[vr]), 95)), 2)]
        conf["contrast_outer"] = conf2["contrast"]; conf["margin_outer"] = conf2["margin"]
        if it == last_it:
            timing["outer_passes"] = it + 1
            break
    if cfg.get("slant_scan"):                                       # the FINAL surface's slant from the semblance scan in
        t0 = time.time()                                            # its own thin slab (the gate's reading must not depend
        Nf, _ = L.normals_crop(Pn, v)                               # on the solve's slope term: the tensor reads an oblique
        cS = dict(cfg, dips_only=True, dip_w=1.0, dip_method=cfg.get("slant_method", "shear"), dip_gain=1.0, kmax=2.0,   # cut flat; no slope term:
                  scan_pmax=cfg.get("outer_scan_pmax", 0.4), scan_np=cfg.get("outer_scan_np", 17))   # no reading at all;
        # slant_method "shear": the scan's zero-tilt noise alone reads ~2.8 deg (the 6 deg gate would then reject
        # 1 500-4 000 cells per patch); the sheared tensor is unbiased at tilt and has the tensor's noise near flat)
        ksS = slab_ks(cS, 2.0)
        FdS = region_feats(Pn, v, Nf, region, ksS, cS)
        conf["slant"], conf["slant_w"] = slant_map(FdS, ksS, np.zeros(FdS["okc"].shape), cS)
        del FdS
        timing["slant_scan_s"] = round(time.time() - t0, 2)
    if verbose:
        print("solve_patch", json.dumps(timing), flush=True)
    return {"P": Pn, "k": ktot, "conf": conf, "timing": timing}


# ------------------------------------------------------------------------------------------------ reference
