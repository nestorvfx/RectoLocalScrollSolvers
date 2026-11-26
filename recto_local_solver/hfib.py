"""THE THICK-HORIZONTAL-FIBRE PRIMITIVES of the detector (hfibwin): render_detect (the fibres on a surface's own
render, the tow tracker on one layer) and detect (the stack read; its 3-D tow tracker places the fibre step's bump).
  slope / straighten               the fibres' common drift; the stack straightened so they run along x
  flow_field, flow_median3d, depth_slope_field   the robust line integral ALONG the fibres (in 3-D, the frame's tilt
                                   followed): crossings, cracks and specks go, the fibre edges stay
  tow_evidence                     the cross-section matched filter (Koller weaker-edge + row / depth contrast)
  tow_extract                      whole-window optimal paths over (width, layer, row) (Viterbi), greedy extraction with
                                   suppression
  tow_refine_batch / _groups, dp_paths   the depth and the two edges of extracted fibres as smooth optimal lines (the
                                   GPU, one pass for many fibres / renders)
  tow_extract_many                 tow_extract for many renders at once (one batched Viterbi per round)
  resolve_overlaps, joint_boundaries   neighbours touch but do not overlap; touching ones share one boundary
  fibre_depth, tracks_table        each fibre's best depth / visibility, and its table in frame stack rows"""
import numpy as np


def _t():
    import torch
    return torch


def slope(S2d, smax=0.08, sstep=0.002):
    """the common slope b (rows per column) for which the fibres run along the columns: the row profile (mean over the
    columns) of the straightened image has the most power (the first maximum over the slopes in ascending order).
    S2d [Y, X] torch (a slab near the surface). The slopes 16 at a time (straighten's arithmetic, one sync)."""
    torch = _t()
    A = S2d - S2d.mean()
    Y, X = A.shape
    x0, x1 = int(X * 0.1), int(X * 0.9)
    ss = np.arange(-smax, smax + 1e-9, sstep)
    xs = torch.arange(X, device=A.device).float()[x0:x1] - X / 2.0
    pws = []
    for a in range(0, len(ss), 16):
        s = torch.as_tensor(ss[a:a + 16], device=A.device, dtype=torch.float32)
        src = torch.arange(Y, device=A.device).float()[None, :, None] + (s[:, None] * xs[None])[:, None, :]
        f0 = src.floor().clamp(0, Y - 2); fr = (src - f0).clamp(0, 1); f0 = f0.long()
        Ae = A[None, :, x0:x1].expand(len(s), Y, x1 - x0)
        pr = (torch.gather(Ae, 1, f0) * (1 - fr) + torch.gather(Ae, 1, f0 + 1) * fr).mean(2)
        pws.append((pr[:, 10:-10] ** 2).mean(1))
    pw = torch.cat(pws).cpu().numpy()
    return float(ss[int(np.argmax(pw))]) if np.isfinite(pw).any() and pw.max() > -1.0 else 0.0


def straighten(A, b, inverse=False):
    """A [..., Y, X] torch: every column moved along the rows by the fibres' common slope b (px / px) about the middle
    column: out[y, x] = A[y + b (x - X / 2), x] (linear; inverse: the other way), so that the fibres run along X."""
    import torch
    Y, X = A.shape[-2:]
    sh = b * (torch.arange(X, device=A.device).float() - X / 2.0) * (-1.0 if inverse else 1.0)
    src = torch.arange(Y, device=A.device).float()[:, None] + sh[None, :]
    f0 = src.floor().clamp(0, Y - 2); fr = (src - f0).clamp(0, 1); f0 = f0.long()
    f0e = f0.expand(*A.shape[:-2], Y, X)
    return torch.gather(A, -2, f0e) * (1 - fr) + torch.gather(A, -2, f0e + 1) * fr


def gauss_blur(A, sy, sx):
    """separable Gaussian blur of A [..., Y, X] torch (reflect padding)."""
    torch = _t()
    from . import kernels as KN
    sh = A.shape
    ks = [(s_, ax, _gk3(s_, A.device)) for s_, ax in ((sy, 0), (sx, 1)) if s_ > 0]
    if A.is_cuda and A.dtype == torch.float32 and KN.available() and all(len(k) // 2 < sh[ax - 2] for _, ax, k in ks):
        def fast():
            B = A.contiguous()
            for _, ax, k in ks:
                B = KN.sepconv(B, k, ax - 2, mode="reflect")
            return B
        return KN.run("sepconv_reflect", fast, lambda: _gauss_blur_torch(A, ks), 1e-5)
    return _gauss_blur_torch(A, ks)


_GK3 = {}


def _gk3(s_, dev):
    """the blur's kernel (radius 3 sigma), cached per (sigma, device)."""
    torch = _t()
    key = (float(s_), str(dev))
    if key not in _GK3:
        r = int(3 * s_ + 0.5); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev).float() / s_) ** 2)
        _GK3[key] = k / k.sum()
    return _GK3[key]


def _gauss_blur_torch(A, ks):
    torch = _t(); F = torch.nn.functional
    sh = A.shape; B = A.reshape(-1, 1, sh[-2], sh[-1])
    for s_, ax, k in ks:
        r = (len(k) - 1) // 2
        if ax == 0:
            B = F.conv2d(F.pad(B, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        else:
            B = F.conv2d(F.pad(B, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return B.reshape(sh)


def flow_field(F0, sg=1.5, st=6.0, smax=0.25, cmin=0.15):
    """the local fibre direction as a slope dy/dx [Y, X] from the structure tensor of F0 [Y, X] (gradient sigma sg,
    integration sigma st); clamped to +-smax; 0 where the coherence < cmin (the window is already straightened)."""
    torch = _t()
    G = gauss_blur(F0, sg, sg)
    gy = torch.zeros_like(G); gx = torch.zeros_like(G)
    gy[1:-1] = 0.5 * (G[2:] - G[:-2]); gx[:, 1:-1] = 0.5 * (G[:, 2:] - G[:, :-2])
    Jxx = gauss_blur(gx * gx, st, st); Jyy = gauss_blur(gy * gy, st, st); Jxy = gauss_blur(gx * gy, st, st)
    phi = 0.5 * torch.atan2(2 * Jxy, Jxx - Jyy)                          # the dominant gradient's angle
    sl = -torch.cos(phi) / torch.sin(phi).abs().clamp_min(1e-3) * torch.sign(torch.sin(phi) + 1e-9)
    coh = ((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) ** 0.5 / (Jxx + Jyy).clamp_min(1e-9)
    sl = torch.where(coh >= cmin, sl, torch.zeros_like(sl))
    return sl.clamp(-smax, smax), coh


def unstraighten_rows(y, x, b, X):
    """row in the straightened frame -> row in the frame stack (straighten: out[y, x] = A[y + b (x - X / 2), x])."""
    return y + b * (x - X / 2.0)


def dp_path(C, x0, lo, hi, sub=4, maxstep=2, lam=0.5):
    """THE OPTIMAL SMOOTH LINE through a cost image: rows y(x) for the consecutive columns x0 .. x0 + n - 1, each inside
    [lo[t], hi[t]] (rows, float), on a 1 / sub px grid; between columns the row moves at most maxstep / sub px at a price
    lam per px moved; minimises sum C(y(x), x) (C [Y, X] numpy, linear interpolation along the rows) + the price.
    Exact (dynamic programming). Returns y [n] (float rows)."""
    Y = C.shape[0]; n = len(lo)
    g0 = np.floor(np.min(lo) * sub) / sub; g1 = np.ceil(np.max(hi) * sub) / sub
    g = np.arange(g0, g1 + 0.5 / sub, 1.0 / sub); m = len(g)
    rows = np.arange(Y, dtype=np.float64)
    inf = 1e18
    D = np.empty((n, m)); P = np.zeros((n, m), np.int32)
    steps = np.arange(-maxstep, maxstep + 1)
    for t in range(n):
        c = np.interp(np.clip(g, 0, Y - 1), rows, C[:, x0 + t])
        c = np.where((g >= lo[t] - 1e-9) & (g <= hi[t] + 1e-9), c, inf)
        if t == 0:
            D[0] = c; continue
        best = np.full(m, inf); arg = np.zeros(m, np.int32)
        for s in steps:                                                   # predecessor at index i - s
            prev = np.full(m, inf)
            if s >= 0:
                prev[s:] = D[t - 1, :m - s] if s > 0 else D[t - 1]
            else:
                prev[:s] = D[t - 1, -s:]
            v = prev + lam * abs(s) / sub
            better = v < best; best = np.where(better, v, best); arg = np.where(better, np.arange(m) - s, arg)
        D[t] = c + best; P[t] = arg
    y = np.empty(n); k = int(np.argmin(D[-1]))
    for t in range(n - 1, -1, -1):
        y[t] = g[k]; k = P[t, k]
    return y


def dp_paths(Cs, los, his, sub=4, maxstep=2, lam=0.5, max_elems=40_000_000):
    """dp_path(C, 0, lo, hi, ...) for MANY independent paths at once: the columns are
    stepped through once for a whole group of paths, on the GPU in float64. Path for path the SAME result as dp_path:
    the same row grids, np.interp's own arithmetic for the costs (exact node values, the last row exact), the same
    predecessor order and strict-improvement tie rule (IEEE float64 add / compare are the same on the GPU), the same
    back-tracking. Cs: list of C [Y_b, n_b] (columns 0 .. n_b - 1 used); los / his: lists of [n_b]. Returns a list of
    y [n_b]. Paths of similar size are grouped (<= max_elems cost entries per group)."""
    torch = _t()
    from . import kernels as KN
    B = len(Cs)
    if B == 0:
        return []
    from . import volume as qc
    dev = qc.DEV
    inf = 1e18; BIG = 1e30
    gs = []
    for lo, hi in zip(los, his):
        g0 = np.floor(np.min(lo) * sub) / sub; g1 = np.ceil(np.max(hi) * sub) / sub
        gs.append(np.arange(g0, g1 + 0.5 / sub, 1.0 / sub))
    steps = list(range(-maxstep, maxstep + 1)); pen = [lam * abs(s_) / sub for s_ in steps]
    order = sorted(range(B), key=lambda b: (len(los[b]), len(gs[b])))
    groups, cur = [], []
    for b_ in order:                                                      # groups of similar size, bounded memory
        nn_ = max([len(los[q]) for q in cur + [b_]]); mm_ = max([len(gs[q]) for q in cur + [b_]])
        if cur and (len(cur) + 1) * nn_ * mm_ > max_elems:
            groups.append(cur); cur = []
        cur.append(b_)
    groups.append(cur)
    out = [None] * B
    for gi in groups:
        ms = [len(gs[b_]) for b_ in gi]; ns = [len(los[b_]) for b_ in gi]; m, n = max(ms), max(ns); G_ = len(gi)
        cost = np.full((G_, n, m), BIG)
        for j_, b_ in enumerate(gi):
            C = np.asarray(Cs[b_], np.float64); Y = C.shape[0]; g = gs[b_]; nb = ns[j_]; Cb = C[:, :nb]
            x = np.clip(g, 0, Y - 1)
            j = np.minimum(np.floor(x).astype(np.int64), max(Y - 2, 0))
            if Y > 1:
                c = (Cb[j + 1] - Cb[j]) * (x - j)[:, None] + Cb[j]        # np.interp: slope (dx = 1) x (x - x_j) + y_j
                c = np.where((x == j)[:, None], Cb[j], c)                  # exactly on a node: y_j
                c = np.where((x >= Y - 1)[:, None], Cb[Y - 1][None, :], c)  # at / past the last node: y_last
            else:
                c = np.repeat(Cb[0][None, :], len(g), 0)
            lo_ = np.asarray(los[b_], float); hi_ = np.asarray(his[b_], float)
            band = (g[None, :] >= lo_[:, None] - 1e-9) & (g[None, :] <= hi_[:, None] + 1e-9)
            cost[j_, :nb, :len(g)] = np.where(band, c.T, inf)
        ct = torch.as_tensor(cost, device=dev)
        if ct.is_cuda and KN.available():                                  # the forward pass in one kernel launch
            Dlt, Pkt = KN.run("dp_paths", lambda: KN.dp_groups(ct, ns, steps, pen),
                              lambda: _dp_forward(ct, ns, steps, pen, inf, BIG), 0.0)
        else:
            Dlt, Pkt = _dp_forward(ct, ns, steps, pen, inf, BIG)
        Dl = Dlt.cpu().numpy()
        start = np.array([int(np.argmin(Dl[j_, :ms[j_]])) for j_ in range(G_)], np.int32)   # every path's best end

        def ref(Pkt=Pkt, start=start):
            Pn = Pkt.cpu().numpy(); Kt = np.full((G_, n), -1, np.int32)
            for j_ in range(G_):
                k = int(start[j_])
                for t in range(ns[j_] - 1, -1, -1):
                    Kt[j_, t] = k
                    si = int(Pn[j_, t, k]); k = (k - steps[si]) if si >= 0 else 0
            return torch.as_tensor(Kt)
        if Pkt.is_cuda and KN.available():                                 # (one thread per path walks back)
            Kt = KN.run("dp_backtrack", lambda: KN.dp_backtrack(Pkt, start, ns, steps).cpu(), ref, 0.0).numpy()
        else:
            Kt = ref().numpy()
        for j_, b_ in enumerate(gi):
            out[b_] = gs[b_][Kt[j_, :ns[j_]]]
    return out


def _dp_forward(ct, ns, steps, pen, inf, BIG):
    """dp_paths's forward pass in PyTorch: ct [G, n, m] float64 -> Dlast [G, m], Pk [G, n, m] int8."""
    torch = _t()
    G_, n, m = ct.shape; dev = ct.device
    nsT = torch.as_tensor(ns, device=dev)
    D = ct[:, 0].clone(); Dlast = torch.full((G_, m), BIG, dtype=torch.float64, device=dev)
    one = nsT == 1
    if bool(one.any()):
        Dlast[one] = D[one]
    Pk = torch.full((G_, n, m), -1, dtype=torch.int8, device=dev)
    for t in range(1, n):
        best = torch.full((G_, m), inf, dtype=torch.float64, device=dev); sidx = torch.full((G_, m), -1, dtype=torch.int8, device=dev)
        for si, s_ in enumerate(steps):                               # predecessor at index i - s
            if s_ == 0:
                prev = D
            else:
                prev = torch.full((G_, m), inf, dtype=torch.float64, device=dev)
                if s_ > 0:
                    prev[:, s_:] = D[:, :m - s_]
                else:
                    prev[:, :s_] = D[:, -s_:]
            v = prev + pen[si]
            better = v < best
            best = torch.where(better, v, best); sidx = torch.where(better, torch.full_like(sidx, si), sidx)
        D = ct[:, t] + best; Pk[:, t] = sidx
        fin = (nsT - 1) == t
        if bool(fin.any()):
            Dlast[fin] = D[fin]
    return Dlast, Pk




def fibre_depth(Ib, offs, tracks, flank=3, dmin=-4.0, dmax=4.0):
    """EACH FIBRE'S OWN DEPTH, column by column: Ib [K, Y, Xs] the straightened stack block-averaged along the columns
    (the tracks' node x indexes its blocks; rows in the same straightened rows), offs [K] the layers' offsets (vox).
    Per node, the fibre's CONTRAST at each depth = mean over its band (top .. bot) minus the mean of the two flanks
    (flank px outside each edge), on the render's own 3-layer average (d - 1, d, d + 1 vox); the best depth in
    [dmin, dmax]; VISIBILITY = the contrast at d = 0 (what the render shows today) / the best contrast; BRIGHT = the band's
    mean at the best depth / at 0. Writes vis, bright, cbest, cren into the nodes (their depth d stays the tracker's).
    Every node at once (the band and flank sums from cumulative sums)."""
    from scipy.ndimage import uniform_filter1d
    K, Y, Xs = Ib.shape
    stepk = float(np.median(np.diff(offs)))
    A = uniform_filter1d(Ib, size=max(1, int(round(2.0 / stepk)) + 1), axis=0, mode="nearest")    # +-1 vox average
    sel = np.nonzero((offs >= dmin) & (offs <= dmax))[0]; k0 = int(np.argmin(np.abs(offs)))
    nodes = [n for tr in tracks for n in tr]
    if not nodes:
        return tracks
    xi = np.minimum(np.array([int(n["x"]) for n in nodes]), Xs - 1)
    t0 = np.maximum(np.floor([n["top"] for n in nodes]).astype(int), 0)
    t1 = np.minimum(np.ceil([n["bot"] for n in nodes]).astype(int) + 1, Y)
    CS = np.concatenate([np.zeros((K, 1, Xs)), np.cumsum(A, axis=1, dtype=np.float64)], 1)            # [K, Y + 1, Xs]
    rows = lambda r0, r1: CS[:, np.clip(r1, 0, Y), xi] - CS[:, np.clip(r0, 0, Y), xi]                # [K, n] sums
    nbd = t1 - t0
    with np.errstate(invalid="ignore", divide="ignore"):
        band = np.where(nbd > 0, rows(t0, t1) / np.maximum(nbd, 1), np.nan)
        fa0 = np.maximum(t0 - flank, 0); fb1 = np.minimum(t1 + flank, Y)
        na, nb = np.maximum(np.minimum(t0, Y) - np.minimum(fa0, Y), 0), np.maximum(fb1 - t1, 0)
        fl = np.where(na + nb > 0, (rows(fa0, t0) * (na > 0) + rows(t1, fb1) * (nb > 0)) / np.maximum(na + nb, 1), 0.0)
        c = band - fl                                                                                  # [K, n]
        csel = c[sel]
        jj = sel[np.argmax(np.where(np.isnan(csel), np.inf, csel), axis=0)]                            # (np.argmax: nan first)
        ar = np.arange(len(nodes))
        cj, ck = c[jj, ar], c[k0, ar]
        vis = np.where(cj > 1e-6, ck / np.where(cj > 1e-6, cj, 1.0), 0.0)
        bright = band[jj, ar] / np.maximum(band[k0, ar], 1e-6)
    for i, n in enumerate(nodes):
        n["vis"] = float(vis[i]); n["bright"] = float(bright[i]); n["cbest"] = float(cj[i]); n["cren"] = float(ck[i])
    return tracks


# ---------------------------------------------------------------------------------------------------------------------
# THE TOW TRACKER. What hand-painted windows show: every
# painted thick fibre runs the whole 160-cell window; in the straightened stack its cross-section (rows x depth) is a
# compact bright blob ~2 vox thick and 10-27 vox wide, ~+0.5..+1.5 vox from the surface; the per-column best depth is
# noise (jumps over +-4 vox) and only the long run identifies it. So: a cross-section MATCHED FILTER (bright core,
# darker rows on both sides and darker layers above / below) for every (width, depth, row), then each fibre as ONE
# globally optimal path along the whole window through (row, depth, width) states (Viterbi, smoothness priced),
# extracted greedily with the found bands suppressed, trimmed where its evidence ends, its depth read over a long
# stretch, its two edges as smooth optimal lines on its own depth.

def box_axis(T, a, b, dim):
    """mean of T over the index window [i - a, i + b] along `dim` (a, b may be negative for one-sided windows), the
    window clipped at the borders (mean over what is inside; empty -> the nearest border value)."""
    torch = _t()
    n = T.shape[dim]
    cs = torch.cat([torch.zeros_like(T.narrow(dim, 0, 1)), T.cumsum(dim)], dim)
    i = torch.arange(n, device=T.device)
    lo = (i - a).clamp(0, n); hi = (i + b + 1).clamp(0, n)
    hi = torch.maximum(hi, (lo + 1).clamp(max=n)); lo = torch.minimum(lo, hi - 1)
    shp = [1] * T.dim(); shp[dim] = n
    s = cs.index_select(dim, hi) - cs.index_select(dim, lo)
    return s / (hi - lo).clamp_min(1).reshape(shp).to(T.dtype)


def tow_evidence(Ib, widths, kcore=2, kfl=(3, 6), fw=4, beta=0.5, mode="box", gamma=0.5, etol=1, esig=1.0, floor=None):
    """THE CROSS-SECTION MATCHED FILTER. Ib [K, Y, Xb] (torch; the straightened stack, robustly smoothed along the
    fibres, block-averaged along x). Per width w (rows, odd): core = mean over rows y +- w//2 and layers k +- kcore;
    row contrast RC = core - the mean of the two flanks (fw rows just outside the core, same layers); depth contrast =
    core - the mean over the same rows of the layers k - kfl[1] .. k - kfl[0] and k + kfl[0] .. k + kfl[1].
    mode "box":    E = RC + beta * depth contrast (each in intensity units).
    mode "koller": the band's two EDGES must both be strong (Koller et al. 1995, the weaker of the two): the row
                   gradient of the depth core (smoothed esig px) as a RISE at y - h (+-etol) and a FALL at y + h
                   (+-etol), EK = min(rise, fall); E = EK / mad(EK) + gamma * RC / mad(RC) + beta * DC / mad(DC):
                   a sub-band cut off by a faint internal line loses (its inner edge is weak), the whole fibre between
                   two strong dark gaps wins.
    floor: intensities below it are raised to it first (voids must not make a neighbour look contrasty).
    Returns E [Wn, K, Y, Xb] and RC (for depth reading)."""
    torch = _t()
    if floor is not None:
        Ib = Ib.clamp_min(floor)
    Ck = box_axis(Ib, kcore, kcore, 0)
    Dk = 0.5 * (box_axis(Ib, kfl[1], -kfl[0], 0) + box_axis(Ib, -kfl[0], kfl[1], 0))
    if mode == "koller":
        Cs = gauss_blur(Ck, esig, 0.0) if esig > 0 else Ck                 # [K, Y, Xb]: smoothed across the rows
        Gy = torch.zeros_like(Cs); Gy[:, 1:-1] = 0.5 * (Cs[:, 2:] - Cs[:, :-2])
        Gu = Gy.clone(); Gd = (-Gy).clone()
        for _ in range(etol):                                             # max over +-etol rows
            Gu = _shiftmax(Gu, 1, 0.0); Gd = _shiftmax(Gd, 1, 0.0)
    Es, Rs, Ks, Ds = [], [], [], []
    for w in widths:
        h = w // 2
        C = box_axis(Ck, h, h, 1)
        Ra = box_axis(Ck, h + fw, -h - 1, 1); Rb = box_axis(Ck, -h - 1, h + fw, 1)
        D = box_axis(Dk, h, h, 1)
        rc = C - 0.5 * (Ra + Rb); dc = C - D
        Rs.append(rc); Ds.append(dc)
        if mode == "koller":
            n = Gu.shape[1]
            up_ = torch.roll(Gu, h, 1); dn_ = torch.roll(Gd, -h, 1)               # rise at y - h, fall at y + h
            up_[:, :h] = 0.0; dn_[:, n - h:] = 0.0
            Ks.append(torch.minimum(up_, dn_))
        else:
            Es.append(rc + beta * dc)
    if mode == "koller":
        Kt, Rt, Dt = torch.stack(Ks), torch.stack(Rs), torch.stack(Ds)
        mad = lambda T: 1.4826 * (T[len(widths) // 2] - T[len(widths) // 2].median()).abs().median().clamp_min(1e-6)
        return Kt / mad(Kt) + gamma * Rt / mad(Rt) + beta * Dt / mad(Dt), Rt
    return torch.stack(Es), torch.stack(Rs)


def _shiftmax(T, dim, lam):
    """max over shifts -1, 0, +1 along dim of T shifted, minus lam per unit shift (borders: -inf)."""
    torch = _t()
    n = T.shape[dim]; ninf = torch.full_like(T.narrow(dim, 0, 1), -1e30)
    up = torch.cat([T.narrow(dim, 1, n - 1), ninf], dim) - lam
    dn = torch.cat([ninf, T.narrow(dim, 0, n - 1)], dim) - lam
    return torch.maximum(T, torch.maximum(up, dn))


def _ends_paths(F, lam, ry=1, rk=1):
    """for every volume of F [B, Xb, Wn, K, Y] (torch): the best end state (the FIRST maximum of the last column, in
    C order) and its back-tracked path [Xb, 3] (w, k, y), or None where the best end score is < -1e5 (nothing left)."""
    torch = _t()
    from . import kernels as KN
    B, Xb, Wn, K, Y = F.shape
    Fl = F[:, -1].reshape(B, -1)
    mx = Fl.max(1).values
    ar = torch.arange(Fl.shape[1], device=F.device)
    first = torch.where(Fl == mx[:, None], ar[None], torch.full_like(ar, Fl.shape[1])[None]).min(1).values
    mx, first = mx.cpu().numpy(), first.cpu().numpy()
    live = [j for j in range(B) if float(mx[j]) >= -1e5]
    ends = np.array([np.unravel_index(int(first[j]), (Wn, K, Y)) for j in live], np.int64).reshape(-1, 3)
    out = [None] * B
    if not live:
        return out
    Fs = F[torch.as_tensor(live, device=F.device)]
    def ref():
        Fn = Fs.cpu().numpy()
        return np.stack([np.asarray(_backtrack(Fn[i], None, lam, tuple(int(q) for q in ends[i]), ry=ry, rk=rk)) for i in range(len(live))])
    if F.is_cuda and F.dtype == torch.float32 and len(live) <= 1024 and KN.available():
        P = KN.run("viterbi_backtrack", lambda: torch.as_tensor(KN.viterbi_backtrack(Fs, ends, lam, ry, rk)),
                   lambda: torch.as_tensor(ref().astype(np.int32)), 0.0).numpy()
    else:
        P = ref()
    for i, j in enumerate(live):
        out[j] = P[i]
    return out


def _backtrack(F, Sc, lam, end_state, ry=1, rk=1):
    """the best path ending in end_state (w, k, y) at the last x: predecessors re-derived from F (numpy)."""
    Xb = F.shape[0]; Wn, K, Y = F.shape[1:]
    path = [end_state]
    w, k, y = end_state
    for t in range(Xb - 1, 0, -1):
        best = None
        for dw in (-1, 0, 1):
            for dk in range(-rk, rk + 1):
                for dy in range(-ry, ry + 1):
                    w2, k2, y2 = w + dw, k + dk, y + dy
                    if 0 <= w2 < Wn and 0 <= k2 < K and 0 <= y2 < Y:
                        v = float(F[t - 1, w2, k2, y2]) - lam[0] * abs(dw) - lam[1] * abs(dk) - lam[2] * abs(dy)
                        if best is None or v > best[0]:
                            best = (v, w2, k2, y2)
        _, w, k, y = best
        path.append((w, k, y))
    return path[::-1]


def tow_extract(E, widths, ksel, valid, lam=(0.3, 0.3, 0.5), clip=1.0, tau=0.5, keep_tau=0.6, smooth=9, gap=12,
                min_len=40, max_fibres=20, margin=1, ry=1, rk=1, allow=None):
    """GREEDY GLOBAL EXTRACTION OF THE FIBRES. E [Wn, K, Y, Xb] normalised evidence (torch); ksel: the layer indices
    the fibres may sit at (e.g. |offset| <= 4 vox); valid [Xb] bool (columns with data). Score = clip(E, -clip) - tau
    over ksel (invalid columns: -clip - tau: crossed at a bounded price). Repeat: Viterbi over the whole window; the
    best path; TRIM it: its evidence smoothed over `smooth` blocks, kept where >= keep_tau, holes <= gap blocks filled,
    runs >= min_len blocks kept; its kept bands (rows y +- w/2 + margin, all widths' states overlapping it, all depths)
    suppressed; until no run survives or max_fibres. Returns a list of runs: dict(t0, t1, y[], k[], w[], e[]).
    rk: layers a path may move per knot. allow [len(ksel), Y, Xb] bool (torch) or None: states outside it are forbidden
    (the solver's ANCHOR: in the context's cells a fibre must lie on the context's recto layer)."""
    torch = _t()
    from scipy.ndimage import uniform_filter1d
    Es = E[:, ksel].clone()
    vmask = torch.as_tensor(valid, device=E.device)
    Sc = torch.where(vmask, Es.clamp_min(-clip), torch.full_like(Es, -clip)) - tau
    if allow is not None:
        Sc = torch.where(allow[None], Sc, torch.full_like(Sc, -1e6))
    Wn, K, Y, Xb = Sc.shape
    wt = torch.as_tensor(np.asarray(widths, float), device=E.device, dtype=Sc.dtype)
    yy = torch.arange(Y, device=E.device, dtype=Sc.dtype)
    runs = []
    for it in range(max_fibres):
        F = _viterbi_fwd_many(Sc[None], lam, ry=ry, rk=rk)[0]
        path = _ends_paths(F[None], lam, ry=ry, rk=rk)[0]
        if path is None:
            break
        pw, pk, py = (np.asarray(path[:, i], np.int64) for i in range(3))
        ti = torch.arange(Xb, device=E.device)
        ev = Es[torch.as_tensor(pw, device=E.device), torch.as_tensor(pk, device=E.device), torch.as_tensor(py, device=E.device), ti].cpu().numpy().astype(float)
        ev = np.where(valid, ev, np.nan)
        evs = uniform_filter1d(np.nan_to_num(ev, nan=0.0), smooth, mode="nearest") / np.maximum(uniform_filter1d(np.isfinite(ev).astype(float), smooth, mode="nearest"), 1e-6)
        keep = evs >= keep_tau
        # fill short holes, drop short runs
        idx = np.nonzero(keep)[0]
        if len(idx):
            segs = np.split(idx, np.nonzero(np.diff(idx) > gap + 1)[0] + 1)
            segs = [s for s in segs if s[-1] - s[0] + 1 >= min_len]
        else:
            segs = []
        if not segs:
            break
        for s in segs:
            t0, t1 = int(s[0]), int(s[-1])
            runs.append(dict(t0=t0, t1=t1, y=py[t0:t1 + 1].astype(float), k=np.asarray(ksel)[pk[t0:t1 + 1]].astype(float),
                             w=np.asarray(widths, float)[pw[t0:t1 + 1]], e=ev[t0:t1 + 1], score=float(np.nanmean(ev[t0:t1 + 1])), order=it))
            # suppress every state whose band overlaps (all of the run's columns at once: the same mask as per column)
            pyt = torch.as_tensor(py[t0:t1 + 1], device=E.device, dtype=Sc.dtype)[:, None, None]
            ht = torch.as_tensor([0.5 * float(widths[pw[t]]) + margin for t in range(t0, t1 + 1)], device=E.device, dtype=Sc.dtype)[:, None, None]
            ov = (yy[None, None, :] - pyt).abs() < (ht + 0.5 * wt[None, :, None])          # [T, Wn, Y]
            seg = Sc[:, :, :, t0:t1 + 1]
            Sc[:, :, :, t0:t1 + 1] = torch.where(ov.permute(1, 2, 0)[:, None], torch.full_like(seg, -1e6), seg)
    return runs


def _viterbi_fwd_many(Sc, lam=(0.3, 0.3, 0.5), ry=1, rk=1):
    """the max-sum DP over x of the score volumes Sc [B, Wn, K, Y, Xb]: F [B, Xb, Wn, K, Y], F[t] = Sc[t] + the max over the
    neighbouring states of F[t - 1] (rows +-ry, layers +-rk, widths +-1) minus lam per step (separable)."""
    torch = _t()
    from . import kernels as KN
    if Sc.is_cuda and Sc.dtype == torch.float32 and KN.available():
        return KN.run("viterbi", lambda: KN.viterbi_forward(Sc, lam, ry, rk), lambda: _viterbi_fwd_torch(Sc, lam, ry, rk), 0.0)
    return _viterbi_fwd_torch(Sc, lam, ry, rk)


def _viterbi_fwd_torch(Sc, lam, ry, rk):
    torch = _t()
    Xb = Sc.shape[-1]
    F = torch.empty((Sc.shape[0], Xb) + tuple(Sc.shape[1:-1]), device=Sc.device, dtype=Sc.dtype)
    def tr(T):
        for _ in range(ry):
            T = _shiftmax(T, 3, lam[2])
        for _ in range(rk):
            T = _shiftmax(T, 2, lam[1])
        return _shiftmax(T, 1, lam[0])
    F[:, 0] = Sc[..., 0]
    for t in range(1, Xb):
        F[:, t] = Sc[..., t] + tr(F[:, t - 1])
    return F


def tow_extract_many(Es, widths, ksel, valids, lam=(0.3, 0.3, 0.5), clip=1.0, tau=0.5, keep_tau=0.6, smooth=9, gap=12,
                     min_len=40, max_fibres=20, margin=1, ry=1, rk=1):
    """tow_extract (allow = None) for SEVERAL evidence volumes of one shape at once (e.g.
    the renders of many candidate surfaces): each extraction round runs ONE batched Viterbi for all volumes still
    extracting; volume for volume the same runs as tow_extract (the same scores, the same first-maximum end state, the
    same back-tracking, trimming and suppression). Es: list of E [Wn, K, Y, Xb] torch; valids: list of [Xb] bool."""
    torch = _t()
    from scipy.ndimage import uniform_filter1d
    B = len(Es)
    if B == 0:
        return []
    dev = Es[0].device
    Ess = [E[:, ksel].clone() for E in Es]
    Sc = torch.stack([torch.where(torch.as_tensor(vd, device=dev), E_.clamp_min(-clip), torch.full_like(E_, -clip)) - tau
                      for E_, vd in zip(Ess, valids)])
    _, Wn, K, Y, Xb = Sc.shape
    wt = torch.as_tensor(np.asarray(widths, float), device=dev, dtype=Sc.dtype)
    yy = torch.arange(Y, device=dev, dtype=Sc.dtype)
    ti = torch.arange(Xb, device=dev)
    runs = [[] for _ in range(B)]; active = list(range(B))
    for it in range(max_fibres):
        if not active:
            break
        F = _viterbi_fwd_many(Sc[active], lam, ry=ry, rk=rk)
        paths = _ends_paths(F, lam, ry=ry, rk=rk); still = []
        for j, b in enumerate(active):
            if paths[j] is None:
                continue
            pw, pk, py = (np.asarray(paths[j][:, i], np.int64) for i in range(3))
            ev = Ess[b][torch.as_tensor(pw, device=dev), torch.as_tensor(pk, device=dev), torch.as_tensor(py, device=dev), ti].cpu().numpy().astype(float)
            valid = valids[b]
            ev = np.where(valid, ev, np.nan)
            evs = uniform_filter1d(np.nan_to_num(ev, nan=0.0), smooth, mode="nearest") / np.maximum(uniform_filter1d(np.isfinite(ev).astype(float), smooth, mode="nearest"), 1e-6)
            keep = evs >= keep_tau
            idx = np.nonzero(keep)[0]
            if len(idx):
                segs = np.split(idx, np.nonzero(np.diff(idx) > gap + 1)[0] + 1)
                segs = [s_ for s_ in segs if s_[-1] - s_[0] + 1 >= min_len]
            else:
                segs = []
            if not segs:
                continue
            still.append(b)
            for s_ in segs:
                t0, t1 = int(s_[0]), int(s_[-1])
                runs[b].append(dict(t0=t0, t1=t1, y=py[t0:t1 + 1].astype(float), k=np.asarray(ksel)[pk[t0:t1 + 1]].astype(float),
                                    w=np.asarray(widths, float)[pw[t0:t1 + 1]], e=ev[t0:t1 + 1], score=float(np.nanmean(ev[t0:t1 + 1])), order=it))
                # suppress every state whose band overlaps (all of the run's columns at once: the same mask as per column)
                pyt = torch.as_tensor(py[t0:t1 + 1], device=dev, dtype=Sc.dtype)[:, None, None]
                ht = torch.as_tensor([0.5 * float(widths[pw[t]]) + margin for t in range(t0, t1 + 1)], device=dev, dtype=Sc.dtype)[:, None, None]
                ov = (yy[None, None, :] - pyt).abs() < (ht + 0.5 * wt[None, :, None])          # [T, Wn, Y]
                seg = Sc[b, :, :, :, t0:t1 + 1]
                Sc[b, :, :, :, t0:t1 + 1] = torch.where(ov.permute(1, 2, 0)[:, None], torch.full_like(seg, -1e6), seg)
        active = still
    return runs


def _refine_depth(offs, run, dsm=5, dmax=4.0, dlam=None, dsel=None, kpath=None):
    """one extracted fibre's depth line: the row-contrast profile over k at the path's (y, w), averaged along the fibre;
    with dlam (and >= 3 blocks) the depth as ONE smooth optimal line along the fibre: returns dict(dp=(cost, lo, hi))
    for dp_path(cost, 0, lo, hi, sub=4, maxstep=dms, lam=dlam) (row k -> offs[0] + k step); else dict(d=..): the best
    layer per block (sub-layer parabola, median of 5)."""
    from scipy.ndimage import median_filter
    K = len(offs)
    n = run["t1"] - run["t0"] + 1
    stepk = float(np.median(np.diff(offs)))
    prof = run["rc"]                                                      # [n, K] (filled by the caller)
    pa = np.stack([prof[max(0, i - dsm):i + dsm + 1].mean(0) for i in range(n)])
    sel = np.nonzero(np.abs(offs) <= dmax)[0] if dsel is None else np.nonzero((offs >= dsel[0]) & (offs <= dsel[1]))[0]
    if dlam is not None and n >= 3:
        Cz = -(pa / max(np.abs(pa[:, sel]).max(), 1e-9)).T                # sub-layer
        if kpath is not None:                                             # the depth read near the extraction path's own
            kp = np.asarray(run["k"], float); wk = kpath / stepk             # depth (the identity), within +- kpath vox
            klo = np.clip(kp - wk, sel[0], sel[-1]); khi = np.clip(kp + wk, sel[0], sel[-1])
        else:
            klo, khi = np.full(n, float(sel[0])), np.full(n, float(sel[-1]))
        return dict(dp=(Cz, klo, khi), off0=float(offs[0]), stepk=stepk)
    d = np.empty(n)
    for i in range(n):
        j = sel[int(np.argmax(pa[i, sel]))]; dj = float(offs[j])
        if 0 < j < K - 1:
            y0, y1, y2 = pa[i, j - 1], pa[i, j], pa[i, j + 1]; den = y0 - 2 * y1 + y2
            if abs(den) > 1e-12:
                dj += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * stepk
        d[i] = dj
    return dict(d=median_filter(d, size=min(5, n), mode="nearest"))


def _refine_prep(Ism, offs, run, S, Y, d, kwin=1.0, edge_pad=4.0, gsy=1.0, gsx=2.0):
    """one extracted fibre (run) with its depth line d (_refine_depth) refined up to its two edge paths: the fibre's own
    image and the edge cost / band of each (each edge searched from up to 2 px from the run's centre to edge_pad px
    beyond its width); gsy / gsx: the edge image's Gaussian across / along the fibre (px)."""
    from scipy.ndimage import gaussian_filter1d
    K, Yi, X = Ism.shape
    t0, t1 = run["t0"], run["t1"]; n = t1 - t0 + 1
    # the fibre's own image at its depth, every column
    xa, xb = t0 * S, min((t1 + 1) * S, X) - 1; cols = np.arange(xa, xb + 1)
    tb = (cols - xa) / S                                                  # block coordinate along the run
    dcol = np.interp(tb, np.arange(n), d); ycol = np.interp(tb, np.arange(n), run["y"]); wcol = np.interp(tb, np.arange(n), run["w"])
    lo_t = np.clip(ycol - 0.5 * wcol - edge_pad, 0, Yi - 1); hi_t = np.clip(ycol - 2.0, 0, Yi - 1)
    lo_b = np.clip(ycol + 2.0, 0, Yi - 1); hi_b = np.clip(ycol + 0.5 * wcol + edge_pad, 0, Yi - 1)
    # the image only on the rows the edges can use + 8 (the filters below reach 5 rows): the same values there
    r0 = max(int(np.floor(min(lo_t.min(), lo_b.min()))) - 8, 0); r1 = min(int(np.ceil(max(hi_t.max(), hi_b.max()))) + 8, Yi - 1)
    Fd = np.empty((r1 - r0 + 1, len(cols)))
    if K == 1 and bool(np.all(np.abs(offs[0] - dcol) <= kwin)):          # one layer (the render): the same values at once
        Fd[:] = Ism[0][r0:r1 + 1][:, cols]
    else:                                                                 # per column the mean of the layers within kwin
        M = np.abs(offs[None, :] - dcol[:, None]) <= kwin                 # of its depth (else the nearest layer); the
        pat, inv = np.unique(M, axis=0, return_inverse=True)              # columns of one layer set at once
        inv = inv.reshape(-1)
        for g, ks in enumerate(pat):
            js = np.nonzero(inv == g)[0]
            if ks.any():
                Fd[:, js] = Ism[np.ix_(np.nonzero(ks)[0], np.arange(r0, r1 + 1), cols[js])].mean(0)   # (the block only)
            else:
                for j in js:
                    Fd[:, j] = Ism[int(np.argmin(np.abs(offs - dcol[j]))), r0:r1 + 1, cols[j]]
    G = gaussian_filter1d(np.gradient(gaussian_filter1d(Fd, gsy, axis=0), axis=0), gsx, axis=1)
    return dict(G=G, r0=r0, lo_t=lo_t, hi_t=hi_t, lo_b=lo_b, hi_b=hi_b, t0=t0, n=n, xa=xa, cols=cols, d=d, Fd=Fd, ycol=ycol, Yi=Yi)


def _refine_nodes(run, S, q, top, bot):
    """a refined fibre's nodes (centre row, width, depth per column) from its two edge paths (q: _refine_prep's record;
    Fd starts at row q["r0"])."""
    t0, n, xa, cols, d, Fd, ycol, Yi, r0 = q["t0"], q["n"], q["xa"], q["cols"], q["d"], q["Fd"], q["ycol"], q["Yi"], q["r0"]
    nodes = []
    for i in range(n):
        c = (t0 + i) * S; j = c - xa
        if j < 0 or j >= len(cols):
            continue
        nodes.append(dict(x=t0 + i, yc=0.5 * (top[j] + bot[j]), top=float(top[j]), bot=float(bot[j]), w=float(bot[j] - top[j]), d=float(d[i]),
                          dlo=0.0, dhi=0.0, lvl=float(Fd[int(np.clip(round(ycol[j]), 0, Yi - 1)) - r0, j]), prom=float(run["e"][i]) if np.isfinite(run["e"][i]) else 0.0,
                          ypath=float(run["y"][i]), wpath=float(run["w"][i])))
    return nodes


def tow_refine_groups(groups, lam=0.5, **kw):
    """the extracted fibres of SEVERAL renders / stacks (groups: list of (Ism, offs, runs, S, Y), e.g. many candidate
    surfaces) refined to their two edges, with ALL their edge paths in one dp_paths call."""
    dkw = {k: kw[k] for k in ("dsm", "dmax", "dlam", "dsel", "kpath") if k in kw}
    pkw = {k: kw[k] for k in ("kwin", "edge_pad", "gsy", "gsx") if k in kw}
    deps = [[_refine_depth(offs, r, **dkw) for r in runs] for (Ism, offs, runs, S, Y) in groups]
    req = [q for dl in deps for q in dl if "dp" in q]                     # every depth line in one dp_paths call
    if req:
        kks = dp_paths([q["dp"][0] for q in req], [q["dp"][1] for q in req], [q["dp"][2] for q in req], sub=4,
                       maxstep=kw.get("dms", 2), lam=kw["dlam"])
        for q, kk in zip(req, kks):
            q["d"] = q["off0"] + kk * q["stepk"]
    prs = [[_refine_prep(Ism, offs, r, S, Y, q["d"], **pkw) for r, q in zip(runs, dl)]
           for (Ism, offs, runs, S, Y), dl in zip(groups, deps)]
    Cs, los, his = [], [], []
    for pr in prs:
        for q in pr:
            Cs += [-q["G"], q["G"]]; los += [q["lo_t"] - q["r0"], q["lo_b"] - q["r0"]]; his += [q["hi_t"] - q["r0"], q["hi_b"] - q["r0"]]
    ys = dp_paths(Cs, los, his, sub=4, maxstep=2, lam=lam)
    out = []; i = 0
    for (Ism, offs, runs, S, Y), pr in zip(groups, prs):
        nl = []
        for r, q in zip(runs, pr):
            nl.append(_refine_nodes(r, S, q, ys[i] + q["r0"], ys[i + 1] + q["r0"])); i += 2
        out.append(nl)
    return out


def tow_refine_batch(Ism, offs, runs, S, Y, lam=0.5, **kw):
    """tow_refine_groups for the fibres of one render / stack."""
    return tow_refine_groups([(Ism, offs, runs, S, Y)], lam=lam, **kw)[0]


def depth_slope_field(St, sg=1.5, st=20.0, smax=0.8):
    """THE FRAME'S TILT AGAINST THE PLIES, as layers per px along x, per (row, column): St [K, Y, X] torch. The plies
    are layered along the depth; in a tilted frame a layer boundary I = f(k - s x) gives grad_x = -s grad_k, so
    s = -J_xk / J_kk with J summed over the layers (the face's strong depth gradient dominates) and smoothed st px."""
    torch = _t()
    G = gauss_blur(St, sg, sg)
    gk = torch.zeros_like(G); gx = torch.zeros_like(G)
    gk[1:-1] = 0.5 * (G[2:] - G[:-2]); gx[..., 1:-1] = 0.5 * (G[..., 2:] - G[..., :-2])
    Jkk = gauss_blur((gk * gk).sum(0), st, st); Jxk = gauss_blur((gx * gk).sum(0), st, st)
    return (-Jxk / Jkk.clamp_min(1e-9)).clamp(-smax, smax)


def flow_median3d(St, sy, sk, L=30, kchunk=16):
    """THE ROBUST LINE INTEGRAL ALONG THE FIBRES IN 3-D: the median along each pixel's streamline (L px both ways),
    which moves through the rows by sy [Y, X] (the fibres' slope) and through the layers by sk [Y, X] (layers per px:
    the frame's tilt, depth_slope_field): a fibre crossing the frame's layers is cleaned along itself, not across
    depths. St [K, Y, X] torch. On a GPU one kernel per voxel (kernels.streamline_median), else _flow_median3d_torch."""
    from . import kernels as KN
    if St.is_cuda and L <= 30 and KN.available():
        return KN.run("streamline_median", lambda: KN.streamline_median(St.float().contiguous(), sy.float().contiguous(),
                                                                         sk.float().contiguous(), L),
                      lambda: _flow_median3d_torch(St, sy, sk, L, kchunk), 1e-3, mean=True)
    return _flow_median3d_torch(St, sy, sk, L, kchunk)


def _flow_median3d_torch(St, sy, sk, L=30, kchunk=16):
    torch = _t(); F = torch.nn.functional
    K, Y, X = St.shape; dev = St.device
    vol = St[None, None]                                                  # [1, 1, K, Y, X]
    fld = torch.stack([sy, sk])[None]                                     # [1, 2, Y, X]
    gy0, gx0 = torch.meshgrid(torch.arange(Y, device=dev, dtype=torch.float32), torch.arange(X, device=dev, dtype=torch.float32), indexing="ij")
    out = torch.empty_like(St)

    def samp3(kk, yy, xx):                                                # kk, yy, xx [kc, Y, X]
        g = torch.stack([2 * xx / (X - 1) - 1, 2 * yy / (Y - 1) - 1, 2 * kk / max(K - 1, 1) - 1], -1)[None]
        return F.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    for k0 in range(0, K, kchunk):
        ks = torch.arange(k0, min(k0 + kchunk, K), device=dev, dtype=torch.float32)
        kc = len(ks)
        acc = [St[k0:k0 + kc]]
        for sgn in (1.0, -1.0):
            yy = gy0.expand(kc, Y, X).clone(); xx = gx0.expand(kc, Y, X).clone(); kk = ks[:, None, None].expand(kc, Y, X).clone()
            for _ in range(L):
                g = torch.stack([2 * xx / (X - 1) - 1, 2 * yy / (Y - 1) - 1], -1)                  # [kc, Y, X, 2]
                fs = F.grid_sample(fld.expand(kc, -1, -1, -1), g, mode="bilinear", padding_mode="border", align_corners=True)   # [kc, 2, Y, X]
                yy = yy + sgn * fs[:, 0]; kk = (kk + sgn * fs[:, 1]).clamp(0, K - 1); xx = xx + sgn
                acc.append(samp3(kk, yy, xx))
        out[k0:k0 + kc] = torch.stack(acc, 0).median(0).values
    return out


def joint_boundaries(Ism, offs, tracks, S, touch=2.0, minrun=6, lam=0.5, kwin=1.0):
    """TOUCHING NEIGHBOURS SHARE ONE BOUNDARY: where two fibres lie next to each other (gap < touch px) over >= minrun
    consecutive column blocks, the boundary between them is ONE smooth optimal line (dp_path) through the darkest
    rows between their centres, on the stack at their mean depth (+-kwin vox), block by block; both edges are set to
    it. Ism [K, Y, X] numpy. Returns tracks, number of runs joined."""
    K, Y, X = Ism.shape
    by_x = {}
    for ti, tr in enumerate(tracks):
        for n in tr:
            by_x.setdefault(n["x"], []).append((ti, n))
    pairs = {}
    for x, lst in by_x.items():
        lst.sort(key=lambda q: q[1]["yc"])
        for (ia, a), (ib, b_) in zip(lst[:-1], lst[1:]):
            if b_["top"] - a["bot"] < touch:
                pairs.setdefault((ia, ib), []).append((x, a, b_))
    nrun = 0
    for (ia, ib), recs in pairs.items():
        recs.sort(key=lambda q: q[0])
        xs = np.array([r[0] for r in recs])
        for seg in np.split(np.arange(len(recs)), np.nonzero(np.diff(xs) > 2)[0] + 1):
            if len(seg) < minrun:
                continue
            rr = [recs[i] for i in seg]
            cols = []
            for x, a, b_ in rr:
                dm = 0.5 * (a["d"] + b_["d"]); ks = np.abs(offs - dm) <= kwin
                if not ks.any():
                    ks = np.abs(offs - dm) == np.abs(offs - dm).min()
                cols.append(Ism[ks, :, x * S:min((x + 1) * S, X)].mean((0, 2)))
            C = np.stack(cols, 1)                                         # [Y, n]
            lo = np.array([a["yc"] + 1.0 for _, a, _ in rr]); hi = np.array([b_["yc"] - 1.0 for _, _, b_ in rr])
            ok = hi > lo
            if not ok.all():
                lo = np.where(ok, lo, 0.5 * (lo + hi) - 0.25); hi = np.where(ok, hi, 0.5 * (lo + hi) + 0.25)
            yb = dp_path(C, 0, lo, hi, sub=4, maxstep=2, lam=lam)
            for (x, a, b_), y in zip(rr, yb):
                a["bot"] = float(y); b_["top"] = float(y)
                for n in (a, b_):
                    n["w"] = n["bot"] - n["top"]; n["yc"] = 0.5 * (n["top"] + n["bot"])
            nrun += 1
    return tracks, nrun


def resolve_overlaps(tracks):
    """NEIGHBOURS MAY TOUCH, NOT OVERLAP: per column block, the fibres present sorted by centre; where one's bottom edge
    passes the next one's top edge the shared boundary is put at the middle of the overlap (both edges moved there)."""
    by_x = {}
    for ti, tr in enumerate(tracks):
        for n in tr:
            by_x.setdefault(n["x"], []).append(n)
    moved = 0
    for x, ns in by_x.items():
        ns.sort(key=lambda n: n["yc"])
        for a, b_ in zip(ns[:-1], ns[1:]):
            if a["bot"] > b_["top"]:
                m = 0.5 * (a["bot"] + b_["top"])
                m = min(max(m, a["yc"] + 1.0), b_["yc"] - 1.0)
                a["bot"] = m; b_["top"] = m; moved += 1
                for n in (a, b_):
                    n["w"] = n["bot"] - n["top"]
    return tracks, moved


def tracks_table(tracks, b, Xstack, s, min_len_px=100, half=True):
    """per fibre: arrays over its sampled columns (stack px): x, top / bottom / centre row IN THE FRAME STACK (un-
    straightened), depth (vox), width, level; plus summary numbers. Keeps fibres >= min_len_px long."""
    out = []
    for tr in tracks:
        xs = np.array([n["x"] for n in tr], float) * s + (s / 2.0 if half else 0.0)
        if xs.max() - xs.min() < min_len_px:
            continue
        o = np.argsort(xs); xs = xs[o]; tr = [tr[i] for i in o]
        f = lambda k: np.array([n[k] for n in tr], float)
        top = unstraighten_rows(f("top"), xs, b, Xstack); bot = unstraighten_rows(f("bot"), xs, b, Xstack)
        g = lambda k: np.array([n.get(k, np.nan) for n in tr], float)
        out.append(dict(x=xs, top=top, bot=bot, yc=0.5 * (top + bot), d=f("d"), dlo=f("dlo"), dhi=f("dhi"), w=f("w"), lvl=f("lvl"),
                        prom=f("prom"), vis=g("vis"), bright=g("bright"), cren=g("cren"), cbest=g("cbest"), score=float(np.nanmedian(g("score"))) if "score" in tr[0] else None,
                        order=tr[0].get("order"), length=float(xs.max() - xs.min()), wmed=float(np.median(f("w"))),
                        dmed=float(np.median(f("d")))))
    return out
