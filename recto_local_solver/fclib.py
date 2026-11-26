"""the frame's library: the slab (the CT along a surface's normals) on a cell-centred grid, the separable Gaussian
(derivative) filters, strided correlations, the normals of a crop.

Conventions
- frame cells (r, c) of P37's working frame; a window win = (r0, r1, c0, c1)
- the slab V[k, y, x]: k = normal offsets (vox; + = toward the scroll axis); y, x = `up` px per cell, CENTRED on the
  cells (px 0..up-1 belong to cell 0), so per-cell pooling is an exact up x up average
- voxel sizes of the slab: k 1 vox (if ks step 1), y / x CELL / up vox
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels as KN
from . import volume as qc

CELL = 5.06943
DEV = qc.DEV


def normals_crop(P, v, margin=10):
    """qc.normals on the bounding box of v (+ margin >= the 4 sigma = 8 cells of its smoothing): the same normals for
    the cells of v, without filtering the whole 1250 x 1321 frame for a patch of a few thousand cells."""
    rr, cc = np.nonzero(v)
    N = np.zeros(P.shape, np.float64); ok = np.zeros(v.shape, bool)
    if len(rr) == 0:
        return N, ok
    a0, a1 = max(int(rr.min()) - margin, 0), min(int(rr.max()) + margin + 1, v.shape[0])
    b0, b1 = max(int(cc.min()) - margin, 0), min(int(cc.max()) + margin + 1, v.shape[1])
    n, o = qc.normals(P[a0:a1, b0:b1], v[a0:a1, b0:b1])
    N[a0:a1, b0:b1] = n; ok[a0:a1, b0:b1] = o
    return N, ok


def slab_dense(P, v, N, win, ks, up=3, max_vox=6e8):
    """slab THROUGH ONE DENSE CROP: the resident chunks of the slab's bounding box are laid out as one
    block and read with the GPU's own trilinear sampler (F.grid_sample, one fused kernel per batch of depths), at the
    valid px only. The per-band path builds every sample point in float64 and reads its eight corners through the chunk
    table one by one (~40 passes over the points); this one is ~4x
    faster and gives the same values to 0.006 of 255 (the sampler's own float32 coordinates).
    With the CUDA kernels (kernels.slab_sample) no crop is built: every sample is read from the chunk cache with the
    per-band path's own arithmetic (its values), and the chunks made resident first are exactly those its samples read
    (volume.GPUVolume.ensure_samples); without them every chunk within reach of the slab (all valid px at every 8th vox of
    depth, margin 5 vox: no sample is further from one of these points). None when the box is too large (> max_vox
    voxels: the caller falls back to the per-band path)."""
    vol = qc.volume()
    r0, r1, c0, c1 = win
    Pw, vw, Nw = P[r0:r1, c0:c1], v[r0:r1, c0:c1], N[r0:r1, c0:c1]
    H, Wd = vw.shape
    K = len(ks)
    if H < 2 or Wd < 2:
        return None
    out = torch.zeros((K, H * up, Wd * up), dtype=torch.float32, device=DEV)
    Pt = torch.as_tensor(np.where(vw[..., None], Pw, 0), device=DEV, dtype=torch.float64)
    Nt = torch.as_tensor(np.where(vw[..., None], Nw, 0), device=DEV, dtype=torch.float64)
    vt = torch.as_tensor(vw, device=DEV)
    kern = DEV.startswith("cuda") and KN.available()
    if kern:                                                         # the px geometry in one kernel
        ok, PZ, NZ = KN.run("slab_geom", lambda: KN.slab_geometry(Pt, Nt, vt, up), lambda: _slab_geometry(Pt, Nt, vt, up), 1e-12)
    else:
        ok, PZ, NZ = _slab_geometry(Pt, Nt, vt, up)
    idx = ok.reshape(-1).nonzero()[:, 0]
    if idx.numel() == 0:
        return out, ok
    pz, nz_ = PZ[idx], NZ[idx]                                       # zyx, the valid px only
    kt = torch.as_tensor(np.asarray(ks, np.float64), device=DEV)
    C = vol.C
    tsh = torch.as_tensor(vol.table.shape, device=DEV)

    def box():                                                       # the resident chunks' box of the slab (the crop)
        e0, e1 = pz + kt[0] * nz_, pz + kt[-1] * nz_                 # (the slab lies between its end planes)
        lo = torch.div(torch.minimum(e0, e1).amin(0).floor().long() - 1, C, rounding_mode="floor").clamp(min=0)
        hi = torch.minimum(torch.div(torch.maximum(e0, e1).amax(0).floor().long() + 2, C, rounding_mode="floor"), tsh - 1)
        return lo, hi
    if kern:                                                         # sampled straight from the chunk cache (no crop)
        vol.ensure_samples(pz, nz_, ks)

        def fast():
            o = torch.zeros((K, out[0].numel()), dtype=torch.float32, device=DEV)
            return KN.slab_sample(vol.vol, vol.table, vol.st.shape, pz.contiguous(), nz_.contiguous(), ks, o, idx).view(out.shape)

        def ref():
            lo, hi = box()
            if bool((hi < lo).any()):
                return out
            return _slab_crop(vol, lo, *(int(x) for x in (hi - lo + 1).tolist()), pz, nz_, kt, out, idx)
        return KN.run("slab_sample", fast, ref, 1e-4), ok
    lo, hi = box()
    if bool((hi < lo).any()):
        return out, ok
    ncz, ncy, ncx = (int(x) for x in (hi - lo + 1).tolist())
    if ncz * ncy * ncx * C ** 3 > max_vox:
        return None
    kl = torch.unique(torch.cat([kt[::max(1, int(round(8.0 / max(float(kt[1] - kt[0]) if K > 1 else 1.0, 1e-6))))], kt[-1:]]))
    vol.ensure((pz[None] + kl[:, None, None] * nz_[None]).reshape(-1, 3), margin=5)
    return _slab_crop(vol, lo, ncz, ncy, ncx, pz, nz_, kt, out, idx), ok


def _slab_geometry(Pt, Nt, vt, up):
    """the slab's px geometry (kernels.slab_geometry's PyTorch form): ok [H up, W up], pz, nz [H up W up, 3] zyx."""
    H, Wd = vt.shape
    Ho, Wo = H * up, Wd * up
    off = (up - 1) / (2.0 * up)                                       # (x * (1 / up): what CUDA PyTorch computes for x / up,
    rows = torch.arange(Ho, device=Pt.device, dtype=torch.float64) * (1.0 / up) - off   # on every device)
    i0 = rows.floor().long().clamp(0, H - 2); a = (rows - i0).clamp(0, 1)[:, None, None]
    cols = torch.arange(Wo, device=Pt.device, dtype=torch.float64) * (1.0 / up) - off
    j0 = cols.floor().long().clamp(0, Wd - 2); b = (cols - j0).clamp(0, 1)[None, :, None]
    I0, J0 = i0[:, None], j0[None, :]
    ok = vt[I0, J0] & vt[I0 + 1, J0] & vt[I0, J0 + 1] & vt[I0 + 1, J0 + 1]
    ci = torch.clamp(torch.arange(Ho, device=Pt.device) // up, 0, H - 1)[:, None]   # (a px whose own cell is invalid is
    cj = torch.clamp(torch.arange(Wo, device=Pt.device) // up, 0, Wd - 1)[None, :]  # invalid too: no bridging)
    ok = ok & vt[ci, cj]
    p = ((1 - a) * (1 - b) * Pt[I0, J0] + a * (1 - b) * Pt[I0 + 1, J0] + (1 - a) * b * Pt[I0, J0 + 1] + a * b * Pt[I0 + 1, J0 + 1]).reshape(-1, 3)
    n = ((1 - a) * (1 - b) * Nt[I0, J0] + a * (1 - b) * Nt[I0 + 1, J0] + (1 - a) * b * Nt[I0, J0 + 1] + a * b * Nt[I0 + 1, J0 + 1]).reshape(-1, 3)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
    return ok, p[:, [2, 1, 0]], n[:, [2, 1, 0]]


def _slab_crop(vol, lo, ncz, ncy, ncx, pz, nz_, kt, out, idx):
    """slab_dense's sampling through the dense crop of the resident chunks (F.grid_sample)."""
    C = vol.C; K = len(kt)
    lo_l = lo.tolist()
    sl = vol.table[lo_l[0]:lo_l[0] + ncz, lo_l[1]:lo_l[1] + ncy, lo_l[2]:lo_l[2] + ncx].reshape(-1)
    dense = vol.vol[sl.clamp(min=0).long()]
    dense = torch.where((sl >= 0)[:, None, None, None], dense, torch.zeros((), dtype=dense.dtype, device=DEV))
    crop = dense.reshape(ncz, ncy, ncx, C, C, C).permute(0, 3, 1, 4, 2, 5).reshape(ncz * C, ncy * C, ncx * C).float()
    del dense
    rp = (pz - (lo * C).double()).float(); rn = nz_.float()                       # crop coordinates (float32: 1e-4 vox)
    D_, Hh, Ww = crop.shape
    sc = torch.tensor([2.0 / (Ww - 1), 2.0 / (Hh - 1), 2.0 / (D_ - 1)], device=DEV)
    kf = kt.float()
    of = out.view(K, -1)
    kb = int(min(max(1, 6e7 // idx.numel()), 64))
    for a0 in range(0, K, kb):
        g = (rp[None] + kf[a0:a0 + kb, None, None] * rn[None]).flip(-1) * sc - 1.0     # [kb, n, 3] xyz in -1..1
        of[a0:a0 + kb, idx] = F.grid_sample(crop[None, None], g[None, :, :, None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, :, :, 0]
    return out


def slab(P, v, N, win, ks, up=3, rows_per_band=48):
    """V [K, H*up, W*up] float32 (CT at P + k N on a cell-centred px grid) and ok [H*up, W*up] (all 4 corners valid),
    torch on the device: through one dense crop (slab_dense) when it fits, else band by band."""
    r = slab_dense(P, v, N, win, ks, up)
    if r is not None:
        return r
    r0, r1, c0, c1 = win
    Pw, vw, Nw = P[r0:r1, c0:c1], v[r0:r1, c0:c1], N[r0:r1, c0:c1]
    H, Wd = vw.shape
    Ho, Wo = H * up, Wd * up
    off = (up - 1) / (2.0 * up)
    Pt = torch.as_tensor(np.where(vw[..., None], Pw, 0), device=DEV)
    Nt = torch.as_tensor(np.where(vw[..., None], Nw, 0), device=DEV)
    vt = torch.as_tensor(vw, device=DEV)
    kt = torch.as_tensor(np.asarray(ks, np.float64), device=DEV)
    cols = torch.arange(Wo, device=DEV, dtype=torch.float64) / up - off
    j0 = cols.floor().long().clamp(0, Wd - 2); b = (cols - j0).clamp(0, 1)[None, :, None]
    out = torch.zeros((len(ks), Ho, Wo), dtype=torch.float32, device=DEV); ok = torch.zeros((Ho, Wo), dtype=torch.bool, device=DEV)
    for y0 in range(0, Ho, rows_per_band):
        y1 = min(Ho, y0 + rows_per_band)
        rows = torch.arange(y0, y1, device=DEV, dtype=torch.float64) / up - off
        i0 = rows.floor().long().clamp(0, H - 2); a = (rows - i0).clamp(0, 1)[:, None, None]
        I0, J0 = i0[:, None], j0[None, :]
        okb = vt[I0, J0] & vt[I0 + 1, J0] & vt[I0, J0 + 1] & vt[I0 + 1, J0 + 1]
        # a px whose own cell is invalid is invalid too (no bridging)
        ci = torch.clamp((torch.arange(y0, y1, device=DEV) // up), 0, H - 1)[:, None]
        cj = torch.clamp((torch.arange(Wo, device=DEV) // up), 0, Wd - 1)[None, :]
        okb = okb & vt[ci, cj]
        if not bool(okb.any()):
            continue
        p = (1 - a) * (1 - b) * Pt[I0, J0] + a * (1 - b) * Pt[I0 + 1, J0] + (1 - a) * b * Pt[I0, J0 + 1] + a * b * Pt[I0 + 1, J0 + 1]
        n = (1 - a) * (1 - b) * Nt[I0, J0] + a * (1 - b) * Nt[I0 + 1, J0] + (1 - a) * b * Nt[I0, J0 + 1] + a * b * Nt[I0 + 1, J0 + 1]
        n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
        pp, nn = p[okb][:, [2, 1, 0]], n[okb][:, [2, 1, 0]]
        pts = pp[None] + kt[:, None, None] * nn[None]
        smp = qc.Sampler(pts.reshape(-1, 3))
        vals = smp(pts.reshape(-1, 3)).reshape(len(ks), -1).float()
        blk = torch.zeros((len(ks), y1 - y0, Wo), dtype=torch.float32, device=DEV); blk[:, okb] = vals
        out[:, y0:y1] = blk; ok[y0:y1] = okb
    return out, ok


# ------------------------------------------------------------------------------------------------- filtering (GPU)
_GK = {}


def gkern(sig, order=0, dev=DEV):
    """1-D Gaussian (derivative) kernel for cross-correlation (conv*d): sum_x V(p + x) k(x) = (G^(order) * V)(p).
    Cached per (sig, order, device): 43 000 calls per solve built the same few kernels (read-only for the callers)."""
    key = (float(sig), int(order), str(dev))
    if key not in _GK:
        _GK[key] = _gkern(sig, order, dev)
    return _GK[key]


def _gkern(sig, order=0, dev=DEV):
    r = max(1, int(np.ceil(3.5 * sig)) + order)
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sig) ** 2); g = g / g.sum()
    if order == 1:
        g = x / sig ** 2 * g                       # = -G'(x) evaluated at -x  -> d/dp
    elif order == 2:
        g = (x ** 2 / sig ** 4 - 1 / sig ** 2) * g
    return g


def filt(X, sigs, orders=None):
    """separable Gaussian (derivative) filtering of a torch tensor X [..., d1, .., dn] along its last len(sigs) axes
    (sig 0 = untouched), replicate padding."""
    if X.is_cuda and X.dtype == torch.float32 and KN.available():
        return KN.run("sepconv", lambda: _filt_kernel(X, sigs, orders), lambda: _filt_torch(X, sigs, orders), 1e-5)
    return _filt_torch(X, sigs, orders)


def _filt_kernel(X, sigs, orders):
    nd = len(sigs)
    orders = orders or (0,) * nd
    Y = X.contiguous()
    for ax, (s, o) in enumerate(zip(sigs, orders)):
        if s <= 0 and o == 0:
            continue
        Y = KN.sepconv(Y, gkern(max(s, 0.5), o, X.device), ax - nd)
    return Y


def _filt_torch(X, sigs, orders):
    nd = len(sigs)
    orders = orders or (0,) * nd
    shp = X.shape
    if nd == 3:
        # every pass filters ONE axis: run it as a 2-D convolution on a reshaped view (the same sums; cudnn's 2-D
        # path is 2.3-2.8x faster than its 3-D one here). A long kernel along the middle
        # axis (the 40 px high-pass of the descriptors' lines) stays on the 3-D path (no gain there).
        d1, d2, d3 = shp[-3:]
        Y = X.reshape(-1, d1, d2, d3)
        nb = Y.shape[0]
        for ax, (s, o) in enumerate(zip(sigs, orders)):
            if s <= 0 and o == 0:
                continue
            g = gkern(max(s, 0.5), o)
            r = (len(g) - 1) // 2
            if ax == 0:
                Y = F.conv2d(F.pad(Y.reshape(nb, 1, d1, d2 * d3), (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1)).reshape(nb, d1, d2, d3)
            elif ax == 2:
                Y = F.conv2d(F.pad(Y.reshape(nb * d1, 1, d2, d3), (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1)).reshape(nb, d1, d2, d3)
            elif len(g) <= 64:
                Y = F.conv2d(F.pad(Y.reshape(nb * d1, 1, d2, d3), (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1)).reshape(nb, d1, d2, d3)
            else:
                Y = F.conv3d(F.pad(Y.reshape(nb, 1, d1, d2, d3), (0, 0, r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1, 1)).reshape(nb, d1, d2, d3)
        return Y.reshape(shp)
    Y = X.reshape(-1, 1, *shp[-nd:])
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[nd]
    for ax, (s, o) in enumerate(zip(sigs, orders)):
        if s <= 0 and o == 0:
            continue
        g = gkern(max(s, 0.5), o)
        r = (len(g) - 1) // 2
        ks = [1] * nd; ks[ax] = len(g)
        pad = [0] * (2 * nd)
        pad[2 * (nd - 1 - ax)] = r; pad[2 * (nd - 1 - ax) + 1] = r
        Y = conv(F.pad(Y, pad, mode="replicate"), g.view(1, 1, *ks))
    return Y.reshape(shp)


def strided_corr(Z, kern, pos, axis):
    """the correlation of Z [K, Hp, Wp] with each row of kern [nk, L] (L odd, centred) ALONG `axis` (1 = rows, 2 =
    columns), evaluated ONLY at the positions `pos` on that axis (ascending, equally spaced), replicate padding:
    -> [nk, K, Hp, n] (axis 2) or [nk, K, n, Wp] (axis 1). = filt along that axis read at those positions, at 1 / spacing
    of its cost (a strided convolution): for filters whose output is only ever read on a sparse set of lines."""
    n = len(pos); p0 = int(pos[0]); st = int(pos[1] - pos[0]) if n > 1 else 1
    if Z.is_cuda and Z.dtype == torch.float32 and KN.available():
        return KN.run("sepconv_strided", lambda: torch.stack([KN.sepconv(Z, kern[i], axis, start=p0, stride=st, count=n)
                                                              for i in range(kern.shape[0])]),
                      lambda: _strided_corr_torch(Z, kern, n, p0, st, axis), 1e-5)
    return _strided_corr_torch(Z, kern, n, p0, st, axis)


def _strided_corr_torch(Z, kern, n, p0, st, axis):
    Lk = kern.shape[1]; r = (Lk - 1) // 2
    size = Z.shape[axis]
    pl = max(r - p0, 0); pr = max(p0 + (n - 1) * st + r - (size - 1), 0)
    X = Z[:, None]
    a = p0 - r + pl
    if axis == 2:
        if pl or pr:
            X = F.pad(X, (pl, pr, 0, 0), mode="replicate")
        out = F.conv2d(X[..., a:a + (n - 1) * st + Lk], kern.view(-1, 1, 1, Lk), stride=(1, st))
    else:
        if pl or pr:
            X = F.pad(X, (0, 0, pl, pr), mode="replicate")
        out = F.conv2d(X[:, :, a:a + (n - 1) * st + Lk], kern.view(-1, 1, Lk, 1), stride=(st, 1))
    return out.permute(1, 0, 2, 3)


def kern_compose(g1, g2):
    """the kernel of the correlation with g1 followed by the correlation with g2 (both centred, odd): their full
    convolution, length len(g1) + len(g2) - 1 (on the CPU in float64: a few hundred taps)."""
    k = np.convolve(g1.detach().double().cpu().numpy(), g2.detach().double().cpu().numpy())
    return torch.as_tensor(k, dtype=torch.float32, device=g1.device)

