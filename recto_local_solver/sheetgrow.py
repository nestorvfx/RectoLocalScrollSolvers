"""SHEETGROW: the geometric extrapolation and the grid relaxation of the frame's growth (sheetflat).
  predict  every new cell from its known grid neighbourhood: a weighted local FIRST-ORDER isometric fit (Gaussian weights
           over a radius of cells, the grid steps renormalised to one cell) -> the predicted 3-D point
  relax    after a solve, a tangential spring relaxation keeps the grid isometric (the depth stays the solve's)"""
from __future__ import annotations

import numpy as np
import torch

from . import fclib as L

DEV = L.DEV
CELLV = 5.06943                    # the grid step (vox): 47.46 um / 9.362 um


def predict(P, v, new, cfg):
    """positions of the `new` cells by a weighted local FIRST-ORDER fit of the known grid (v): P ~ a + Jr dr + Jc dc
    (Gaussian weights sig over radius rad), the grid steps Jr / Jc renormalised to one cell (ISOMETRIC: a traced
    sheet's grid is isometric, p1-p99 4.87-5.28 vox), the prediction = the weighted mean of every neighbour's own prediction
    P_k - dr_k Jr - dc_k Jc. A first-order isometric continuation is stable ring after ring (a quadratic one amplifies
    the noise: measured 4.5 -> 10^7 vox over 80 cells); the curvature over 4 cells (sagitta < 0.3 vox) is the solve's.
    Returns P_pred (only new cells changed) and the mask of cells that could be predicted."""
    rad, sig = cfg["rad"], cfg["sig"]
    H, W = v.shape
    out = P.copy()
    ok = np.zeros_like(new)
    rr, cc = np.nonzero(new)
    if len(rr) == 0:
        return out, ok
    f8 = dict(device=DEV, dtype=torch.float64)
    offs = [(di, dj) for di in range(-rad, rad + 1) for dj in range(-rad, rad + 1) if (di, dj) != (0, 0)]
    od = torch.tensor(offs, **f8); oi = od.long()
    w0 = torch.exp(-0.5 * (od ** 2).sum(1) / sig ** 2)
    A_full = torch.stack([torch.ones(len(od), **f8), od[:, 0], od[:, 1]], 1)
    Pt = torch.as_tensor(np.asarray(P, np.float64), device=DEV); vt = torch.as_tensor(v, device=DEV)
    rt = torch.as_tensor(rr, device=DEV); ct = torch.as_tensor(cc, device=DEV)
    for s0 in range(0, len(rr), 4096):                               # (on the GPU in float64: the same fit)
        r_, c_ = rt[s0:s0 + 4096], ct[s0:s0 + 4096]
        Ri = r_[:, None] + oi[None, :, 0]; Ci = c_[:, None] + oi[None, :, 1]
        inside = (Ri >= 0) & (Ri < H) & (Ci >= 0) & (Ci < W)
        R = Ri.clamp(0, H - 1); C = Ci.clamp(0, W - 1)
        m = vt[R, C] & inside                                        # [n, K]
        wts = m * w0[None]
        nk = m.sum(1)
        Y = Pt[R, C]                                                 # [n, K, 3]
        Aw = A_full[None] * wts[..., None]
        M = torch.einsum("nki,kj->nij", Aw, A_full) + 1e-6 * torch.eye(3, **f8)[None]
        rhs = torch.einsum("nki,nkd->nid", Aw, Y)
        coef = torch.linalg.solve(M, rhs)                            # [n, 3, 3]: a, Jr, Jc
        Jr, Jc = coef[:, 1], coef[:, 2]
        lr = torch.linalg.vector_norm(Jr, dim=1, keepdim=True); lc = torch.linalg.vector_norm(Jc, dim=1, keepdim=True)
        good = (nk >= cfg["min_nb"]) & (lr[:, 0] > 2.5) & (lr[:, 0] < 8) & (lc[:, 0] > 2.5) & (lc[:, 0] < 8)
        Jr = Jr / lr.clamp(min=1e-9) * CELLV; Jc = Jc / lc.clamp(min=1e-9) * CELLV
        pk = Y - od[None, :, 0:1] * Jr[:, None, :] - od[None, :, 1:2] * Jc[:, None, :]
        a = (pk * wts[..., None]).sum(1) / wts.sum(1, keepdim=True).clamp(min=1e-9)
        g = good.cpu().numpy(); ra, ca = rr[s0:s0 + 4096][g], cc[s0:s0 + 4096][g]
        out[ra, ca] = a[good].cpu().numpy()
        ok[ra, ca] = True
    return out, ok


def relax(P, v, move, N, iters=8):
    """the TANGENTIAL SPRING RELAXATION of the moved cells (float64, on their bounding box + 1): every cell is pulled
    toward the rest lengths of its 8 grid neighbours (one cell, diagonals sqrt 2), the pull's normal component
    removed (the depth stays the solve's); Jacobi iterations (every move from the previous iterate)."""
    X = P.copy()
    rr, cc = np.nonzero(move)
    if len(rr) == 0:
        return X
    H, W = v.shape
    a0, a1 = max(int(rr.min()) - 1, 0), min(int(rr.max()) + 2, H)
    b0, b1 = max(int(cc.min()) - 1, 0), min(int(cc.max()) + 2, W)
    Xt = torch.as_tensor(P[a0:a1, b0:b1], device=DEV, dtype=torch.float64).clone()
    vt = torch.as_tensor(v[a0:a1, b0:b1], device=DEV)
    Nt = torch.as_tensor(N[a0:a1, b0:b1], device=DEV, dtype=torch.float64)
    nr = Nt / torch.linalg.vector_norm(Nt, dim=-1, keepdim=True).clamp(min=1e-9)
    hh, ww = a1 - a0, b1 - b0
    r_ = torch.as_tensor(rr - a0, device=DEV); c_ = torch.as_tensor(cc - b0, device=DEV)
    nb = [(dy, dx, CELLV * (2 ** 0.5 if dy and dx else 1.0)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    n_ = nr[r_, c_]
    for _ in range(iters):
        Fm = torch.zeros((len(rr), 3), device=DEV, dtype=torch.float64); cnt = torch.zeros(len(rr), device=DEV, dtype=torch.float64)
        Xc = Xt[r_, c_]
        for dy, dx, rest in nb:
            r2, c2 = r_ + dy, c_ + dx
            ins = (r2 >= 0) & (r2 < hh) & (c2 >= 0) & (c2 < ww)
            r2c, c2c = r2.clamp(0, hh - 1), c2.clamp(0, ww - 1)
            okn = ins & vt[r2c, c2c]
            d = Xt[r2c, c2c] - Xc; ln = torch.linalg.vector_norm(d, dim=1)
            okn = okn & (ln > 1e-6)
            Fm += torch.where(okn[:, None], d * ((ln - rest) / ln.clamp(min=1e-6))[:, None], torch.zeros_like(d)); cnt += okn.double()
        Fm = Fm / cnt.clamp(min=1)[:, None]
        Fm = Fm - (Fm * n_).sum(1, keepdim=True) * n_
        Xt[r_, c_] = Xc + 0.5 * Fm
    X[a0:a1, b0:b1] = Xt.cpu().numpy()
    return X
