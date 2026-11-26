"""Surface helpers for inference: oriented normals, height differences along the normals, the window blend weight,
the smoothed sheet and the thin-plate bridge used as a start where no draft surface is given."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from ..common.geometry import Axis, grid_normals, orient_to_axis


def normals(P: torch.Tensor, V: torch.Tensor, axis: Axis | None) -> torch.Tensor:
    N, ok = grid_normals(P, V)
    return orient_to_axis(N, P, ok, axis)


def height_deltas(P: torch.Tensor, N: torch.Tensor, unit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Current-surface height differences q - p along the mean normal, in labels, on right / down edges."""
    nx = 0.5 * (N[:, 1:] + N[:, :-1])
    ny = 0.5 * (N[1:] + N[:-1])
    dx = ((P[:, 1:] - P[:, :-1]) * nx).sum(-1) / unit
    dy = ((P[1:] - P[:-1]) * ny).sum(-1) / unit
    return dx, dy


def gauss_weight(h, w, device, sigma_frac=0.25):
    y = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2
    x = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    g = torch.exp(-0.5 * ((y[:, None] / (sigma_frac * h)) ** 2 + (x[None] / (sigma_frac * w)) ** 2))
    return g.clamp(min=1e-3)


def smooth_surface(P: np.ndarray, V: np.ndarray, sigma: float) -> np.ndarray:
    """P [H, W, 3] smoothed over the grid by a Gaussian of `sigma` cells, normalised over the valid cells."""
    w = V.astype(np.float64)
    den = np.maximum(ndimage.gaussian_filter(w, sigma), 1e-6)
    out = P.astype(np.float64).copy()
    for ch in range(3):
        out[..., ch] = ndimage.gaussian_filter(np.where(V, P[..., ch], 0.0), sigma) / den
    return np.where(V[..., None], out, P)


def thin_plate_fill(P: np.ndarray, known: np.ndarray, solve: np.ndarray, pad: int = 12) -> np.ndarray:
    """A start for `solve` cells bridged from the `known` cells only (their own positions are never used): the solve
    cells minimise the squared second differences along rows and columns and the cross differences over the known +
    solve cells, the known cells fixed. Where the known cells lie on one side only the grid is extrapolated linearly
    and keeps its cell spacing (a membrane fill would bunch the cells). -> P with the solve cells replaced."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    rr, cc = np.nonzero(solve)
    a0, a1 = max(rr.min() - pad, 0), rr.max() + pad + 1
    b0, b1 = max(cc.min() - pad, 0), cc.max() + pad + 1
    S_ = (known | solve)[a0:a1, b0:b1]
    Z_ = solve[a0:a1, b0:b1]
    # solve for the residual from the affine fit of the known cells: an affine grid has zero second differences, and
    # the small regularisation then pulls weakly constrained cells toward that extrapolation, not toward the origin
    ki, kj = np.nonzero(S_ & ~Z_)
    I, J = np.mgrid[0:S_.shape[0], 0:S_.shape[1]]
    Pk = P[a0:a1, b0:b1][ki, kj].astype(np.float64)
    coef = np.linalg.lstsq(np.c_[np.ones(len(ki)), ki, kj], Pk, rcond=None)[0]           # [3 (1, i, j), 3 (xyz)]
    base = coef[0] + I[..., None] * coef[1] + J[..., None] * coef[2]
    Pc = P[a0:a1, b0:b1] - base
    idx = -np.ones(S_.shape, int)
    zi, zj = np.nonzero(Z_)
    idx[zi, zj] = np.arange(len(zi))
    rows, cols, vals, rhs = [], [], [], []

    def add(cells, w):
        if not all(0 <= i < S_.shape[0] and 0 <= j < S_.shape[1] and S_[i, j] for i, j in cells):
            return
        if not any(Z_[i, j] for i, j in cells):
            return
        r = len(rhs)
        b = np.zeros(3)
        for (i, j), c in zip(cells, w):
            if Z_[i, j]:
                rows.append(r)
                cols.append(idx[i, j])
                vals.append(c)
            else:
                b -= c * Pc[i, j]
        rhs.append(b)

    for i in range(S_.shape[0]):
        for j in range(S_.shape[1]):
            if not S_[i, j]:
                continue
            add([(i - 1, j), (i, j), (i + 1, j)], [1.0, -2.0, 1.0])
            add([(i, j - 1), (i, j), (i, j + 1)], [1.0, -2.0, 1.0])
            add([(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)], [1.0, -1.0, -1.0, 1.0])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), len(zi)))
    Rh = np.array(rhs)
    AtA = (A.T @ A).tocsc() + sp.identity(len(zi), format="csc") * 1e-9
    X = np.stack([spl.spsolve(AtA, A.T @ Rh[:, k]) for k in range(3)], 1) + base[zi, zj]
    F = P.astype(np.float64).copy()
    F[a0:a1, b0:b1][zi, zj] = X
    return F
