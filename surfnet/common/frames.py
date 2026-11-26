"""Crop geometry on the GPU.

Offsets are measured along the TRUE normal N from the true surface P (band voxels; + = toward the scroll axis), as the
bands are sampled along N. A surface Q given as xyz has the offset field (Q - P) . N.

- `blur_masked`: normalised Gaussian convolution over the valid cells (missing cells do not pull the average);
- `lowpass_xyz`: the low-pass of a surface in xyz (per-sample sigma, cells): the frame and the "smoothed trace";
- `harmonic`: the membrane (Laplace) extension of the surface from fixed cells (what extrapolation from green gives);
- `offset_of`: (Q - P) . N."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _kernel(sig: float, device) -> tuple[torch.Tensor, int]:
    r = max(1, int(math.ceil(3 * sig)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (t / sig) ** 2)
    return g / g.sum(), r


def _sep(x: torch.Tensor, g: torch.Tensor, r: int) -> torch.Tensor:
    """Separable blur of [n, C, h, w] with zero padding."""
    C = x.shape[1]
    kx = g.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0)), kx, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r)), ky, groups=C)


def blur_masked(f: torch.Tensor, m: torch.Tensor, sig: float) -> torch.Tensor:
    """f [C, h, w] or [n, C, h, w], m [h, w] / [n, h, w] -> normalised blur (cells without support keep f)."""
    one = f.dim() == 3
    if one:
        f, m = f[None], m[None]
    if sig <= 0:
        return f[0] if one else f
    g, r = _kernel(float(sig), f.device)
    mf = m.float()[:, None]
    num = _sep(f * mf, g, r)
    den = _sep(mf, g, r)
    out = torch.where(den > 1e-4, num / den.clamp(min=1e-4), f)
    return out[0] if one else out


def lowpass_xyz(P: torch.Tensor, V: torch.Tensor, sig) -> torch.Tensor:
    """P [B, 3, h, w], V [B, h, w], sig: float or [B] list (cells) -> [B, 3, h, w]."""
    B = P.shape[0]
    sl = [float(sig)] * B if not isinstance(sig, (list, tuple)) else [float(s) for s in sig]
    return torch.stack([blur_masked(P[b], V[b], sl[b]) for b in range(B)])


def offset_of(Q: torch.Tensor, P: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """[B, 3, h, w] x3 -> [B, h, w]."""
    return ((Q - P) * N).sum(1)


def harmonic(P: torch.Tensor, fixed: torch.Tensor, V: torch.Tensor, iters: int = 120, init_sigma: float = 12.0,
             check: bool = True) -> torch.Tensor:
    """Membrane extension of P [3, h, w] from the `fixed` cells over the valid cells: start from a wide normalised
    blur of the fixed cells, then Jacobi iterations of the 4-neighbour average (fixed cells kept)."""
    if check and not bool(fixed.any()):
        return P.clone()
    X = blur_masked(torch.where(fixed[None], P, torch.zeros_like(P)), fixed, init_sigma)
    X = torch.where(fixed[None], P, X)
    vm = V.float()[None, None]
    cp = F.pad(vm, (1, 1, 1, 1))                                     # neighbour counts: constant over the iterations
    c = (cp[..., :-2, 1:-1] + cp[..., 2:, 1:-1] + cp[..., 1:-1, :-2] + cp[..., 1:-1, 2:]).clamp(min=1)
    keep = fixed[None] | ~V[None]
    for _ in range(iters):
        xp = F.pad(X[None] * vm, (1, 1, 1, 1))
        s = xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1] + xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:]
        X = torch.where(keep, X, (s / c)[0])
    return X
