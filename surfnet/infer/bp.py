"""Global surface solve: min-sum loopy belief propagation on the 4-connected cell grid (GPU, batched).

Labels l = offset bins. Data cost D_p(l) = -log p_p(l) (clipped). Pairwise cost on each edge p -> q (right / down):
    V(l_p, l_q) = s * min(|l_q - l_p + delta_pq|, tau)
where delta_pq (labels) is the CURRENT surface's height difference q - p along the normal: smoothness is imposed on
the TRUE surface (current + offset), so a step in the current mesh is undone by the offsets, not preserved.
Messages use the linear-time min-convolution for truncated-linear costs (Felzenszwalb & Huttenlocher 2006, "Efficient
belief propagation for early vision"): two running minima (torch.cummin) instead of an O(K^2) minimum.
Pins (verified cells) have cost 0 at their label and BIG elsewhere. Synchronous updates with damping; messages are
normalised to min 0. Result: per-cell label (argmin belief, parabolic sub-bin refinement) and min-marginal margins.

The label shifts by delta are fixed during the iterations: their gather indices, weights and out-of-range masks are
computed once (identical results, fewer kernels per iteration); `compiled=True` runs the iteration
through torch.compile on the GPU (automatic dynamic shapes: one fixed-shape graph per label count, then one
shape-general one). On the CPU it runs eagerly: compiling there took 25-40 s per process for the same labels, against
about 1 s for a whole solve."""
from __future__ import annotations

import torch
import torch.nn.functional as F

BIG = 1e4


def minconv(h: torch.Tensor, s: float, tau: float) -> torch.Tensor:
    """m(l) = min_j h(j) + s * min(|l - j|, tau), along dim 1 of h [B, K, ...]."""
    K = h.shape[1]
    j = torch.arange(K, device=h.device, dtype=h.dtype).view(1, K, *([1] * (h.dim() - 2)))
    f = torch.cummin(h - s * j, 1).values + s * j
    g = torch.cummin((h + s * j).flip(1), 1).values.flip(1) - s * j
    m = torch.minimum(f, g)
    return torch.minimum(m, h.amin(1, keepdim=True) + s * tau)


def shift_plan(d: torch.Tensor, K: int):
    """For m'(l) = m(l + d) (d broadcast over the label dim 1): gather indices, weights and the BIG mask."""
    l = torch.arange(K, device=d.device, dtype=d.dtype).view(1, K, *([1] * (d.dim() - 1))) + d.unsqueeze(1)
    l0 = l.floor()
    fr = l - l0
    i0 = l0.long()
    i1 = i0 + 1
    bad = ((i0 < 0) | (i1 > K - 1)) & ~((i0 == K - 1) & (fr < 1e-6))
    return i0.clamp(0, K - 1), i1.clamp(0, K - 1), fr, bad


def apply_shift(m: torch.Tensor, plan) -> torch.Tensor:
    i0, i1, fr, bad = plan
    v = (1 - fr) * torch.gather(m, 1, i0) + fr * torch.gather(m, 1, i1)
    return torch.where(bad, torch.full_like(v, BIG), v)


def shift(m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """m'(l) = m(l + d) along dim 1 (linear interpolation; outside the label range = BIG). d broadcasts over dim 1."""
    return apply_shift(m, shift_plan(d.to(m.dtype), m.shape[1]))


def _iteration(D, mL, mR, mU, mD, ex, ey, pL, pR, pU, pD, s: float, tau: float, damp: float):
    bel = D + mL + mR + mU + mD
    nL = apply_shift(minconv((bel - mR)[..., :-1], s, tau), pL)       # p -> right neighbour (arrives "from the left")
    nR = apply_shift(minconv((bel - mL)[..., 1:], s, tau), pR)        # q -> left neighbour
    nU = apply_shift(minconv((bel - mD)[:, :, :-1], s, tau), pU)
    nD = apply_shift(minconv((bel - mU)[:, :, 1:], s, tau), pD)
    nL = (nL - nL.amin(1, keepdim=True)) * ex
    nR = (nR - nR.amin(1, keepdim=True)) * ex
    nU = (nU - nU.amin(1, keepdim=True)) * ey
    nD = (nD - nD.amin(1, keepdim=True)) * ey
    mL = F.pad(damp * mL[..., 1:] + (1 - damp) * nL, (1, 0))
    mR = F.pad(damp * mR[..., :-1] + (1 - damp) * nR, (0, 1))
    mU = F.pad(damp * mU[:, :, 1:] + (1 - damp) * nU, (0, 0, 1, 0))
    mD = F.pad(damp * mD[:, :, :-1] + (1 - damp) * nD, (0, 0, 0, 1))
    return mL, mR, mU, mD


_iteration_compiled = None


def solve(D: torch.Tensor, valid: torch.Tensor, dx: torch.Tensor | None = None, dy: torch.Tensor | None = None,
          s: float = 0.3, tau: float = 6.0, iters: int = 60, damp: float = 0.5,
          pin_label: torch.Tensor | None = None, compiled: bool = False) -> dict:
    """D [B, K, h, w] data costs; valid [B, h, w]; dx [B, h, w-1], dy [B, h-1, w] current-surface height differences
    (q - p, labels) on right / down edges; pin_label [B, h, w] (float label, NaN = free)."""
    global _iteration_compiled
    B, K, h, w = D.shape
    D = torch.where(valid[:, None], D, torch.zeros_like(D))
    if pin_label is not None:
        pin = torch.isfinite(pin_label) & valid
        lab = torch.arange(K, device=D.device).view(1, K, 1, 1)
        pin_cost = torch.where(lab == pin_label.nan_to_num(0).round().long()[:, None], 0.0, BIG)
        D = torch.where(pin[:, None], pin_cost.to(D.dtype), D)
    if dx is None:
        dx = torch.zeros((B, h, w - 1), device=D.device, dtype=D.dtype)
    if dy is None:
        dy = torch.zeros((B, h - 1, w), device=D.device, dtype=D.dtype)
    dx, dy = dx.to(D.dtype), dy.to(D.dtype)
    ex = (valid[:, :, 1:] & valid[:, :, :-1])[:, None].to(D.dtype)    # right edges
    ey = (valid[:, 1:] & valid[:, :-1])[:, None].to(D.dtype)          # down edges
    pL, pR = shift_plan(dx, K), shift_plan(-dx, K)
    pU, pD = shift_plan(dy, K), shift_plan(-dy, K)
    mL = torch.zeros_like(D)
    mR = torch.zeros_like(D)
    mU = torch.zeros_like(D)
    mD = torch.zeros_like(D)
    step = _iteration
    if compiled and D.is_cuda:
        if _iteration_compiled is None:
            # automatic dynamic shapes (the default): the first area of each label count compiles a fixed-shape graph,
            # the first other area one shape-general graph, reused for every later area. With dynamic=False every new
            # area recompiled (~22 s each), and after 8 the iteration fell back to eager.
            import torch._dynamo.config as dyn
            for name in ("recompile_limit", "cache_size_limit"):
                if hasattr(dyn, name):
                    setattr(dyn, name, max(getattr(dyn, name), 64))
            _iteration_compiled = torch.compile(_iteration)
        step = _iteration_compiled
    for _ in range(iters):
        mL, mR, mU, mD = step(D, mL, mR, mU, mD, ex, ey, pL, pR, pU, pD, s, tau, damp)
    bel = D + mL + mR + mU + mD
    l = bel.argmin(1)
    b0 = torch.gather(bel, 1, l[:, None])[:, 0]
    bm = torch.gather(bel, 1, (l - 1).clamp(0, K - 1)[:, None])[:, 0]
    bp = torch.gather(bel, 1, (l + 1).clamp(0, K - 1)[:, None])[:, 0]
    den = (bm - 2 * b0 + bp)
    off = torch.where((den > 1e-6) & (l > 0) & (l < K - 1), 0.5 * (bm - bp) / den, torch.zeros_like(den))
    lab = l.float() + off.clamp(-0.5, 0.5)
    second = torch.where(torch.arange(K, device=D.device).view(1, K, 1, 1) == l[:, None], BIG, bel).amin(1)
    return {"label": lab, "margin": second - b0, "belief": bel}


def data_cost(logits: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return (-torch.log_softmax(logits.float(), 1)).clamp(max=clip)
