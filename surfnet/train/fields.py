"""Batched random-field primitives on the GPU (all [B, h, w] unless noted). Every function takes a torch.Generator so
training-sample synthesis is reproducible per seed."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


_SCALAR_RNG: dict = {}


def srng(gen) -> np.random.Generator:
    """A CPU generator paired with the torch generator `gen` (seeded from its seed). Scalar random decisions are drawn
    here: a scalar drawn on a CUDA generator and read with float() forces a host-device sync every time."""
    key = (id(gen), gen.initial_seed())
    r = _SCALAR_RNG.get(key)
    if r is None:
        if len(_SCALAR_RNG) > 256:
            _SCALAR_RNG.clear()
        r = _SCALAR_RNG[key] = np.random.default_rng(gen.initial_seed() & 0xFFFFFFFFFFFF)
    return r


def su(gen, lo=0.0, hi=1.0) -> float:
    return float(srng(gen).uniform(lo, hi))


def si(gen, lo, hi) -> int:
    """Integer in [lo, hi)."""
    return int(srng(gen).integers(lo, hi))


def sc(gen, p) -> bool:
    return bool(srng(gen).random() < p)


def urand(shape, lo, hi, gen, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def loguniform(shape, lo, hi, gen, device) -> torch.Tensor:
    return torch.exp(urand(shape, math.log(lo), math.log(hi), gen, device))


def coin(shape, p, gen, device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=device) < p


def rsign(shape, gen, device) -> torch.Tensor:
    return torch.where(coin(shape, 0.5, gen, device), 1.0, -1.0)


def grid(h, w, device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    return torch.meshgrid(y, x, indexing="ij")


def smooth_noise(B, h, w, scale, gen, device) -> torch.Tensor:
    """Unit-std smooth noise: white noise on a grid `scale` cells apart, bicubic-upsampled. `scale` scalar or [B]."""
    scales = scale.tolist() if torch.is_tensor(scale) else [float(scale)] * B
    out = torch.empty((B, h, w), device=device)
    for b in range(B):
        s = float(scales[b])
        gh, gw = max(2, int(math.ceil(h / s)) + 2), max(2, int(math.ceil(w / s)) + 2)
        g = torch.randn((1, 1, gh, gw), generator=gen, device=device)
        up = F.interpolate(g, size=(int(gh * s), int(gw * s)), mode="bicubic", align_corners=False)
        oy = si(gen, 0, max(1, up.shape[-2] - h))
        ox = si(gen, 0, max(1, up.shape[-1] - w))
        f = up[0, 0, oy:oy + h, ox:ox + w]
        out[b] = (f - f.mean()) / f.std().clamp(min=1e-6)
    return out


def multiscale_noise(B, h, w, gen, device, scales=(48.0, 16.0, 6.0), weights=None) -> torch.Tensor:
    """Sum of smooth noises at several scales with random weights, unit std."""
    out = torch.zeros((B, h, w), device=device)
    for i, s in enumerate(scales):
        wgt = urand((B, 1, 1), 0.0, 1.0, gen, device) if weights is None else weights[i]
        out = out + wgt * smooth_noise(B, h, w, s, gen, device)
    return out / out.flatten(1).std(1).clamp(min=1e-6)[:, None, None]


def plane(B, h, w, gen, device) -> torch.Tensor:
    """Random planar ramp with unit slope (cells) in a random direction, zero at the centre."""
    yy, xx = grid(h, w, device)
    th = urand((B, 1, 1), 0, 2 * math.pi, gen, device)
    return torch.cos(th) * (xx - w / 2) + torch.sin(th) * (yy - h / 2)


def line_sdf(B, h, w, gen, device, curved_p: float = 0.5) -> torch.Tensor:
    """Signed distance (cells) to a random line through the crop; with probability curved_p the line is bent by a
    smooth perpendicular offset (a curve)."""
    yy, xx = grid(h, w, device)
    th = urand((B, 1, 1), 0, math.pi, gen, device)
    cy = urand((B, 1, 1), 0.2 * h, 0.8 * h, gen, device)
    cx = urand((B, 1, 1), 0.2 * w, 0.8 * w, gen, device)
    nx, ny = torch.cos(th), torch.sin(th)
    d = nx * (xx - cx) + ny * (yy - cy)
    along = -ny * (xx - cx) + nx * (yy - cy)
    bend = coin((B, 1, 1), curved_p, gen, device).float()
    amp = urand((B, 1, 1), 2.0, 0.25 * min(h, w), gen, device)
    lam = urand((B, 1, 1), 0.5 * min(h, w), 2.0 * min(h, w), gen, device)
    ph = urand((B, 1, 1), 0, 2 * math.pi, gen, device)
    return d - bend * amp * torch.sin(2 * math.pi * along / lam + ph)


def blobs(B, h, w, gen, device, frac=None, scale=None) -> torch.Tensor:
    """Binary blob masks covering about `frac` of the crop (thresholded smooth noise at the matching quantile)."""
    frac = [su(gen, 0.05, 0.5) for _ in range(B)] if frac is None else (frac.tolist() if torch.is_tensor(frac) else list(frac))
    scale = [su(gen, 8.0, 32.0) for _ in range(B)] if scale is None else (scale.tolist() if torch.is_tensor(scale) else list(scale))
    n = smooth_noise(B, h, w, torch.tensor(scale), gen, device)
    sub = n.flatten(1)[:, ::max(1, (h * w) // 4096)]
    srt = sub.sort(dim=1).values                                   # per-row quantile at its own level, on the device
    idx = torch.stack([torch.full((), round((1 - f) * (srt.shape[1] - 1)), device=device, dtype=torch.long)
                       for f in frac])                              # built on the device: no blocking copy
    q = srt.gather(1, idx[:, None])[:, 0]
    return n > q.view(B, 1, 1)


def disk(B, h, w, gen, device, rmin, rmax) -> torch.Tensor:
    yy, xx = grid(h, w, device)
    cy = urand((B, 1, 1), 0, h, gen, device)
    cx = urand((B, 1, 1), 0, w, gen, device)
    r = urand((B, 1, 1), rmin, rmax, gen, device)
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r * r


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of [B, h, w] (reflect padding)."""
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()
    y = F.pad(x[:, None], (r, r, r, r), mode="replicate")
    y = F.conv2d(y, g.view(1, 1, 1, -1))
    y = F.conv2d(y, g.view(1, 1, -1, 1))
    return y[:, 0]


def dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return m
    return F.max_pool2d(m[:, None].float(), 2 * k + 1, stride=1, padding=k)[:, 0] > 0.5


def erode(m: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return m
    return ~dilate(~m, k)


def distance_to(mask: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """Chamfer-like distance (cells) of every cell to the nearest True cell of mask [B, h, w], by repeated 3 x 3
    dilation (exact in the chessboard metric, capped at max_iter). Cells of an empty mask get max_iter."""
    d = torch.where(mask, 0.0, float(max_iter)).to(torch.float32)
    cur = mask.clone()
    for i in range(1, max_iter + 1):                 # fixed iteration count: no host sync
        nxt = dilate(cur, 1)
        d = torch.where(nxt & ~cur, float(i), d)
        cur = nxt
    return d


def grad2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences of [B, h, w] along rows and columns (per cell)."""
    gy = torch.zeros_like(x)
    gx = torch.zeros_like(x)
    gy[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    gx[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) / 2
    gy[:, 0], gy[:, -1] = x[:, 1] - x[:, 0], x[:, -1] - x[:, -2]
    gx[:, :, 0], gx[:, :, -1] = x[:, :, 1] - x[:, :, 0], x[:, :, -1] - x[:, :, -2]
    return gy, gx
