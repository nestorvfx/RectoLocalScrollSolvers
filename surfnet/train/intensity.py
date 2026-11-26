"""CT appearance augmentation of input stacks x [B, 1, D, H, W] (raw intensities 0..255) -> normalised stacks.

Order:
1. Physical artefacts in the raw intensity domain:
   - bright inclusions and dark voids (3-D Gaussian blobs);
   - cracks (thin dark discs at a random orientation);
   - ring artefacts (thin stripes along depth, i.e. ~parallel to the windings, drifting slowly laterally: they must
     not be taken for layers);
   - streaks;
   - a smooth multiplicative field (beam hardening / cupping, cf. the bias fields of SynthSeg / TorchIO);
   - masked-zero cuts (masked volumes are 0 outside the scroll mask).
2. Random-window normalisation to [-1, 1] from per-sample percentile bounds (lo 0.5-5 %, hi 95-99.5 %), applied BEFORE
   any clipping (random windowing, Wickstrom et al. 2025, arXiv 2510.08116).
3. Appearance set of nnU-Net (Isensee et al. 2021; nnUNetTrainer defaults):
   - multiplicative brightness 0.75-1.25 (p 0.15);
   - contrast 0.75-1.25 (p 0.15);
   - gamma 0.7-1.5 with retained statistics (p 0.3), inverted gamma (p 0.1);
   - Gaussian blur sigma 0.5-1 per axis (p 0.2);
   - simulated low resolution 0.5-1 (p 0.25);
   - Gaussian noise variance <= 0.1 (p 0.1).
   Plus:
   - GIN (randomly-weighted shallow conv nets mixed with the input) and IPA (two GIN outputs blended by a smooth
     map) (Ouyang et al. 2022, IEEE TMI; RandConv, Xu et al. 2021);
   - phase-contrast-like edge enhancement (unsharp masking);
   - correlated (blurred) noise.
Probabilities and ranges are configurable (`augment.intensity` in the train configs). Invalid pixels are 0 at the
end. `normalise_eval` is the deterministic validation / inference normalisation (window 1-99 %)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import fields as R

DEFAULTS = {
    "p_inclusion": 0.2, "p_crack": 0.15, "p_ring": 0.1, "p_streak": 0.1, "p_field": 0.2, "p_cut": 0.05,
    "window_lo": [0.005, 0.05], "window_hi": [0.95, 0.995],
    "p_brightness": 0.15, "brightness": [0.75, 1.25],
    "p_contrast": 0.15, "contrast": [0.75, 1.25],
    "p_gamma": 0.3, "p_gamma_inv": 0.1, "gamma": [0.7, 1.5],
    "p_gin": 0.25, "p_ipa": 0.5,
    "p_blur": 0.2, "blur": [0.5, 1.0],
    "p_sharpen": 0.15, "sharpen": [0.2, 1.0],
    "p_lowres": 0.25, "lowres": [0.5, 1.0],
    "p_noise": 0.1, "noise_var": [0.0, 0.1],
    "p_cnoise": 0.15, "cnoise_std": [0.05, 0.2],
}


def _cfg(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


def _sel(B, p, gen, dev):
    """Indices of the samples that get a transform (CPU decision: no host sync)."""
    return [b for b in range(B) if R.sc(gen, p)]


def _coords(D, H, W, vox, dev):
    """Voxel-space coordinates of the stack grid; vox = (depth vox per sample, lateral vox per pixel)."""
    d = torch.arange(D, device=dev, dtype=torch.float32)[:, None, None] * vox[0]
    y = torch.arange(H, device=dev, dtype=torch.float32)[None, :, None] * vox[1]
    x = torch.arange(W, device=dev, dtype=torch.float32)[None, None, :] * vox[1]
    return d, y, x


def _smooth3d(n, D, H, W, cells, gen, dev):
    g = torch.randn((n, 1, cells[0], cells[1], cells[2]), generator=gen, device=dev)
    return F.interpolate(g, size=(D, H, W), mode="trilinear", align_corners=True)


def _blur3d(x, sig):
    """Separable Gaussian blur of [n, 1, D, H, W]; sig = (sd, sy, sx) in pixels (0 = skip)."""
    for ax, s in zip((2, 3, 4), sig):
        if s <= 0:
            continue
        r = max(1, int(3 * s + 0.5))
        t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        g = torch.exp(-0.5 * (t / s) ** 2)
        g = g / g.sum()
        shape = [1, 1, 1, 1, 1]
        shape[ax] = -1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - ax)] = r
        pad[2 * (4 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), g.view(shape))
    return x


# ------------------------------------------------------------------------------------------------ raw-domain artefacts
def inclusions(x, gen, vox, idx):
    B, _, D, H, W = x.shape
    dev = x.device
    d, y, xx = _coords(D, H, W, vox, dev)
    for b in idx:
        for _ in range(R.si(gen, 1, 5)):
            c = [R.su(gen, 0, D * vox[0]), R.su(gen, 0, H * vox[1]), R.su(gen, 0, W * vox[1])]
            r = R.su(gen, 1.0, 3.0)
            a = R.su(gen, 60, 120) if R.sc(gen, 0.6) else -R.su(gen, 40, 80)
            d2 = (d - c[0]) ** 2 + (y - c[1]) ** 2 + (xx - c[2]) ** 2
            x[b, 0] += a * torch.exp(-d2 / (2 * r * r))
    return x


def cracks(x, gen, vox, idx):
    B, _, D, H, W = x.shape
    dev = x.device
    d, y, xx = _coords(D, H, W, vox, dev)
    for b in idx:
        c = [R.su(gen, 0, D * vox[0]), R.su(gen, 0, H * vox[1]), R.su(gen, 0, W * vox[1])]
        nrm = R.srng(gen).normal(size=3)
        nrm = nrm / max(float((nrm ** 2).sum()) ** 0.5, 1e-9)
        dist = nrm[0] * (d - c[0]) + nrm[1] * (y - c[1]) + nrm[2] * (xx - c[2])
        rad2 = (d - c[0]) ** 2 + (y - c[1]) ** 2 + (xx - c[2]) ** 2 - dist ** 2
        ext = R.su(gen, 10, 40)
        t = R.su(gen, 0.7, 1.5)
        depth = R.su(gen, 0.4, 0.8)
        m = torch.exp(-dist ** 2 / (2 * t * t)) * torch.sigmoid((ext * ext - rad2) / (ext * 2))
        x[b, 0] *= 1 - depth * m
    return x


def rings(x, gen, vox, idx, depth=None):
    """Thin stripes ~parallel to the windings: drawn on the TRUE band depth when `depth` [B, D, H, W] (vox) is given
    (rings follow the scan geometry, not the current surface), else on the stack's depth index."""
    B, _, D, H, W = x.shape
    dev = x.device
    d0 = torch.arange(D, device=dev, dtype=torch.float32)[:, None, None] * vox[0]
    for b in idx:
        d = depth[b] if depth is not None else d0
        lo, hi = (-D * vox[0] / 2, D * vox[0] / 2) if depth is not None else (0.0, D * vox[0])
        drift = R.su(gen, 0, 10) * R.smooth_noise(1, H, W, 64.0, gen, dev)[0] / 2
        for _ in range(R.si(gen, 1, 4)):
            pos = R.su(gen, lo, hi)
            wd = R.su(gen, 0.5, 1.0)
            a = R.su(gen, 10, 25) * (1 if R.sc(gen, 0.5) else -1)
            x[b, 0] += a * torch.exp(-((d - pos - drift[None]) ** 2) / (2 * wd * wd))
    return x


def streaks(x, gen, vox, idx):
    B, _, D, H, W = x.shape
    dev = x.device
    for b in idx:
        sdf = R.line_sdf(1, H, W, gen, dev, curved_p=0.0)[0]
        wd = R.su(gen, 0.5, 1.5)
        a = R.su(gen, 10, 20) * (1 if R.sc(gen, 0.5) else -1)
        x[b, 0] += a * torch.exp(-sdf ** 2 / (2 * wd * wd))[None]
    return x


def field(x, gen, idx, amp=0.2):
    if len(idx) == 0:
        return x
    B, _, D, H, W = x.shape
    a = R.urand((len(idx), 1, 1, 1, 1), 0, amp, gen, x.device)
    f = _smooth3d(len(idx), D, H, W, (3, 4, 4), gen, x.device)
    ii = torch.tensor(idx, device=x.device)
    x[ii] = x[ii] * torch.exp(a * f / 2)
    return x


def cuts(x, gen, idx):
    B, _, D, H, W = x.shape
    for b in idx:
        m = R.blobs(1, H, W, gen, x.device, frac=[R.su(gen, 0.05, 0.3)])[0]
        dthr = R.si(gen, 0, D)
        dm = torch.zeros(D, dtype=torch.bool, device=x.device)
        if R.sc(gen, 0.5):
            dm[dthr:] = True
        else:
            dm[:dthr] = True
        x[b, 0][dm[:, None, None] & m[None]] = 0.0
    return x


# ------------------------------------------------------------------------------------------------ normalisation
def window(x, valid, gen, cfg, train=True):
    """Per-sample percentile window to [-1, 1]. valid [B, 1, 1, H, W]."""
    B = x.shape[0]
    out = torch.empty_like(x)
    for b in range(B):
        flat = x[b].flatten()
        vm = valid[b].expand_as(x[b]).flatten()
        # a fixed-size random subsample (deterministic stride at evaluation), invalid voxels as NaN: no host read
        if train:
            pick = torch.randint(0, flat.numel(), (131072,), generator=gen, device=x.device)
        else:
            pick = torch.arange(0, flat.numel(), max(1, flat.numel() // 131072), device=x.device)
        v = torch.where(vm[pick], flat[pick], torch.full_like(flat[pick], float("nan")))
        if train:
            ql = R.su(gen, *cfg["window_lo"])
            qh = R.su(gen, *cfg["window_hi"])
        else:
            ql, qh = 0.01, 0.99
        vf = v.float()
        lo, hi = torch.nanquantile(vf, float(ql)), torch.nanquantile(vf, float(qh))
        lo, hi = torch.nan_to_num(lo, nan=0.0), torch.nan_to_num(hi, nan=1.0)
        out[b] = 2 * (x[b] - lo) / (hi - lo).clamp(min=1e-3) - 1
    return out


# ------------------------------------------------------------------------------------------------ appearance
def _retain(xn, x):
    m, s = x.mean((1, 2, 3, 4), keepdim=True), x.std((1, 2, 3, 4), keepdim=True)
    mn, sn = xn.mean((1, 2, 3, 4), keepdim=True), xn.std((1, 2, 3, 4), keepdim=True)
    return (xn - mn) / sn.clamp(min=1e-6) * s + m


def gamma(x, gen, idx, rng, invert=False):
    if len(idx) == 0:
        return x
    idx = torch.tensor(idx, device=x.device)
    y = x[idx]
    lo = y.amin((1, 2, 3, 4), keepdim=True)
    hi = y.amax((1, 2, 3, 4), keepdim=True)
    u = (y - lo) / (hi - lo).clamp(min=1e-6)
    g = R.urand((len(idx), 1, 1, 1, 1), *rng, gen, x.device)
    u = 1 - (1 - u).clamp(min=0) ** g if invert else u.clamp(min=0) ** g
    x[idx] = _retain(u * (hi - lo) + lo, y)
    return x


def gin(x, gen, idx, p_ipa=0.5, hidden=4, layers=4):
    """Global intensity non-linear augmentation: a randomly-weighted shallow 3-D conv net (kernel 1 or 3, leaky ReLU)
    per sample, output rescaled to the input's statistics and mixed with the input; IPA: two such outputs blended by
    a smooth random map."""
    def net(y):
        h = y
        cin = 1
        for i in range(layers):
            cout = 1 if i == layers - 1 else hidden
            k = 1 if R.sc(gen, 0.5) else 3
            wgt = torch.randn((cout, cin, k, k, k), generator=gen, device=y.device) / math.sqrt(cin * k ** 3)
            h = F.conv3d(h, wgt, padding=k // 2)
            if i < layers - 1:
                h = F.leaky_relu(h, 0.2)
            cin = cout
        h = _retain(h, y)
        # align the polarity with the input (a random net inverts contrast in ~1 of 9 draws); no host sync
        yc, hc = y - y.mean(), h - h.mean()
        dot = (yc * hc).sum()
        sign = torch.sign(dot) + (dot == 0).float()
        return sign * hc + h.mean()

    for b in idx:
        y = x[b:b + 1]
        a = R.su(gen, 0.0, 0.8)
        g1 = net(y)
        if R.sc(gen, p_ipa):
            g2 = net(y)
            D, H, W = y.shape[2:]
            m = torch.sigmoid(2 * _smooth3d(1, D, H, W, (2, 4, 4), gen, x.device))
            g1 = m * g1 + (1 - m) * g2
        x[b:b + 1] = a * g1 + (1 - a) * y
    return x


def augment(x: torch.Tensor, valid: torch.Tensor, gen, vox: tuple[float, float], cfg: dict | None = None,
            depth: torch.Tensor | None = None) -> torch.Tensor:
    """x [B, 1, D, H, W] raw (0..255) -> augmented, normalised. valid [B, 1, 1, H, W] bool. vox = (depth voxels per
    sample, lateral voxels per pixel). depth [B, D, H, W]: the true band depth of every sample (rings)."""
    c = _cfg(cfg)
    B = x.shape[0]
    dev = x.device
    x = x.float().clone()
    x = inclusions(x, gen, vox, _sel(B, c["p_inclusion"], gen, dev))
    x = cracks(x, gen, vox, _sel(B, c["p_crack"], gen, dev))
    x = rings(x, gen, vox, _sel(B, c["p_ring"], gen, dev), depth)
    x = streaks(x, gen, vox, _sel(B, c["p_streak"], gen, dev))
    x = field(x, gen, _sel(B, c["p_field"], gen, dev))
    x = cuts(x, gen, _sel(B, c["p_cut"], gen, dev))
    x = window(x, valid, gen, c, train=True)
    x = torch.where(valid, x, torch.zeros_like(x))         # invalid pixels must not leak into blur / GIN / gamma stats
    idx = _sel(B, c["p_brightness"], gen, dev)
    if len(idx):
        ii = torch.tensor(idx, device=dev)
        x[ii] = x[ii] * R.urand((len(idx), 1, 1, 1, 1), *c["brightness"], gen, dev)
    idx = _sel(B, c["p_contrast"], gen, dev)
    if len(idx):
        idx = torch.tensor(idx, device=dev)
        y = x[idx]
        m = y.mean((1, 2, 3, 4), keepdim=True)
        lo, hi = y.amin((1, 2, 3, 4), keepdim=True), y.amax((1, 2, 3, 4), keepdim=True)
        x[idx] = ((y - m) * R.urand((len(idx), 1, 1, 1, 1), *c["contrast"], gen, dev) + m).clamp(lo, hi)
    x = gamma(x, gen, _sel(B, c["p_gamma_inv"], gen, dev), c["gamma"], invert=True)
    x = gamma(x, gen, _sel(B, c["p_gamma"], gen, dev), c["gamma"], invert=False)
    x = gin(x, gen, _sel(B, c["p_gin"], gen, dev), c["p_ipa"])
    for b in _sel(B, c["p_blur"], gen, dev):
        sig = [R.su(gen, *c["blur"]) if R.sc(gen, 0.5) else 0.0 for _ in range(3)]
        x[b:b + 1] = _blur3d(x[b:b + 1], sig)
    for b in _sel(B, c["p_sharpen"], gen, dev):
        a = R.su(gen, *c["sharpen"])
        y = x[b:b + 1]
        x[b:b + 1] = y + a * (y - _blur3d(y, (1.0, 1.0, 1.0)))
    for b in _sel(B, c["p_lowres"], gen, dev):
        s = R.su(gen, *c["lowres"])
        y = x[b:b + 1]
        D, H, W = y.shape[2:]
        lo = F.interpolate(y, size=(max(2, int(D * s)), max(2, int(H * s)), max(2, int(W * s))), mode="area")
        x[b:b + 1] = F.interpolate(lo, size=(D, H, W), mode="trilinear", align_corners=False)
    idx = _sel(B, c["p_noise"], gen, dev)
    if len(idx):
        idx = torch.tensor(idx, device=dev)
        sd = torch.sqrt(R.urand((len(idx), 1, 1, 1, 1), *c["noise_var"], gen, dev))
        x[idx] = x[idx] + sd * torch.randn(x[idx].shape, generator=gen, device=dev)
    for b in _sel(B, c["p_cnoise"], gen, dev):
        sd = R.su(gen, *c["cnoise_std"])
        n = _blur3d(torch.randn(x[b:b + 1].shape, generator=gen, device=dev), (1.0, 1.0, 1.0))
        x[b:b + 1] = x[b:b + 1] + sd * n / n.std().clamp(min=1e-6)
    x = x.clamp(-3, 3)
    return torch.where(valid, x, torch.zeros_like(x))


def normalise_eval(x: torch.Tensor, valid: torch.Tensor, gen=None) -> torch.Tensor:
    x = window(x.float(), valid, gen, DEFAULTS, train=False).clamp(-3, 3)
    return torch.where(valid, x, torch.zeros_like(x))
