"""Provisional / frame-source surfaces for training crops: offset fields g [B, h, w] along the TRUE normal (band voxels,
+ = toward the scroll axis), i.e. what a wrong (or right) trace looks like. Each crop is a mosaic of 1 (40 %), 2 (35 %)
or 3-4 (25 %) regions, hard or soft (1-6-cell) seams; each region draws one generator:

| generator | share | what it is |
|---|---|---|
| right | 0.25 | truth + N(0, 0.3) roughness; p 0.3 a smooth bias <= 0.7; p 0.2 sparse spikes <= 2 vox |
| synth | 0.25 | broadband: smooth multi-scale field, power-law field (random slope 1.2-3.5, uneven strength), lamina patches (2.5-6 vox: verso / fibre interface), steps / tears, creases, dense rough patches; energy down to 1-2-cell wavelengths |
| snap | 0.20 | a randomised tracer: a displaced seed snapped 1-4x to the nearest CT feature (random feature: intensity peak, either edge, DoG ridge, a random 1-D filter bank, an intensity remap) within a random radius (2-6 vox), with random lateral smoothing between rounds: a real wrong structure, coherent over distance, as tracers lock on |

Generator shares are adapted during training: kinds the model currently fails on most are drawn more often (loss-
prioritised sampling; the trainer passes `cfg["shares"]`).
| neighbour | 0.15 | real neighbouring windings (k +-1 / +-2 / +-3, p 0.7 / 0.2 / 0.1): a patch, a staircase across windings, an excursion that returns, or a switch placed where the windings touch; gradual (2-20 cells) or abrupt |
| lowpass | 0.10 | the truth smoothed in xyz (sigma 2-12 cells) + an offset: a smoothed / inpainted trace |
| extension | 0.05 | the membrane extension of the green geometry: what extrapolation from green produces |

Amplitudes (synth, snap seed, lowpass offset), s = the object's winding spacing: 35 % U(0.3, 3) vox, 40 % U(3, 0.6 s),
25 % U(0.6 s, 1.5 s) (cfg "amp_law": the cumulative shares, default (0.35, 0.75); the fine-tune uses (0.1, 0.4)).
Global options per crop: one-sided (p 0.3), fading in from green (p 0.5, length log-U(1, 40)
cells), a common roughness up to 0.6 vox at 1-4-cell wavelengths on every region (p 0.5: right regions are not
"cleaner").

IDENTIFIABILITY: every label must be determinable from the inputs. With no green in view the winding cannot be told
apart, so such crops get no neighbour-winding regions and amplitudes <= 0.45 s (the nearest sheet is then the right
one by construction).

The sampler supplies the truth-dependent parts: `hp(sigma)` (the truth's high-pass along N), `ext()` (the membrane
extension offset), `snap(seed)` (feature snapping on the CT)."""
from __future__ import annotations

import math

import torch

from . import fields as R

GENS = ("right", "synth", "snap", "neighbour", "lowpass", "extension")
SHARES = (0.25, 0.25, 0.20, 0.15, 0.10, 0.05)


def _choice(gen, shares) -> int:
    u = R.su(gen) * sum(shares)
    acc = 0.0
    for i, s in enumerate(shares):
        acc += s
        if u < acc:
            return i
    return len(shares) - 1


def amplitude(gen, s: float, ident: bool, law=(0.35, 0.75)) -> float:
    """law = cumulative shares of the small (<= 3 vox) and middle (<= 0.6 s) amplitude classes (the fine-tune
    shifts them toward large errors via cfg["amp_law"])."""
    u = R.su(gen)
    if u < law[0]:
        a = R.su(gen, 0.3, 3.0)
    elif u < law[1]:
        a = R.su(gen, 3.0, max(3.5, 0.6 * s))
    else:
        a = R.su(gen, 0.6 * s, 1.5 * s)
    return a if ident else min(a, 0.45 * s)


def rough(h, w, gen, dev, lo=1.0, hi=4.0) -> torch.Tensor:
    return R.smooth_noise(1, h, w, R.su(gen, lo, hi), gen, dev)[0]


# ------------------------------------------------------------------------------------------------ single generators
def g_right(h, w, gen, dev) -> torch.Tensor:
    g = 0.3 * rough(h, w, gen, dev, 1.0, 3.0)
    if R.sc(gen, 0.3):
        g = g + R.su(gen, 0.2, 0.7) * R.smooth_noise(1, h, w, R.su(gen, 12, 48), gen, dev)[0] / 2
    if R.sc(gen, 0.2):
        m = torch.rand((h, w), generator=gen, device=dev) < R.su(gen, 0.002, 0.02)
        g = g + m.float() * R.su(gen, 0.5, 2.0) * R.rsign((h, w), gen, dev)
    return g


def powerlaw(h, w, gen, dev, beta: float) -> torch.Tensor:
    """Gaussian random field with a 1/f^beta power spectrum (spectral synthesis), unit std."""
    ky = torch.fft.fftfreq(h, device=dev).view(-1, 1)
    kx = torch.fft.rfftfreq(w, device=dev).view(1, -1)
    k = torch.sqrt(ky * ky + kx * kx).clamp(min=1.0 / max(h, w))
    amp = k ** (-beta / 2)
    z = torch.complex(torch.randn((h, w // 2 + 1), generator=gen, device=dev),
                      torch.randn((h, w // 2 + 1), generator=gen, device=dev))
    f = torch.fft.irfft2(z * amp, s=(h, w))
    return (f - f.mean()) / f.std().clamp(min=1e-6)


def g_synth(h, w, gen, dev, A: float) -> torch.Tensor:
    k = _choice(gen, (0.25, 0.15, 0.2, 0.15, 0.1, 0.15))
    sg = 1.0 if R.sc(gen, 0.5) else -1.0
    if k == 5:                                                       # power-law field, random slope, uneven strength
        f = powerlaw(h, w, gen, dev, R.su(gen, 1.2, 3.5))
        env = torch.exp(R.su(gen, 0.0, 0.8) * R.smooth_noise(1, h, w, R.su(gen, 16, 64), gen, dev)[0])
        return A * f * env / env.mean() / 2
    if k == 0:                                                       # smooth multi-scale + tilt + offset, broadband
        f = R.multiscale_noise(1, h, w, gen, dev, scales=(R.su(gen, 24, 64), R.su(gen, 8, 24), R.su(gen, 2, 8)))[0]
        g = A * f / 2 + R.su(gen, 0, 0.05) * R.plane(1, h, w, gen, dev)[0] + R.su(gen, -0.5, 0.5) * A
    elif k == 1:                                                     # lamina patches (verso / fibre interface)
        m = R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.05, 0.5)]).float()
        g = sg * R.su(gen, 2.5, 6.0) * R.blur(m, R.su(gen, 0.5, 4.0))[0].clamp(0, 1)
    elif k == 2:                                                     # step / tear along a line or curve
        sdf = R.line_sdf(1, h, w, gen, dev)[0]
        wd = R.su(gen, 0.0, 5.0)
        s = (sdf > 0).float() if wd < 0.25 else (0.5 + sdf / wd).clamp(0, 1)
        g = sg * A * (s - (0.5 if R.sc(gen, 0.3) else 0.0))
    elif k == 3:                                                     # crease (V or U)
        x = (R.line_sdf(1, h, w, gen, dev)[0].abs() / R.su(gen, 2, 10)).clamp(max=1.0)
        g = sg * A * (torch.cos(0.5 * math.pi * x) ** 2 if R.sc(gen, 0.5) else 1 - x)
    else:                                                            # dense rough patch (jagged trace)
        reg = R.blur(R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.2, 1.0)]).float(), 1.0)[0].clamp(0, 1)
        g = min(A, 3.0) * rough(h, w, gen, dev, 1.5, 4.0) * reg
    if R.sc(gen, 0.5):                                               # broadband: fine-scale energy on top
        g = g + R.su(gen, 0.1, 0.4) * min(A, 4.0) * rough(h, w, gen, dev, 1.0, 3.0)
    return g


def g_neighbour(h, w, gen, dev, nbr: torch.Tensor | None, ks: list, s: float) -> torch.Tensor:
    """nbr [2K, h, w] offsets to windings ks (NaN unknown) or None."""
    kk = 1 + _choice(gen, (0.7, 0.2, 0.1))
    sgn = 1 if R.sc(gen, 0.5) else -1
    jit = R.su(gen, 0.85, 1.15)

    def winding(k):
        synth = torch.full((h, w), -k * s * jit, device=dev)
        if nbr is not None and k in ks:
            fr = nbr[ks.index(k)]
            ok = torch.isfinite(fr)
            fill = torch.nanmedian(torch.where(ok, fr, torch.full_like(fr, float("nan"))))
            return torch.where(ok, fr, torch.where(torch.isfinite(fill), fill, synth[0, 0]))
        return synth

    res = 0.75 * R.smooth_noise(1, h, w, 16.0, gen, dev)[0]
    mode = _choice(gen, (0.35, 0.25, 0.2, 0.2))                      # patch, staircase, return, contact switch
    if mode == 3 and nbr is not None and (sgn * 1) in ks:
        d = winding(sgn * 1).abs()                                    # windings touch where |nbr| is small
        q = torch.nanquantile(d.flatten()[::7].float(), R.su(gen, 0.2, 0.6))
        patch = (d <= q).float()
        levels = [(sgn * kk, patch)]
    else:
        f = R.plane(1, h, w, gen, dev)[0] if R.sc(gen, 0.5) else R.smooth_noise(1, h, w, R.su(gen, 12, 48), gen, dev)[0]
        srt = f.flatten()[::max(1, (h * w) // 4096)].sort().values
        cover = R.su(gen, 0.2, 0.8)
        nlev = 1 if mode in (0, 2) else min(3, 1 + R.si(gen, 1, 3))
        levels = []
        for j in range(nlev):
            q = srt[min(len(srt) - 1, int(round((1 - cover) * (len(srt) - 1))))]
            m = (f > q).float()
            if mode == 2:                                            # excursion that returns: a band, not a half
                q2 = srt[min(len(srt) - 1, int(round((1 - cover * R.su(gen, 0.3, 0.7)) * (len(srt) - 1))))]
                m = m * (f <= q2).float()
            levels.append((sgn * (kk + j), m))
            cover *= R.su(gen, 0.3, 0.7)
    acc = torch.zeros((h, w), device=dev)
    ramp = R.sc(gen, 0.5)
    rw = R.su(gen, 2.0, 20.0) / 2.0
    for k, m in levels:
        if abs(k) > 3:
            continue
        mm = R.blur(m[None], rw)[0].clamp(0, 1) if ramp else m
        acc = acc * (1 - mm) + mm * (winding(k) + res)
    return acc


def g_lowpass(hp_fn, gen, s: float, ident: bool, h, w, dev) -> torch.Tensor:
    g = -hp_fn(R.su(gen, 2.0, 12.0))
    if R.sc(gen, 0.5):
        g = g + R.su(gen, -0.6, 0.6) * s * (1.0 if ident else 0.75)
    return g


def g_snap(snap_fn, h, w, gen, dev, A: float) -> torch.Tensor:
    seed = A * R.multiscale_noise(1, h, w, gen, dev, scales=(R.su(gen, 16, 48), R.su(gen, 6, 16), 3.0))[0] / 2
    seed = seed + R.su(gen, -1, 1) * A
    return snap_fn(seed)


# ------------------------------------------------------------------------------------------------ mosaic
def _regions(n, h, w, gen, dev) -> torch.Tensor:
    """[n, h, w] weights summing to 1: argmax of n smooth fields (hard) or a softmax (soft seams)."""
    if n == 1:
        return torch.ones((1, h, w), device=dev)
    f = R.smooth_noise(n, h, w, torch.tensor([R.su(gen, 20, 60)] * n), gen, dev)
    if R.sc(gen, 0.5):
        return torch.nn.functional.one_hot(f.argmax(0), n).permute(2, 0, 1).float()
    t = R.su(gen, 0.05, 0.4)
    return torch.softmax(f / t, 0)


def provisional(B, h, w, gen, dev, spacing: list, G: torch.Tensor, nbr, ks: list, hp_fns: list, ext_fns: list,
                snap_fns: list, cfg: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """-> g [B, h, w] (vox along N), kind [B, h, w] int8 (index into GENS of the dominant region)."""
    cfg = cfg or {}
    shares = tuple(cfg.get("shares", SHARES))
    law = tuple(cfg.get("amp_law", (0.35, 0.75)))
    p_one = float(cfg.get("p_one_sided", 0.3))
    out = torch.zeros((B, h, w), device=dev)
    kind = torch.zeros((B, h, w), dtype=torch.int8, device=dev)
    ident_l = G.flatten(1).any(1).tolist()                            # one host read per batch
    for b in range(B):
        s = float(spacing[b])
        ident = bool(ident_l[b])
        n = 1 + _choice(gen, (0.4, 0.35, 0.15, 0.10))
        Wr = _regions(n, h, w, gen, dev)
        acc = torch.zeros((h, w), device=dev)
        best = torch.full((h, w), -1.0, device=dev)
        one_sided = R.sc(gen, p_one)
        osign = 1.0 if R.sc(gen, 0.5) else -1.0
        for r in range(n):
            sh = list(shares)
            if not ident:
                sh[GENS.index("neighbour")] = 0.0
            k = _choice(gen, sh)
            A = amplitude(gen, s, ident, law)
            name = GENS[k]
            if name == "right":
                g = g_right(h, w, gen, dev)
            elif name == "synth":
                g = g_synth(h, w, gen, dev, A)
            elif name == "snap":
                g = g_snap(snap_fns[b], h, w, gen, dev, A)
            elif name == "neighbour":
                g = g_neighbour(h, w, gen, dev, None if nbr is None else nbr[b], ks, s)
            elif name == "lowpass":
                g = g_lowpass(hp_fns[b], gen, s, ident, h, w, dev)
            else:
                g = ext_fns[b]()
            if one_sided and name in ("synth", "snap", "lowpass"):
                g = osign * g.abs()
            if not ident:
                g = g.clamp(-0.45 * s, 0.45 * s)
            acc = acc + Wr[r] * g
            kind[b] = torch.where(Wr[r] > best, torch.full_like(kind[b], k), kind[b])
            best = torch.maximum(best, Wr[r])
        if ident and R.sc(gen, 0.5):                                 # errors fade in from green
            d = R.distance_to(G[b:b + 1], max_iter=64)[0]
            L = math.exp(R.su(gen, 0.0, math.log(40.0)))
            acc = acc * (1 - torch.exp(-d / L))
        if R.sc(gen, 0.5):                                           # common roughness on every region
            acc = acc + R.su(gen, 0.0, 0.6) * rough(h, w, gen, dev)
        out[b] = acc
    return out, kind


def presence(B, h, w, gen, dev) -> torch.Tensor:
    """Where the provisional is given as a prompt: absent 20 %, everywhere 50 %, partial (a 30-80 % blob) 30 %."""
    out = torch.zeros((B, h, w), dtype=torch.bool, device=dev)
    for b in range(B):
        u = R.su(gen)
        if u < 0.2:
            continue
        if u < 0.7:
            out[b] = True
        else:
            out[b] = R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.3, 0.8)], scale=[R.su(gen, 20, 60)])[0]
    return out
