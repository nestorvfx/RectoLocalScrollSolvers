"""Verified (green) layouts for training crops, drawn on the CONTEXT grid (h x w cells; the output core is the central
`core` x `core` cells). Every layout that verified areas take on real sheets, not only large nearby ones:

| class | share | form |
|---|---|---|
| none in view | 15 % | nothing verified in the crop |
| sparse | 25 % | 1-4 components: seed disks (r 3-15), strips (3-10 wide), partial tiles, 5-40 single-cell hints |
| tiles | 30 % | 50-cell tiles at a random phase, each green with p ~ U(0.1, 0.7); >= 1 green and >= 1 open tile |
| large | 20 % | half-plane 8, corner 4, strip 4, ring around a hole 4 |
| mostly | 10 % | 80-98 % green with holes (fill-in) |
| adjacent | 0 % (fine-tune: 50 %) | 1-2 areas (30-80 cells along the side, 15-40 deep) touching the output core from outside; the core is the area to recover |

Of the crops with green in view, 35 % have no green inside the core (+ a 0-10-cell margin): far green. Corruption as
nnU-Net's cascade input (`corrupt`: morphology p 0.4, part removal p 0.2).

`rejected_masks`: weak negatives, as rejected areas on a sheet: a tile-shaped or blob region over cells whose provisional
surface is wrong (> 1 vox) on most, not all, of its cells."""
from __future__ import annotations

import math

import torch

from . import fields as R

CLASSES = ("none", "sparse", "tiles", "large", "mostly", "adjacent")
SHARES = (0.15, 0.25, 0.30, 0.20, 0.10)                             # "adjacent" only when cfg["shares"] has 6 entries


def _choice(gen, shares) -> int:
    u = R.su(gen)
    acc = 0.0
    for i, s in enumerate(shares):
        acc += s
        if u < acc:
            return i
    return len(shares) - 1


def _sparse(h, w, gen, dev) -> torch.Tensor:
    yy, xx = R.grid(h, w, dev)
    m = torch.zeros((h, w), dtype=torch.bool, device=dev)
    for _ in range(R.si(gen, 1, 5)):
        k = R.si(gen, 0, 4)
        cy, cx = R.su(gen, 0, h), R.su(gen, 0, w)
        if k == 0:                                                   # seed disk
            r = R.su(gen, 3, 15)
            m |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        elif k == 1:                                                 # strip segment
            th = R.su(gen, 0, math.pi)
            wd, ln = R.su(gen, 3, 10), R.su(gen, 20, 120)
            u = math.cos(th) * (xx - cx) + math.sin(th) * (yy - cy)
            v = -math.sin(th) * (xx - cx) + math.cos(th) * (yy - cy)
            m |= (v.abs() < wd / 2) & (u.abs() < ln / 2)
        elif k == 2:                                                 # partial tile
            a, b = R.su(gen, 15, 50), R.su(gen, 15, 50)
            m |= ((yy - cy).abs() < a / 2) & ((xx - cx).abs() < b / 2)
        else:                                                        # single-cell hints
            n = R.si(gen, 5, 41)
            idx = torch.randint(0, h * w, (n,), generator=gen, device=dev)
            m.view(-1)[idx] = True
    return m


def _tiles(h, w, gen, dev, tile=50) -> torch.Tensor:
    yy, xx = R.grid(h, w, dev)
    oy, ox = R.si(gen, 0, tile), R.si(gen, 0, tile)
    ty = torch.div(yy.long() + oy, tile, rounding_mode="floor")
    tx = torch.div(xx.long() + ox, tile, rounding_mode="floor")
    nt = ((h - 1 + oy) // tile + 1, (w - 1 + ox) // tile + 1)
    p = R.su(gen, 0.1, 0.7)
    pk = torch.rand(nt, generator=gen, device=dev) < p
    n = nt[0] * nt[1]
    j = torch.full((), R.si(gen, 0, n), device=dev, dtype=torch.long)
    flat = torch.arange(n, device=dev).view(nt)
    if n > 1:                                                        # at least one green and one open tile
        pk = torch.where(pk.all() & (flat == j), torch.zeros_like(pk), pk)
    pk = torch.where(~pk.any() & (flat == j), torch.ones_like(pk), pk)
    return pk[ty, tx]


def _large(h, w, gen, dev) -> torch.Tensor:
    yy, xx = R.grid(h, w, dev)
    k = _choice(gen, (0.4, 0.2, 0.2, 0.2))
    if k == 0:                                                       # half-plane
        th = R.su(gen, 0, 2 * math.pi)
        d = math.cos(th) * (xx - w / 2) + math.sin(th) * (yy - h / 2)
        return d < R.su(gen, -0.3, 0.3) * min(h, w)
    if k == 1:                                                       # corner
        th = R.su(gen, 0, 2 * math.pi)
        a = math.cos(th) * (xx - w / 2) + math.sin(th) * (yy - h / 2)
        b = -math.sin(th) * (xx - w / 2) + math.cos(th) * (yy - h / 2)
        return (a < R.su(gen, -0.2, 0.3) * w) & (b < R.su(gen, -0.2, 0.3) * h)
    if k == 2:                                                       # strip
        th = R.su(gen, 0, math.pi)
        d = (math.cos(th) * (xx - w / 2) + math.sin(th) * (yy - h / 2) - R.su(gen, -0.3, 0.3) * w).abs()
        return d < R.su(gen, 5, 40)
    hole = R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.2, 0.6)], scale=[R.su(gen, 24, 64)])[0]   # ring around a hole
    return ~hole


def _mostly(h, w, gen, dev) -> torch.Tensor:
    holes = R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.02, 0.2)], scale=[R.su(gen, 6, 30)])[0]
    return ~holes


def _adjacent(h, w, core, gen, dev) -> torch.Tensor:
    """1 or 2 green areas touching the output core from outside (different sides)."""
    c0 = (h - core) // 2
    m = torch.zeros((h, w), dtype=torch.bool, device=dev)
    s0 = R.si(gen, 0, 4)
    sides = [s0] if R.sc(gen, 0.5) else [s0, (s0 + R.si(gen, 1, 4)) % 4]
    for s in sides:
        L = int(min(core, R.su(gen, 30, 80)))
        dp = max(1, min(c0, int(R.su(gen, min(15, c0), c0 + 1))))
        o = c0 + R.si(gen, 0, core - L + 1)
        if s == 0:
            m[c0 - dp:c0, o:o + L] = True
        elif s == 1:
            m[c0 + core:c0 + core + dp, o:o + L] = True
        elif s == 2:
            m[o:o + L, c0 - dp:c0] = True
        else:
            m[o:o + L, c0 + core:c0 + core + dp] = True
    return m


def green_masks(B: int, h: int, w: int, core: int, gen, dev, cfg: dict | None = None) -> tuple[torch.Tensor, list]:
    """-> G [B, h, w] bool (before & valid), class index per sample, far flag per sample."""
    cfg = cfg or {}
    shares = tuple(cfg.get("shares", SHARES))
    far_p = float(cfg.get("far_p", 0.35))
    out = torch.zeros((B, h, w), dtype=torch.bool, device=dev)
    cls, far = [], []
    c0 = (h - core) // 2
    for b in range(B):
        k = _choice(gen, shares)
        cls.append(k)
        name = CLASSES[k]
        if name == "none":
            far.append(False)
            continue
        if name == "adjacent":
            m = _adjacent(h, w, core, gen, dev)
        else:
            m = {"sparse": _sparse, "tiles": _tiles, "large": _large, "mostly": _mostly}[name](h, w, gen, dev)
        m = corrupt(m[None], gen, dev, cfg)[0]
        f = R.sc(gen, far_p) if name != "adjacent" else False
        if f:
            mg = R.si(gen, 0, 11)
            m[max(0, c0 - mg):c0 + core + mg, max(0, c0 - mg):c0 + core + mg] = False
        far.append(f)
        out[b] = m
    return out, cls, far


def rejected_masks(wrong: torch.Tensor, cand: torch.Tensor, gen, dev, p: float = 0.25) -> torch.Tensor:
    """wrong [B, h, w]: the provisional is > 1 vox off; cand: cells that may be marked (unknown, provisional present).
    -> rejected regions (tile-shaped or blob) where >= 60 % of the region is wrong."""
    B, h, w = wrong.shape
    out = torch.zeros_like(wrong)
    yy, xx = R.grid(h, w, dev)
    for b in range(B):
        if not R.sc(gen, p):
            continue
        for _ in range(4):                                           # a few tries to find a mostly-wrong region
            if R.sc(gen, 0.5):
                a, c = R.su(gen, 20, 60), R.su(gen, 20, 60)
                cy, cx = R.su(gen, 0, h), R.su(gen, 0, w)
                reg = ((yy - cy).abs() < a / 2) & ((xx - cx).abs() < c / 2)
            else:
                reg = R.blobs(1, h, w, gen, dev, frac=[R.su(gen, 0.05, 0.25)], scale=[R.su(gen, 10, 30)])[0]
            reg = reg & cand[b]
            n = reg.sum()
            ok = (n > 50) & ((wrong[b] & reg).sum() >= 0.6 * n)
            out[b] = torch.where(ok, reg, out[b])
    return out


def corrupt(m: torch.Tensor, gen, device, cfg: dict) -> torch.Tensor:
    B = m.shape[0]
    p_morph = cfg.get("p_morph", 0.4)
    p_remove = cfg.get("p_remove", 0.2)
    out = m.clone()
    for b in range(B):
        if R.sc(gen, p_morph):
            k = R.si(gen, 1, 9)
            op = R.si(gen, 0, 4)
            x = out[b:b + 1]
            x = (R.dilate(x, k), R.erode(x, k), R.dilate(R.erode(x, k), k), R.erode(R.dilate(x, k), k))[op]
            out[b] = x[0]
        if R.sc(gen, p_remove):
            h, w = out.shape[1:]
            cut = R.disk(1, h, w, gen, device, 0.1 * min(h, w), 0.25 * min(h, w))[0]
            # remove only a part that is < 15 % of the known mask (on the device: no host read)
            keep = (out[b] & cut).sum() >= 0.15 * out[b].sum()
            out[b] = torch.where(keep, out[b], out[b] & ~cut)
    return out
