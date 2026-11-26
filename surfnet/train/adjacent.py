"""The adjacent-area recovery test: one or two locked (green) areas next to the
target area; the target starts very broken; the answer must be the true continuation of the locked sheet, whatever
the start. Part of every validation.

Layout: green = 1 or 2 tile-sized areas (50 cells along the side x the 40-cell ring) touching the output core
(80 x 80 cells) from outside, on different sides; the core is the target (0-80 cells from green). Nothing else is
known. Starts (K per crop; the same crop and green for every start): a natural field (`infer/perturb.py`, never
used in training), |A| from 4 vox up to 0.6-1.5 windings, one-sided with a random sign, diverging from green over
2-14 cells, inner roughness; with p 0.5 plus a tile-shaped patch on the +-1 winding (the real neighbour where
known). The frame is the low-pass of the start (no extension frames, no trace channel, no rejected prompt).
Chain per start, as deployed: A -> BP -> refine frame -> B. The final estimate per cell is the fine head, or the
context head where the fine stream is invalid (frame > fine_max off) or calls "outside" (as `infer.run_sheet`), so
cells the chain leaves far off are scored, not dropped.

Metrics on the target (core) cells of crops with green:
| key | meaning |
|---|---|
| right | share within half a winding of the truth (on the same sheet as green) |
| p50 / p90 | \\|error\\| vox |
| signed | median error, + = toward the scroll axis (a systematic offset adds up in band growth) |
| agree1 / agree2 | share of cells where all K starts end within 1 / 2 vox of each other (start-independence) |
| right_broken | right-winding share on cells whose start was > half a winding off (broken_share = how many) |
| by_dist | distance to green 0-10 / 10-25 / 25-50 / 50-80 cells: [right, p50] |"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import fields as Rf
from ..infer.perturb import FAMILIES as NATURAL, natural_field
from ..model import bins_ctx, bins_fine, readout

BINS = ((0, 10), (10, 25), (25, 50), (50, 81))


def adjacent_greens(B, h, w, core, gen, dev, cfg=None):
    """1 (even samples) or 2 (odd) green areas touching the core from outside -> (G, cls, far)."""
    c0 = (h - core) // 2
    ring, L = c0, min(50, core)
    out = torch.zeros((B, h, w), dtype=torch.bool, device=dev)
    for b in range(B):
        s0 = Rf.si(gen, 0, 4)
        sides = [s0] if b % 2 == 0 else [s0, (s0 + Rf.si(gen, 1, 4)) % 4]
        for s in sides:
            o = c0 + Rf.si(gen, 0, core - L + 1)
            if s == 0:
                out[b, c0 - ring:c0, o:o + L] = True
            elif s == 1:
                out[b, c0 + core:c0 + core + ring, o:o + L] = True
            elif s == 2:
                out[b, o:o + L, c0 - ring:c0] = True
            else:
                out[b, o:o + L, c0 + core:c0 + core + ring] = True
    return out, [3] * B, [False] * B


def broken_start(seed: int):
    """A provisional_fn (generators.provisional signature): a very broken start, numpy-seeded (crop-independent)."""
    def fn(B, h, w, gen, dev, spacing, G, nbr, ks, hp_fns, ext_fns, snap_fns, ncfg):
        rng = np.random.default_rng(seed)
        out = torch.zeros((B, h, w), device=dev)
        Gn = G.cpu().numpy()
        for b in range(B):
            s = float(spacing[b])
            fam = NATURAL[int(rng.integers(len(NATURAL)))]
            sign = float(rng.choice([-1.0, 1.0]))
            dmax = max(6.0, rng.uniform(0.6, 1.5) * s)
            A, ell, rough = natural_field(fam, h, w, rng, 4.0, dmax, sign)
            r = ndimage.distance_transform_edt(~Gn[b]) if Gn[b].any() else np.full((h, w), 1e3)
            T = (1 - np.exp(-r / ell)) * (A + 0.6 * rough)
            if rng.random() < 0.5:                                    # a patch on the neighbouring winding
                k = int(rng.choice([-1, 1]))
                a, c = rng.uniform(25, 60), rng.uniform(25, 60)
                cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
                yy, xx = np.mgrid[0:h, 0:w]
                reg = (np.abs(yy - cy) < a / 2) & (np.abs(xx - cx) < c / 2) & ~Gn[b]
                field = np.full((h, w), -k * s)
                if nbr is not None and k in ks:
                    fr = nbr[b, ks.index(k)].cpu().numpy()
                    field = np.where(np.isfinite(fr), fr, field)
                T = np.where(reg, field, T)
            out[b] = torch.as_tensor(np.clip(T, -1.6 * s, 1.6 * s), device=dev, dtype=torch.float32)
        return out, torch.zeros((B, h, w), dtype=torch.int8, device=dev)
    return fn


def _estimate(out, bt, spec):
    """Signed error vs the truth (vox along the ray, core cells): fine head, or the context head where the fine stream
    is invalid or says "outside"."""
    bf, bc = bins_fine(spec, out["fine"].device), bins_ctx(spec, out["fine"].device)
    p = torch.softmax(out["fine"].float(), 1)
    pf = readout(torch.log(p[:, :-1].clamp(min=1e-12)), bf)["pred"]
    lc = out["ctx"][:, 0].float()
    pc = readout(F.log_softmax(lc, 1), bc)["pred"]                          # [B, C/2, C/2]
    C = bt["L"]["U"].shape[-1]
    pc = F.interpolate(pc[:, None], size=(C, C), mode="bilinear", align_corners=False)[:, 0]
    core = bt["t_fine"].shape[-1]
    c0 = (C - core) // 2
    sl = (slice(None), slice(c0, c0 + core), slice(c0, c0 + core))
    err_f = pf - bt["t_fine"]
    err_c = pc[sl] - bt["t_ctx"][sl]
    use_f = bt["fine_ok"] & (p[:, -1] <= 0.5)
    return torch.where(use_f, err_f, err_c)


@torch.no_grad()
def evaluate_adjacent(tr, model, vp, vs, nb: int = 3, B: int = 8, K: int = 3, seed0: int = 5000) -> dict:
    """tr: a Trainer or cli3.Shim; vp / vs: a validation pool and its sampler (their settings are restored)."""
    dev = tr.device
    sp = vs.sp
    c0 = (sp.C - sp.core) // 2
    sl = (slice(None), slice(c0, c0 + sp.core), slice(c0, c0 + sp.core))
    g_old, n_old = dict(vs.gcfg), dict(vs.ncfg)
    vs.gcfg["layout_fn"] = adjacent_greens
    vs.ncfg.update(presence="none", p_rejected=0.0, p_ext_frame=0.0, amp_cap=None)
    acc = {"err": [], "right": [], "agree1": [], "agree2": [], "rb": [], "nb": [], "tot": []}
    dist_acc = {f"{lo}-{hi - 1 if hi == 81 else hi}": ([], []) for lo, hi in BINS}
    try:
        for i in range(nb):
            finals, starts = [], []
            for k in range(K):
                vs.seed(seed0 + i)                                         # same crop and green for every start
                vp.rng = np.random.default_rng(seed0 + i)
                slots = [vp.slots[j % len(vp.slots)] for j in range(i * B, (i + 1) * B)]
                vs.ncfg["provisional_fn"] = broken_start(seed0 * 7 + i * 31 + k)
                L = vs.layout(B, train=False, slots=slots)
                b0 = vs.build(L, L["T_A"], train=False, fine=False)
                b1 = tr.self_condition(b0, model, sampler=vs)
                with torch.autocast(dev.type, dtype=tr.amp, enabled=tr.amp is not None):
                    o1 = model(tr._probe(b1["x_ctx"]), tr._probe(b1["x_fine"]))
                finals.append(_estimate(o1, b1, tr.spec))
                starts.append(L["g_prov"][sl])
            s_vox = torch.tensor(L["spacing"], device=dev, dtype=torch.float32).view(B, 1, 1)
            has_g = (L["G"].flatten(1).sum(1) > min(200, c0 * min(50, sp.core) // 2)).view(B, 1, 1)
            tgt = L["target_ok"][sl] & has_g
            if not bool(tgt.any()):
                continue
            Fk = torch.stack(finals)                                       # [K, B, core, core]
            spread = Fk.max(0).values - Fk.min(0).values
            Gd = Rf.distance_to(L["G"], max_iter=128)[sl]
            for k in range(K):
                e = Fk[k]
                right = e.abs() < 0.5 * s_vox
                broken = starts[k].abs() > 0.5 * s_vox
                acc["err"].append(e[tgt])
                acc["right"].append(right[tgt].float())
                acc["rb"].append(right[tgt & broken].float())
                acc["nb"].append(float((tgt & broken).sum()))
                acc["tot"].append(float(tgt.sum()))
                for lo, hi in BINS:
                    key = f"{lo}-{hi - 1 if hi == 81 else hi}"
                    m = tgt & (Gd >= lo) & (Gd < hi)
                    dist_acc[key][0].append(right[m].float())
                    dist_acc[key][1].append(e[m].abs())
            acc["agree1"].append((spread[tgt] < 1).float())
            acc["agree2"].append((spread[tgt] < 2).float())
    finally:
        vs.gcfg.clear()
        vs.gcfg.update(g_old)
        vs.ncfg.clear()
        vs.ncfg.update(n_old)
    if not acc["err"]:
        return {}
    e = torch.cat(acc["err"]).float()
    if e.numel() > 2_000_000:
        e = e[torch.randperm(e.numel(), device=e.device)[:2_000_000]]
    rb = torch.cat(acc["rb"]) if sum(acc["nb"]) > 0 else torch.zeros(0, device=dev)

    def med(xs):
        x = torch.cat(xs) if xs else torch.zeros(0, device=dev)
        return round(float(x.float().median()), 2) if x.numel() else None

    def mean(xs):
        x = torch.cat(xs) if xs else torch.zeros(0, device=dev)
        return round(float(x.float().mean()), 3) if x.numel() else None

    return {"right": mean(acc["right"]), "p50": round(float(e.abs().median()), 2),
            "p90": round(float(e.abs().quantile(0.9)), 2), "signed": round(float(e.median()), 2),
            "agree1": mean(acc["agree1"]), "agree2": mean(acc["agree2"]),
            "right_broken": round(float(rb.mean()), 3) if rb.numel() else None,
            "broken_share": round(sum(acc["nb"]) / max(1.0, sum(acc["tot"])), 3),
            "by_dist": {k: [mean(v[0]), med(v[1])] for k, v in dist_acc.items()}}


def fmt(r: dict) -> str:
    if not r:
        return "no crops with green"
    return (f"right {r['right']:.3f} | p50/p90 {r['p50']:.2f}/{r['p90']:.2f} signed {r['signed']:+.2f} | agree<1 "
            f"{r['agree1']:.3f} <2 {r['agree2']:.3f} | right where start > 1/2 winding {r['right_broken']} "
            f"(share {r['broken_share']:.2f}) | by distance [right, p50] {r['by_dist']}")
