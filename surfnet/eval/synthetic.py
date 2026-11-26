"""Held-out tests of a SurfNet model on the bands of held-out objects. Start families NEVER used in training:

| family | start (offset along the true normal) |
|---|---|
| hybrid / warped / plates / streaks / ridged | the natural procedural fields of `infer/perturb.py` (Musgrave multifractals, Quilez domain warping, Worley plates, fibre streaks), \\|A\\| 4-11 vox, diverging from green over 2-14 cells, noise within |
| winding1 / winding2 | an abrupt tile-shaped region on the +-1 / +-2 neighbouring winding (no ramp, no residual) |
| antishape | low-pass minus high-pass of the truth (sigma 6): the truth's fine shape inverted |

Per test object set, family, green class (none / sparse / tiles / large / mostly) and hint (on: the start is given
as the soft prompt; off: not given), fixed seeds: the chain A -> BP -> B (as validation), |error| p50 / p90,
right-winding share, error by distance to green, the start's influence (slope between frames shifted by +-0.45
winding), and the spread between two different starts of the same family on the same crops."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import torch

from ..train import fields as Rf
from ..infer.perturb import FAMILIES as NATURAL, natural_field
from ..data.store import SPLIT_HELDOUT
from ..train import greens as GR
from ..train.pool import Pool
from ..train.sampler import Sampler

TEST_FAMILIES = ("hybrid", "warped", "plates", "streaks", "ridged", "winding1", "winding2", "antishape")


def make_provisional(family: str, seed: int):
    """A provisional_fn for Sampler (same signature as generators.provisional)."""
    def fn(B, h, w, gen, dev, spacing, G, nbr, ks, hp_fns, ext_fns, snap_fns, ncfg):
        out = torch.zeros((B, h, w), device=dev)
        rng = np.random.default_rng(seed)
        for b in range(B):
            s = float(spacing[b])
            if family in ("hybrid", "warped", "plates", "streaks", "ridged"):
                fam = NATURAL[("hybrid", "warped", "plates", "streaks", "ridged").index(family)]
                sign = float(rng.choice([-1.0, 1.0]))
                A, ell, rough = natural_field(fam, h, w, rng, 4.0, 11.0, sign)
                Gn = G[b].cpu().numpy()
                from scipy import ndimage
                r = ndimage.distance_transform_edt(~Gn) if Gn.any() else np.full((h, w), 1e3)
                T = np.clip((1 - np.exp(-r / ell)) * (A + 0.6 * rough), -12, 12)
                out[b] = torch.as_tensor(T, device=dev, dtype=torch.float32)
            elif family in ("winding1", "winding2"):
                k = (1 if family == "winding1" else 2) * int(rng.choice([-1, 1]))
                a, c = rng.uniform(30, 70), rng.uniform(30, 70)
                cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
                yy, xx = Rf.grid(h, w, dev)
                reg = ((yy - cy).abs() < a / 2) & ((xx - cx).abs() < c / 2) & ~G[b]
                field = torch.full((h, w), -k * s, device=dev)
                if nbr is not None and k in ks:
                    fr = nbr[b, ks.index(k)]
                    field = torch.where(torch.isfinite(fr), fr, field)
                out[b] = torch.where(reg, field, torch.zeros_like(field))
            else:                                                       # antishape: LP(truth) - HP(truth)
                out[b] = -2.0 * hp_fns[b](6.0)
        return out, torch.zeros((B, h, w), dtype=torch.int8, device=dev)
    return fn


@torch.no_grad()
def run_tests(tr, model, sets: list[dict], out_path: Path, batches: int = 4, B: int = 8, log=None) -> dict:
    """tr: a Trainer (model / spec / amp / helpers); sets: [{name, include, split}]."""
    from ..data.registry import open_stores
    from ..train.trainer import evaluate_pool
    res = {}
    d = tr.cfg["data"]
    for st in sets:
        stores = open_stores(d["root"], st.get("include", ["*"]), st.get("exclude", []))
        if not stores:
            continue
        split = {"heldout": SPLIT_HELDOUT, "any": None}[st.get("split", "any")]
        pool = Pool(stores, region=int(d.get("region", 320)), slots=int(st.get("regions", 12)), device=tr.device,
                     split=split, seed=0, prefetch=False, min_usable=64)
        for fam in TEST_FAMILIES:
            for gi, gname in enumerate(GR.CLASSES):
                for hint in ("off",):                                  # the trace enters only as the frame
                    cfg = json.loads(json.dumps(tr.cfg))
                    shares = [0.0] * len(GR.CLASSES)
                    shares[gi] = 1.0
                    cfg.setdefault("crops", {})["greens"] = {"shares": shares, "far_p": 0.0}
                    vs = Sampler(pool, cfg, tr.device, seed=0)
                    vs.ncfg.update(provisional_fn=make_provisional(fam, 7), presence="all" if hint == "on" else "none",
                                   p_rejected=0.0)
                    r = evaluate_pool(tr, model, pool, vs, batches, B)
                    vs2 = Sampler(pool, cfg, tr.device, seed=0)                  # a second start, same crops
                    vs2.ncfg.update(provisional_fn=make_provisional(fam, 8), presence=vs.ncfg["presence"], p_rejected=0.0)
                    r2 = evaluate_pool(tr, model, pool, vs2, batches, B)
                    r["start_spread_chain_p50"] = abs(r["chain"][0] - r2["chain"][0])
                    key = f"{st['name']}/{fam}/{gname}/hint_{hint}"
                    res[key] = r
                    if log:
                        log.info(f"TEST {key}: chain {r['chain'][0]:.2f}/{r['chain'][1]:.2f} right {r['chain_right']:.3f} "
                                 f"slope {r['slope']:.3f} by distance {r['by_dist']}")
        pool.close()
    out_path.write_text(json.dumps(res, indent=1))
    return res


class Shim:
    """What evaluate_pool / run_tests need from a trainer, around a loaded checkpoint."""

    def __init__(self, ckpt: str, device="cuda"):
        from ..model import NetSpec, TwoStream
        from ..train.trainer import Trainer
        st = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = st["cfg"]
        self.spec = NetSpec.from_dict(st["spec"])
        self.device = torch.device(device)
        self.amp = torch.bfloat16
        self.probe = None
        self.model = TwoStream(self.spec).to(device)
        self.model.load_state_dict(st["ema"])
        self.model.eval()
        self.step = st["step"]
        self._probe = types.MethodType(Trainer._probe, self)
        self.self_condition = types.MethodType(Trainer.self_condition, self)
        self.sampler = None
