"""The SurfNet training loop.

Batches (one kind per step):
- A "locate": the context stream on the mode-A frame (the low-pass of the frame source), context heads only;
- B synthetic "refine": a frame from a synthetic post-locate estimate (truth + broadband residual, low-passed in xyz),
  both streams;
- B self-conditioned: the CURRENT model's own locate pass on the mode-A input (no gradient) -> BP (green pinned) ->
  its low-pass in xyz = the frame; both streams. The model learns to correct its own residuals (AlphaFold recycling,
  Analog Bits self-conditioning, RITM's previous-mask input).
Curriculum: steps < c0 (3k): B synthetic only, provisional amplitudes capped at 0.6 windings; c0-c1: no cap; c1-c2
(4k-8k): the shares of A (-> 0.35) and of self-conditioning within B (-> 0.5) ramp in.
Generator shares adapt every `adapt_every` steps: kinds with a larger fine-head error are drawn more often
(x 0.5-2 of the base share).
Leak probes (`crops.probe`): "no_ct" zeroes the CT channels, "no_green" the green prompts.
Validation (fixed seeds) on held-out objects: the A head, the B head on synthetic frames, the full chain A -> BP ->
B; right-winding share; error by distance to green; start influence (the chain from frames shifted by +- 0.45 of a
winding: the slope of the output with the start, ideally 0). Score (lower = better) = chain p90 + 10 x (1 -
right winding) + 5 x slope."""
from __future__ import annotations

import queue
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..common.config import dump
from ..common.log import JsonlWriter, get_logger
from . import fields as Rf
from ..data.registry import open_stores
from ..infer import bp as BP
from ..data.store import SPLIT_HELDOUT, SPLIT_TRAIN
from . import checkpoint as CK
from .ema import EMA
from .schedule import wsd_lambda
from . import generators as GN
from .adjacent import evaluate_adjacent
from .adjacent import fmt as fmt_adjacent
from ..model import NetSpec, TwoStream, bins_ctx, bins_fine, count_params, readout, total_loss
from .pool import Pool
from .sampler import Sampler


def net_spec(cfg: dict) -> NetSpec:
    d = dict(cfg.get("model", {}))
    sp = cfg.get("crops", {}).get("spec", {})
    for k in ("fk", "fout", "ck", "cout"):
        if k in sp:
            d[k] = sp[k]
    return NetSpec.from_dict(d)


def bp_locate(lc: torch.Tensor, bt: dict, s=0.15, tau=8.0, iters=12, compiled=False) -> torch.Tensor:
    """Context logits [B, nc, h, w] (target winding, at the head's 2-cell resolution) -> BP solution in vox along the
    ray, solved on the 2-cell grid (green pinned at their targets) and interpolated to the cells. The frame built from
    it is low-passed over 3-8 cells, so the coarser solve loses nothing (1.9 -> ~0.25 s per batch)."""
    L = bt["L"]
    C = L["U"].shape[-1]
    nc = lc.shape[1]
    cout = (nc - 1) / 2
    sub = (slice(None), slice(0, None, 2), slice(0, None, 2))
    D = (-torch.log_softmax(lc.float(), 1)).clamp(max=10.0)
    T = bt["T"][sub]
    dx = (T[:, :, 1:] - T[:, :, :-1]).float()                                   # height step per 2-cell edge (vox = labels)
    dy = (T[:, 1:] - T[:, :-1]).float()
    pin = torch.where(L["G"][sub], (bt["t_ctx"][sub] + cout).clamp(0, nc - 1), torch.full_like(T, float("nan")))
    r = BP.solve(D, L["U"][sub], dx, dy, s=s, tau=tau, iters=iters, pin_label=pin, compiled=compiled)
    t = (r["label"] - cout)[:, None]
    return torch.nn.functional.interpolate(t, size=(C, C), mode="bilinear", align_corners=False)[:, 0]


class Producer:
    """Background batch synthesis (layout + the inputs that do not need the model) on its own CUDA stream."""

    def __init__(self, tr, depth=4):
        self.tr = tr
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        # high priority: the synthesis' many small kernels are not queued behind the model's large convolutions
        self.stream = torch.cuda.Stream(tr.device, priority=-1) if tr.device.type == "cuda" else None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        import contextlib
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()
        try:
            with ctx:
                while not self._stop.is_set():
                    item = self.tr.make_batch()
                    ev = None
                    if self.stream is not None:
                        ev = torch.cuda.Event()
                        ev.record(self.stream)
                    while not self._stop.is_set():
                        try:
                            self.q.put((item, ev), timeout=0.5)
                            break
                        except queue.Full:
                            continue
        except Exception as e:
            self.q.put((e, None))

    def get(self):
        item, ev = self.q.get()
        if isinstance(item, Exception):
            raise item
        if ev is not None:
            cur = torch.cuda.current_stream(self.tr.device)
            cur.wait_event(ev)
            _record(item, cur)            # memory made on the producer's stream is in use on this one until freed
        return item


    def close(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.th.join(timeout=5)


def _record(obj, stream, depth=0):
    """record_stream on every CUDA tensor of a (nested) batch, so the caching allocator never hands its memory back to
    the producer while kernels queued on `stream` (e.g. the backward pass) may still read it."""
    if depth > 4:
        return
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            _record(v, stream, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _record(v, stream, depth + 1)

class Trainer:
    def __init__(self, cfg: dict, run_dir, device="cuda", resume=None):
        self.cfg = cfg
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("train3", self.run / "train.log")
        self.mlog = JsonlWriter(self.run / "metrics.jsonl")
        self.device = torch.device(device)
        dump(cfg, self.run / "config.yaml")
        torch.manual_seed(int(cfg.get("seed", 0)))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        d = cfg["data"]
        stores = open_stores(d["root"], d.get("include", ["*"]), d.get("exclude", []))
        if not stores:
            raise SystemExit("no training stores")
        self.log.info(f"training objects: {len(stores)}: {', '.join(s.meta['id'] for s in stores)}")
        kw = dict(region=int(d.get("region", 320)), slots=int(d.get("slots", 24)), device=self.device,
                  object_weights=d.get("object_weights", {}), seed=int(cfg.get("seed", 0)),
                  prefetch=bool(d.get("prefetch", True)), min_usable=int(d.get("min_usable", 64)))
        self.pool = Pool(stores, split=SPLIT_TRAIN, **kw)
        self.sampler = Sampler(self.pool, cfg, self.device, seed=int(cfg.get("seed", 0)))
        self.gen_main = torch.Generator(device=self.device)
        self.gen_main.manual_seed(int(cfg.get("seed", 0)) + 77)
        self.vals = []
        for v in cfg.get("val", {}).get("sets", []):
            vs = open_stores(d["root"], v.get("include", ["*"]), v.get("exclude", []))
            if not vs:
                self.log.info(f"validation {v['name']}: no stores")
                continue
            split = {"heldout": SPLIT_HELDOUT, "any": None, "train": SPLIT_TRAIN}[v.get("split", "any")]
            vk = dict(kw, slots=int(v.get("regions", 12)), prefetch=False, seed=0)
            try:
                vp = Pool(vs, split=split, **vk)
            except ValueError:
                self.log.info(f"validation {v['name']}: no usable cells")
                continue
            self.vals.append((v["name"], vp, Sampler(vp, cfg, self.device, seed=0)))
            self.log.info(f"validation {v['name']}: {', '.join(s.meta['id'] for s in vp.stores)}")
        self.spec = net_spec(cfg)
        self.model = TwoStream(self.spec).to(self.device)
        self.log.info(f"model: {count_params(self.model) / 1e6:.1f} M parameters; {self.spec}")
        oc = cfg.get("optim", {})
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(oc.get("lr", 1e-3)),
                                     weight_decay=float(oc.get("wd", 0.05)), betas=tuple(oc.get("betas", (0.9, 0.99))),
                                     fused=self.device.type == "cuda")
        self.steps = int(oc.get("steps", 70000))
        warm = int(oc.get("warmup", 1000))
        minr = float(oc.get("min_lr", 1e-5)) / float(oc.get("lr", 1e-3))
        self.cool_frac = float(oc.get("cooldown_frac", 0.2))
        self.cool_min = int(oc.get("cooldown_min", 2500))
        self.cool_start, self.cool_len = None, 0
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: wsd_lambda(s, warm, self.cool_start,
                                                                                     self.cool_len, minr))
        tc = cfg.get("train", {})
        self.ema = EMA(self.model, float(tc.get("ema", 0.999)))
        self.amp = torch.bfloat16 if tc.get("amp", "bf16") == "bf16" else None
        self.compiled = bool(tc.get("compile", True)) and self.device.type == "cuda"
        self.fwd = torch.compile(self.model) if self.compiled else self.model
        self.step, self.best, self.since_best = 0, float("inf"), 0
        crops = cfg.get("crops", {})
        self.probe = crops.get("probe")
        cur = crops.get("curriculum", {})
        self.c0, self.c1, self.c2 = int(cur.get("c0", 3000)), int(cur.get("c1", 4000)), int(cur.get("c2", 8000))
        self.pA_max, self.pself_max = float(cur.get("p_A", 0.35)), float(cur.get("p_self", 0.5))
        self.adapt_every = int(crops.get("adapt_every", 500))
        self.base_shares = list(GN.SHARES)
        self.kind_err = torch.zeros(len(GN.GENS), device=self.device)
        self.kind_cnt = torch.zeros(len(GN.GENS), device=self.device)
        if resume:
            self._resume(resume)
        init = tc.get("init")
        if init and self.step == 0:                                   # fine-tune: weights only, fresh optimiser / schedule
            st = torch.load(init, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st.get("model", st["ema"]))          # an exported model has the EMA only
            self.ema.load_state_dict(st["ema"])
            self.log.info(f"initialised from {init} (step {st.get('step')}: model + EMA weights; fresh optimiser)")
        self._stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))

    # ------------------------------------------------------------------ batch synthesis
    def _mode(self, step: int) -> str:
        gen = self.sampler.gen_cpu
        t = 0.0 if step < self.c1 else min(1.0, (step - self.c1) / max(1, self.c2 - self.c1))
        u = float(torch.rand(1, generator=gen))
        if u < self.pA_max * t:
            return "A"
        return "Bself" if float(torch.rand(1, generator=gen)) < self.pself_max * t else "Bsyn"

    def make_batch(self) -> dict:
        B = int(self.cfg["data"].get("batch", 8))
        self.pool.refresh(int(self.cfg["data"].get("refresh_per_step", 1)))
        self.sampler.ncfg["amp_cap"] = 0.6 if self.step < self.c0 else None
        mode = self._mode(self.step)
        L = self.sampler.layout(B, train=True)
        if mode == "A":
            return {"mode": "A", "bt": self.sampler.build(L, L["T_A"], train=True, fine=False)}
        if mode == "Bsyn":
            T = self.sampler.refine_frame(L, self.sampler.synthetic_refine_offset(L))
            return {"mode": "Bsyn", "bt": self.sampler.build(L, T, train=True, fine=True)}
        return {"mode": "Bself", "bt": self.sampler.build(L, L["T_A"], train=True, fine=False)}

    def _probe(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "amp", None) is None:
            x = x.float()                                  # bf16 inputs are for autocast runs; fp32 runs need fp32
        if self.probe == "no_ct":
            x = x.clone()
            x[:, 0] = 0
        elif self.probe == "no_green":
            x = x.clone()
            x[:, 1:4] = 0
        return x

    @torch.no_grad()
    def self_condition(self, btA: dict, model, sampler=None, gen=None) -> dict:
        """The model's own locate pass -> BP -> frame -> a B batch (with its own generator: the producer thread uses
        the sampler's)."""
        sampler = sampler or self.sampler
        L = btA["L"]
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
            lc = model(self._probe(btA["x_ctx"]))["ctx"][:, 0]
        t = bp_locate(lc, btA, compiled=getattr(self, "compiled", False) and model.training)
        o = btA["T"] + btA["sigc"] * t                                                 # the estimate, vox along N
        o = torch.where(L["U"], o, btA["T"])
        T_B = sampler.refine_frame(L, o, gen=gen)
        return sampler.build(L, T_B, train=model.training, fine=True, gen=gen)

    # ------------------------------------------------------------------ checkpoints
    def _state(self):
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "step": self.step, "best": self.best, "since_best": self.since_best,
                "cfg": self.cfg, "spec": vars(self.spec), "cool_start": self.cool_start, "cool_len": self.cool_len,
                "rng": CK.rng_state(self.sampler, self.pool), "shares": self.sampler.ncfg.get("shares")}

    def save(self, tag=None):
        st = self._state()
        CK.save(self.run / f"ckpt_{self.step:07d}.pt", **st)
        CK.save(self.run / "last.pt", **st)
        if tag:
            CK.save(self.run / f"{tag}.pt", **st)
        CK.rotate(self.run, int(self.cfg.get("train", {}).get("keep", 3)))

    def _resume(self, resume):
        p = CK.latest(self.run) if resume == "auto" else Path(resume)
        if p is None or not p.exists():
            self.log.info("resume: none")
            return
        st = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.ema.load_state_dict(st["ema"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.step, self.best, self.since_best = int(st["step"]), float(st["best"]), int(st.get("since_best", 0))
        self.cool_start, self.cool_len = st.get("cool_start"), int(st.get("cool_len", 0))
        if st.get("shares"):
            self.sampler.ncfg["shares"] = st["shares"]
        CK.set_rng_state(st["rng"], self.sampler, self.pool)
        self.log.info(f"resumed from {p} at step {self.step}")

    # ------------------------------------------------------------------ loop
    def train(self):
        tc = self.cfg.get("train", {})
        log_every, val_every = int(tc.get("log_every", 50)), int(tc.get("val_every", 2500))
        ckpt_every = int(tc.get("ckpt_every", 2500))
        clip = float(self.cfg.get("optim", {}).get("clip", 1.0))
        lcfg = self.cfg.get("loss", {})
        self.model.train()
        prod = Producer(self) if tc.get("async_data", True) else None
        acc, nacc = {}, {}
        t_log, t_data, t_comp, t_self = time.time(), 0.0, 0.0, 0.0
        while self.step < self.steps:
            if self._stop or (self.run / "STOP").exists():
                self.log.info("stop requested")
                self.save()
                break
            t0 = time.perf_counter()
            item = prod.get() if prod is not None else self.make_batch()
            mode, bt = item["mode"], item["bt"]
            ts = time.perf_counter()
            if mode == "Bself":
                bt = self.self_condition(bt, self.model, gen=self.gen_main)
            t1 = time.perf_counter()
            t_self += t1 - ts
            with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
                out = self.fwd(self._probe(bt["x_ctx"]), self._probe(bt["x_fine"]) if mode != "A" else None)
            ls = total_loss(out, bt, self.spec, "A" if mode == "A" else "B", lcfg)
            self.opt.zero_grad(set_to_none=True)
            ls["total"].backward()
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.opt.step()
            self.sched.step()
            self.ema.update(self.model, self.step)
            self.step += 1
            if mode != "A":
                self._track_kinds(out, bt)
            t2 = time.perf_counter()
            t_data += t1 - t0
            t_comp += t2 - t1
            for k, v in ls.items():
                key = f"{mode}/{k}"
                acc[key] = acc.get(key, 0.0) + v.detach()
                nacc[key] = nacc.get(key, 0) + 1
            acc["gnorm"] = acc.get("gnorm", 0.0) + gn.detach()
            nacc["gnorm"] = nacc.get("gnorm", 0) + 1
            if self.step % self.adapt_every == 0 and self.step >= self.c1:
                self._adapt()
            if self.step % log_every == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                el = time.time() - t_log
                rec = {k: float(v) / nacc[k] for k, v in acc.items()}
                rec.update(step=self.step, lr=self.sched.get_last_lr()[0], s_per_step=el / log_every,
                           data_frac=t_data / max(t_data + t_comp, 1e-9),
                           wait_frac=(t_data - t_self) / max(el, 1e-9), self_frac=t_self / max(el, 1e-9))
                if self.device.type == "cuda":
                    rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                self.mlog.write(kind="train", **rec)
                self.log.info(f"step {self.step}: " + " ".join(f"{k} {v:.3f}" for k, v in sorted(rec.items())
                                                                if "/" in k) + f" lr {rec['lr']:.2e} "
                              f"{rec['s_per_step']:.3f} s/step data {100 * rec['data_frac']:.0f} % (waiting "
                              f"{100 * rec['wait_frac']:.0f} %, self-conditioning {100 * rec['self_frac']:.0f} %)")
                acc, nacc = {}, {}
                t_data = t_comp = t_self = 0.0
                t_log = time.time()
            if self.vals and self.step % val_every == 0:
                self.evaluate()
                pat = int(tc.get("patience", 3))
                if self.since_best >= pat and self.step >= int(tc.get("min_steps", 0)) and self.cool_start is None:
                    self._cooldown(f"validation plateau ({pat} evaluations)")
            if self.cool_start is None and self.step >= self.steps - max(self.cool_min, int(self.cool_frac * self.steps)):
                self._cooldown("step cap", self.steps - self.step)
            if self.cool_start is not None and self.step >= self.cool_start + self.cool_len:
                self.log.info(f"cooldown finished at step {self.step}")
                if self.vals and self.step % val_every != 0:
                    self.evaluate()
                self.save()
                break
            if self.step % ckpt_every == 0:
                self.save()
        else:
            self.save()
        if prod is not None:
            prod.close()
        self.pool.close()

    def _cooldown(self, why, length=None):
        self.cool_start = self.step
        self.cool_len = max(1, length) if length is not None else max(self.cool_min, int(self.cool_frac * self.step))
        self.steps = max(self.steps, self.cool_start + self.cool_len)
        self.log.info(f"cooldown: {why}; {self.cool_len} steps")

    def _track_kinds(self, out, bt):
        with torch.no_grad():
            bf = bins_fine(self.spec, self.device)
            p = torch.softmax(out["fine"].float(), 1)
            pred = (p[:, :-1] * bf.view(1, -1, 1, 1)).sum(1) / p[:, :-1].sum(1).clamp(min=1e-6)
            err = torch.where(p[:, -1] > 0.5, torch.full_like(pred, self.spec.fout), (pred - bt["t_fine"]).abs())
            w = bt["w_fine"] > 0
            sp = self.sampler.sp
            c0 = (sp.C - sp.core) // 2
            kind = bt["L"]["kind"][:, c0:c0 + sp.core, c0:c0 + sp.core].long()
            for k in range(len(GN.GENS)):
                m = w & (kind == k)
                self.kind_err[k] += (err * m).sum()
                self.kind_cnt[k] += m.sum()

    def _adapt(self):
        e = (self.kind_err / self.kind_cnt.clamp(min=1)).cpu().numpy()
        cnt = self.kind_cnt.cpu().numpy()
        ok = cnt > 100
        if ok.sum() < 2:
            return
        mean = float(e[ok].mean())
        fac = np.where(ok, np.clip(e / max(mean, 1e-6), 0.5, 2.0), 1.0)
        sh = [float(b * f) for b, f in zip(self.base_shares, fac)]
        self.sampler.ncfg["shares"] = sh
        self.log.info("generator shares: " + ", ".join(f"{g} {s:.3f} (err {x:.2f})" for g, s, x in zip(GN.GENS, sh, e)))
        self.kind_err.zero_()
        self.kind_cnt.zero_()

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def evaluate(self) -> dict:
        t0 = time.time()
        model = self.ema.shadow
        model.eval()
        res = {}
        nb = int(self.cfg.get("val", {}).get("batches", 6))
        B = int(self.cfg["data"].get("batch", 8))
        scores = []
        vc = self.cfg.get("val", {})
        for name, vp, vs in self.vals:
            r = evaluate_pool(self, model, vp, vs, nb, B)
            res[name] = r
            scores.append(r["score"])
            self.log.info(f"val[{name}] step {self.step}: score {r['score']:.3f} | A p50/p90 {r['A'][0]:.2f}/{r['A'][1]:.2f} "
                          f"right-winding {r['A_right']:.3f} | B(syn) {r['Bsyn'][0]:.2f}/{r['Bsyn'][1]:.2f} | chain "
                          f"{r['chain'][0]:.2f}/{r['chain'][1]:.2f} right {r['chain_right']:.3f} | slope {r['slope']:.3f} | "
                          f"by distance {r['by_dist']}")
            # 1-2 locked areas, the adjacent area very broken (reported; not in the score)
            r["adjacent"] = evaluate_adjacent(self, model, vp, vs, int(vc.get("adjacent_batches", 3)), B,
                                              int(vc.get("adjacent_starts", 3)))
            self.log.info(f"val[{name}] step {self.step}: ADJACENT {fmt_adjacent(r['adjacent'])}")
        model.train()
        sc = float(np.mean(scores)) if scores else float("inf")
        res["score"] = sc
        self.mlog.write(kind="val", step=self.step, results=res)
        self.log.info(f"val step {self.step}: score {sc:.3f} [{time.time() - t0:.0f} s]")
        keep = self.run / "keep"                                    # every validated checkpoint is kept (not rotated)
        keep.mkdir(exist_ok=True)
        CK.save(keep / f"step_{self.step:07d}.pt", **self._state())
        if sc < self.best:
            self.best, self.since_best = sc, 0
            self.save("best")
        else:
            self.since_best += 1
        return res


def _q(x: torch.Tensor):
    if x.numel() == 0:
        return [float("nan"), float("nan")]
    x = x.float()
    if x.numel() > 2_000_000:
        x = x[torch.randperm(x.numel(), device=x.device)[:2_000_000]]
    return [float(x.median()), float(torch.quantile(x, 0.9))]


@torch.no_grad()
def evaluate_pool(tr: Trainer, model, vp, vs: Sampler, nb: int, B: int) -> dict:
    """Fixed-seed crops: A head, B head (synthetic frames), the chain A -> BP -> B, start influence."""
    dev = tr.device
    spec = tr.spec
    bc, bf = bins_ctx(spec, dev), bins_fine(spec, dev)
    eA, eB, eC, rightA, rightC, slopes, dist_err = [], [], [], [], [], [], {}
    for i in range(nb):
        vs.seed(1000 + i)
        vp.rng = np.random.default_rng(1000 + i)
        slots = [vp.slots[j % len(vp.slots)] for j in range(i * B, (i + 1) * B)]
        L = vs.layout(B, train=False, slots=slots)
        sp_vox = torch.tensor(L["spacing"], device=dev).view(B, 1, 1)
        btA = vs.build(L, L["T_A"], train=False, fine=False)
        with torch.autocast(dev.type, dtype=tr.amp, enabled=tr.amp is not None):
            outA = model(tr._probe(btA["x_ctx"]))
        lc = outA["ctx"][:, 0]
        pa = readout(lc.float(), bc)["pred"]
        sub = (slice(None), slice(0, None, 2), slice(0, None, 2))
        wA = btA["w_ctx"][sub] > 0
        errA = (pa - btA["t_ctx"][sub]).abs()
        eA.append(errA[wA])
        rightA.append((errA < 0.5 * sp_vox[:, :1, :1])[wA].float())
        # B on synthetic frames
        Ts = vs.refine_frame(L, vs.synthetic_refine_offset(L), sig=[6.0] * B)
        btB = vs.build(L, Ts, train=False, fine=True)
        with torch.autocast(dev.type, dtype=tr.amp, enabled=tr.amp is not None):
            outB = model(tr._probe(btB["x_ctx"]), tr._probe(btB["x_fine"]))
        eB.append(_fine_err(outB, btB, bf, spec)[btB["w_fine"] > 0])
        # the chain A -> BP -> B, from the mode-A frame and from frames shifted by +- 0.45 winding
        finals = []
        for sh in (0.0, 0.45, -0.45):
            T0 = (L["T_A"] + sh * sp_vox).clamp(-vs.sp.ctx_max, vs.sp.ctx_max)
            b0 = vs.build(L, T0, train=False, fine=False)
            b1 = tr.self_condition(b0, model, sampler=vs)
            with torch.autocast(dev.type, dtype=tr.amp, enabled=tr.amp is not None):
                o1 = model(tr._probe(b1["x_ctx"]), tr._probe(b1["x_fine"]))
            fin = _fine_est(o1, b1, bf, spec)                                    # vox along N, relative to the truth
            finals.append(fin)
            if sh == 0.0:
                w = b1["w_fine"] > 0
                eC.append(fin.abs()[w])
                rightC.append((fin.abs() < 0.5 * sp_vox[:, :1, :1])[w].float())
                sp = vs.sp
                c0 = (sp.C - sp.core) // 2
                Gd = Rf.distance_to(L["G"], max_iter=128)[:, c0:c0 + sp.core, c0:c0 + sp.core]
                for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 129)):
                    m = w & (Gd >= lo) & (Gd < hi)
                    dist_err.setdefault(f"{lo}-{hi}", []).append(fin.abs()[m])
        w = (b1["w_fine"] > 0)
        slopes.append(((finals[1] - finals[2]).abs() / (0.9 * sp_vox))[w])
    A = _q(torch.cat(eA))
    r = {"A": A, "A_right": float(torch.cat(rightA).mean()), "Bsyn": _q(torch.cat(eB)), "chain": _q(torch.cat(eC)),
         "chain_right": float(torch.cat(rightC).mean()), "slope": float(torch.cat(slopes).median()),
         "by_dist": {k: round(_q(torch.cat(v))[0], 2) for k, v in dist_err.items()}}
    r["score"] = r["chain"][1] + 10 * (1 - r["chain_right"]) + 5 * r["slope"]
    return r


def _fine_est(out, bt, bf, spec) -> torch.Tensor:
    """The fine head's estimate as an error vs the truth (vox along the ray; 'outside' -> the context residual)."""
    p = torch.softmax(out["fine"].float(), 1)
    pred = readout(torch.log(p[:, :-1].clamp(min=1e-12)), bf)["pred"]
    err = pred - bt["t_fine"]
    return torch.where(p[:, -1] > 0.5, bt["t_fine"].abs().clamp(min=spec.fout), err)


def _fine_err(out, bt, bf, spec) -> torch.Tensor:
    return _fine_est(out, bt, bf, spec).abs()
