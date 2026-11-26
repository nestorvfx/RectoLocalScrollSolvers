"""GPU-resident pool of band regions for training-sample synthesis.

A region = R x R cells of one object: its fine OR coarse band (uint8, kept on the device) + per-cell maps:
- usable: the input may show the cell;
- tier, split;
- neg / neg_tile: rejected cells (known wrong) and their tile ids;
- nbr: neighbour-winding offsets;
- real: actual error fields of our tracing history;
- axdir: the scroll-axis direction in grid coordinates;
- onsheet: the mesh lies on the papyrus (`prep/onsheet.py`; all True when not computed);
- the object's face offset.
A background thread reads new regions from the band memmaps and pins them in host memory; `refresh(n)` swaps n slots
for ready regions (non-blocking copies), so disk I/O never stalls the training step.

Usable cells = valid & split == the pool's split (None: every non-excluded cell). Offset targets additionally need
tier > 0; user-rejected cells (neg) are inputs and verify negatives only. Cells that are not usable are invalid inputs
(their CT is zeroed by the sampler), so held-out or excluded geometry never enters a training crop.
Objects are drawn with probability proportional to (usable target cells x object weight); within an object, region
origins are drawn around tiles holding such cells (or, with `focus`, cells of that split: validation on held-out
tiles surrounded by their real context)."""
from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from ..common.geometry import axis_direction
from ..data.store import SPLIT_EXCLUDED, SPLIT_TRAIN, BandStore


class RegionPool:
    def __init__(self, stores: list[BandStore], kind: str, region: int = 256, slots: int = 24, device="cuda",
                 split: int | None = SPLIT_TRAIN, object_weights: dict | None = None, seed: int = 0,
                 prefetch: bool = True, min_usable: int = 64, focus: int | None = None):
        assert kind in ("fine", "coarse")
        self.kind = kind
        self.R = region
        self.device = torch.device(device)
        self.split = split                                   # SPLIT_* or None (any non-excluded cell)
        self.rng = np.random.default_rng(seed)
        self.stores, self.weights, self.tiles = [], [], []
        ow = object_weights or {}
        for s in stores:
            g = s.geom
            neg = g["neg"] if "neg" in g else np.zeros_like(g["V"])
            if focus is None:
                U = g["V"] & self._split_ok(g["split"]) & ((g["tier"] > 0) | neg)
            else:
                U = g["V"] & (g["split"] == focus) & (g["tier"] > 0)
            T = s.spec.tile
            cnt = np.zeros((s.nty, s.ntx), np.int64)
            for ty in range(s.nty):
                for tx in range(s.ntx):
                    cnt[ty, tx] = U[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T].sum()
            tl = np.argwhere(cnt >= min_usable)
            if len(tl) == 0:
                continue
            w = float(U.sum()) * float(ow.get(s.meta["id"], ow.get(s.meta.get("scroll", ""), 1.0)))
            if w <= 0:
                continue
            self.stores.append(s)
            self.weights.append(w)
            self.tiles.append((tl, cnt[tl[:, 0], tl[:, 1]].astype(np.float64)))
        if not self.stores:
            raise ValueError("no object has usable cells for this pool")
        self.weights = np.asarray(self.weights) / np.sum(self.weights)
        self._axdir = {}
        self.n_slots = slots
        self.q: queue.Queue = queue.Queue(maxsize=max(4, slots // 2))
        self._stop = threading.Event()
        self.slots: list[dict] = []
        for _ in range(slots):
            self.slots.append(self._to_device(self._pin(self._load_random())))
        self._th = None
        if prefetch:
            self._th = threading.Thread(target=self._worker, daemon=True)
            self._th.start()

    def _split_ok(self, split: np.ndarray) -> np.ndarray:
        if self.split is None:
            return split != SPLIT_EXCLUDED
        return split == self.split

    # ------------------------------------------------------------------ loading
    def _load_random(self) -> dict:
        si = int(self.rng.choice(len(self.stores), p=self.weights))
        tl, cnt = self.tiles[si]
        t = tl[int(self.rng.choice(len(tl), p=cnt / cnt.sum()))]
        s = self.stores[si]
        T = s.spec.tile
        r0 = int(t[0] * T + T // 2 - self.R // 2 + self.rng.integers(-T // 2, T // 2 + 1))
        c0 = int(t[1] * T + T // 2 - self.R // 2 + self.rng.integers(-T // 2, T // 2 + 1))
        return self.load(si, r0, c0)

    def _axis_dir(self, si: int) -> np.ndarray:
        if si not in self._axdir:
            g = self.stores[si].geom
            self._axdir[si] = axis_direction(g["P"].astype(np.float64), g["V"])
        return self._axdir[si]

    def load(self, si: int, r0: int, c0: int) -> dict:
        s = self.stores[si]
        g = s.geom
        R = self.R
        band = s.read_region(r0, c0, R, which=(self.kind,))[self.kind]
        V = s.cells("V", r0, c0, R, fill=False)
        split = s.cells("split", r0, c0, R, fill=255)
        tier = s.cells("tier", r0, c0, R, fill=0)
        ax = self._axis_dir(si)
        axc = np.zeros((2, R, R), np.float32)
        ra, rb, ca, cb = max(r0, 0), min(r0 + R, s.H), max(c0, 0), min(c0 + R, s.W)
        if ra < rb and ca < cb:
            axc[:, ra - r0:rb - r0, ca - c0:cb - c0] = ax[:, ra:rb, ca:cb]
        d = {"ct": band, "usable": V & self._split_ok(split), "tier": tier, "split": split.astype(np.uint8),
             "neg": s.cells("neg", r0, c0, R, fill=False) if "neg" in g else np.zeros((R, R), bool),
             "neg_tile": s.cells("neg_tile", r0, c0, R, fill=-1).astype(np.int32) if "neg_tile" in g
             else np.full((R, R), -1, np.int32),
             "axdir": axc,
             "id": s.meta["id"], "origin": (r0, c0),
             "cell_vox": float(s.meta.get("cell_vox", float(s.meta["cell_um"]) / float(s.meta["pitch_um"]))),
             "spacing": float(s.meta.get("spacing_vox", 12.0)),
             "face": float(s.meta.get("face_offset_vox", 0.0)),
             "ks": list(s.meta.get("winding_index_offsets", [])),
             "spec": s.spec}
        d["onsheet"] = s.cells("onsheet", r0, c0, R, fill=False) if "onsheet" in g else np.ones((R, R), bool)
        if "nbr" in g:
            d["nbr"] = s.cells("nbr", r0, c0, R, fill=np.nan).astype(np.float32)
        if "real_err" in g:
            d["real"] = s.cells("real_err", r0, c0, R, fill=np.nan).astype(np.float32)
        return d

    @staticmethod
    def _pin(d: dict) -> dict:
        """numpy -> torch (pinned when CUDA is available), in the loader thread."""
        out = dict(d)
        for k in ("ct", "usable", "tier", "split", "neg", "neg_tile", "axdir", "onsheet", "nbr", "real"):
            if k in d:
                t = torch.from_numpy(np.ascontiguousarray(d[k]))
                out[k] = t.pin_memory() if torch.cuda.is_available() else t
        return out

    def _to_device(self, d: dict) -> dict:
        out = dict(d)
        for k in ("ct", "usable", "tier", "split", "neg", "neg_tile", "axdir", "onsheet", "nbr", "real"):
            if k in d:
                out[k] = d[k].to(self.device, non_blocking=True)
        return out

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._pin(self._load_random())
            except Exception as e:                                   # keep training alive; report and retry
                print(f"[pool] load failed: {e}", flush=True)
                continue
            while not self._stop.is_set():
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    # ------------------------------------------------------------------ use
    def refresh(self, n: int = 1) -> int:
        done = 0
        for _ in range(n):
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            self.slots[int(self.rng.integers(len(self.slots)))] = self._to_device(item)
            done += 1
        return done

    def pick(self, B: int, gen: torch.Generator | None = None) -> list[dict]:
        idx = torch.randint(0, len(self.slots), (B,), generator=gen).tolist()
        return [self.slots[i] for i in idx]

    def close(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)


KEYS = ("ct", "ctc", "usable", "tier", "split", "neg", "neg_tile", "axdir", "onsheet", "nbr", "real", "P", "N")


class Pool(RegionPool):
    def __init__(self, stores, region: int = 320, **kw):
        super().__init__(stores, "fine", region=region, **kw)

    def load(self, si: int, r0: int, c0: int) -> dict:
        d = super().load(si, r0, c0)                                 # fine band as "ct" + maps
        s = self.stores[si]
        d["ctc"] = s.read_region(r0, c0, self.R, which=("coarse",))["coarse"]
        d["P"] = np.moveaxis(s.cells("P", r0, c0, self.R, fill=0.0), -1, 0).astype(np.float32)
        d["N"] = np.moveaxis(s.cells("N", r0, c0, self.R, fill=0.0), -1, 0).astype(np.float32)
        # official segments: their tier 2 marks ink-labelled cells; SurfNet uses no ink information -> uniform sigma 1.0
        d["seg"] = "labelled_cells" in s.meta
        return d

    @staticmethod
    def _pin(d: dict) -> dict:
        out = dict(d)
        for k in KEYS:
            if k in d and isinstance(d[k], np.ndarray):
                t = torch.from_numpy(np.ascontiguousarray(d[k]))
                out[k] = t.pin_memory() if torch.cuda.is_available() else t
        return out

    def _to_device(self, d: dict) -> dict:
        out = dict(d)
        for k in KEYS:
            if k in d and torch.is_tensor(d[k]):
                out[k] = d[k].to(self.device, non_blocking=True)
        return out
