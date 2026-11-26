"""Band builder: samples the CT along every object's normals into its BandStore (`prep/store.py`).

Per tile (T x T cells):
- fine band: positions and normals bilinear at the fine pixels (3 per cell, cell r at pixel 3r + 1), depths
  -fine_half..fine_half at the processing level, trilinear;
- coarse band: at the cells, depths -coarse_half..coarse_half (step 2) sampled from level + 1 (anti-aliased by the
  official pyramid), trilinear.

Scheduling (one volume at a time, all its objects together): tiles are sorted by height (z) and grouped. A
background fetcher downloads the chunks of the upcoming groups in large batches (`fetch_batch` chunks per s5cmd run,
so the transfer is bandwidth-bound, not latency-bound) and runs ahead of the GPU sampler up to `budget` resident
chunks. Every chunk's first and last group are known from planning: it is fetched once, before its first group, and
dropped from the GPU cache and (transient sources) deleted from disk right after its last group, so the disk holds
only a moving z-window of the volume."""
from __future__ import annotations

import threading
import time
from collections import defaultdict

import numpy as np
import torch

from ..common.geometry import as_t, bilinear, valid_at
from ..common.log import get_logger
from .store import BandStore
from ..volume import ChunkSource, GPUChunkCache, to_coarser


class TileGeometry:
    """Geometry of one object on the device + the sample points of any tile."""

    def __init__(self, store: BandStore, device):
        g = store.geom
        self.store = store
        self.spec = store.spec
        self.device = device
        self.P = as_t(g["P"], device, torch.float64)
        self.N = as_t(g["N"], device, torch.float64)
        self.V = as_t(g["V"], device, torch.bool)
        self.H, self.W = g["V"].shape
        self.fd = as_t(self.spec.fine_depths(), device, torch.float64)
        self.cd = as_t(self.spec.coarse_depths(), device, torch.float64)

    def fine_points(self, ty: int, tx: int, stride: int = 1):
        T, p = self.spec.tile, self.spec.ppc
        i = torch.arange(p * T * ty, p * T * (ty + 1), stride, device=self.device, dtype=torch.float64)
        j = torch.arange(p * T * tx, p * T * (tx + 1), stride, device=self.device, dtype=torch.float64)
        R, C = torch.meshgrid((i - 1) / p, (j - 1) / p, indexing="ij")
        ok = valid_at(self.V, R, C)
        pos = bilinear(self.P, R, C)
        nrm = bilinear(self.N, R, C)
        nrm = nrm / torch.linalg.vector_norm(nrm, dim=-1, keepdim=True).clamp(min=1e-9)
        d = self.fd[::stride] if stride > 1 else self.fd
        pts = pos[None] + d[:, None, None, None] * nrm[None]                  # [D, h, w, 3] xyz
        return pts.flip(-1), ok                                                # zyx (processing level)

    def coarse_points(self, ty: int, tx: int, stride: int = 1):
        T = self.spec.tile
        r = torch.arange(T * ty, T * (ty + 1), stride, device=self.device)
        c = torch.arange(T * tx, T * (tx + 1), stride, device=self.device)
        R, C = torch.meshgrid(r, c, indexing="ij")
        inside = (R < self.H) & (C < self.W)
        Rc, Cc = R.clamp(max=self.H - 1), C.clamp(max=self.W - 1)
        ok = inside & self.V[Rc, Cc]
        pos, nrm = self.P[Rc, Cc], self.N[Rc, Cc]
        d = self.cd[::stride] if stride > 1 else self.cd
        pts = pos[None] + d[:, None, None, None] * nrm[None]
        return to_coarser(pts.flip(-1)), ok                                    # zyx at level + 1

    def tile_z(self, ty: int, tx: int) -> float:
        T = self.spec.tile
        v = self.V[T * ty:T * (ty + 1), T * tx:T * (tx + 1)]
        if not bool(v.any()):
            return float("inf")
        return float(self.P[T * ty:T * (ty + 1), T * tx:T * (tx + 1), 2][v].mean())


class BandBuilder:
    def __init__(self, source: ChunkSource, level: int, device="cuda", fine_capacity: int = 2048,
                 coarse_capacity: int = 768, group: int = 16, log=None, fetch_batch: int = 2048,
                 budget: int = 12000):
        self.src = source
        self.level = level
        self.device = torch.device(device)
        self.fine_cache = GPUChunkCache(source, level, device, fine_capacity)
        self.coarse_cache = GPUChunkCache(source, level + 1, device, coarse_capacity)
        self.group = group
        self.fetch_batch = fetch_batch
        self.budget = budget
        self.log = log or get_logger("bands")

    @staticmethod
    def _plan_keys(cache: GPUChunkCache, pts: torch.Tensor, ok: torch.Tensor, margin: float) -> np.ndarray:
        if not bool(ok.any()):
            return np.zeros((0, 3), np.int64)
        q = pts[:, ok]                                                         # [D, n, 3]
        return cache.keys_for(q, margin=margin)

    def build(self, stores: list[BandStore]) -> dict:
        t0 = time.time()
        geoms = [TileGeometry(s, self.device) for s in stores]
        tiles = []
        for si, (s, g) in enumerate(zip(stores, geoms)):
            ok = s.tile_ok()
            for ty in range(s.nty):
                for tx in range(s.ntx):
                    if ok[ty, tx] and not s.done[ty, tx]:
                        tiles.append((g.tile_z(ty, tx), si, ty, tx))
        tiles.sort()
        if not tiles:
            self.log.info("bands: nothing to build")
            return {"tiles": 0}
        groups = [tiles[i:i + self.group] for i in range(0, len(tiles), self.group)]
        # planning: chunk keys per group (subsampled points + margin), last use of every key
        fk, ck = [], []
        last_f: dict = {}
        last_c: dict = {}
        for gi, grp in enumerate(groups):
            kf, kc = [], []
            for _, si, ty, tx in grp:
                g = geoms[si]
                pf, okf = g.fine_points(ty, tx, stride=4)
                kf.append(self._plan_keys(self.fine_cache, pf, okf, margin=9.0))
                pc, okc = g.coarse_points(ty, tx, stride=2)
                kc.append(self._plan_keys(self.coarse_cache, pc, okc, margin=5.0))
            kf = np.unique(np.concatenate(kf), axis=0) if kf else np.zeros((0, 3), np.int64)
            kc = np.unique(np.concatenate(kc), axis=0) if kc else np.zeros((0, 3), np.int64)
            fk.append(kf)
            ck.append(kc)
            for k in map(tuple, kf):
                last_f[k] = gi
            for k in map(tuple, kc):
                last_c[k] = gi
        release_f, release_c = defaultdict(list), defaultdict(list)
        for k, gi in last_f.items():
            release_f[gi].append(k)
        for k, gi in last_c.items():
            release_c[gi].append(k)
        self.log.info(f"bands: {len(tiles)} tiles in {len(groups)} groups; {len(last_f)} fine chunks (level "
                      f"{self.level}), {len(last_c)} coarse chunks (level {self.level + 1}); planned in "
                      f"{time.time() - t0:.0f} s")

        first_f, first_c = {}, {}
        for gi in range(len(groups)):
            for k in map(tuple, fk[gi]):
                first_f.setdefault(k, gi)
            for k in map(tuple, ck[gi]):
                first_c.setdefault(k, gi)
        new_f, new_c = defaultdict(list), defaultdict(list)
        for k, gi in first_f.items():
            new_f[gi].append(k)
        for k, gi in first_c.items():
            new_c[gi].append(k)
        cond = threading.Condition()
        state = {"ready": 0, "resident": 0, "cur": 0, "err": None}

        def fetcher():
            gi = 0
            while gi < len(groups):
                bf, bc, g1 = [], [], gi
                while g1 < len(groups) and (len(bf) + len(bc) < self.fetch_batch or g1 == gi):
                    bf += new_f[g1]
                    bc += new_c[g1]
                    g1 += 1
                with cond:                                   # stay within the disk budget unless the sampler starves
                    while state["resident"] > self.budget and state["cur"] < state["ready"]:
                        cond.wait(timeout=1.0)
                try:
                    if bf:
                        self.src.fetch(self.level, np.asarray(bf, np.int64))
                    if bc:
                        self.src.fetch(self.level + 1, np.asarray(bc, np.int64))
                except Exception as e:                       # surfaced in the main loop
                    with cond:
                        state["err"] = e
                        cond.notify_all()
                    return
                with cond:
                    state["resident"] += len(bf) + len(bc)
                    state["ready"] = g1
                    cond.notify_all()
                gi = g1

        th = threading.Thread(target=fetcher, daemon=True)
        th.start()
        n_done = 0
        t_wait = 0.0
        for gi, grp in enumerate(groups):
            tw = time.time()
            with cond:
                state["cur"] = gi
                cond.notify_all()
                while state["ready"] <= gi and state["err"] is None:
                    cond.wait(timeout=1.0)
                if state["err"] is not None:
                    raise state["err"]
            t_wait += time.time() - tw
            self.fine_cache.ensure(fk[gi])
            self.coarse_cache.ensure(ck[gi])
            for _, si, ty, tx in grp:
                g = geoms[si]
                pf, okf = g.fine_points(ty, tx)
                vf = self.fine_cache.sample(pf)
                vf = torch.where(okf[None], vf, torch.zeros_like(vf))
                pc, okc = g.coarse_points(ty, tx)
                vc = self.coarse_cache.sample(pc)
                vc = torch.where(okc[None], vc, torch.zeros_like(vc))
                stores[si].write_tile(ty, tx, vf.round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
                                      vc.round().clamp(0, 255).to(torch.uint8).cpu().numpy())
                n_done += 1
            for s in {stores[si] for _, si, _, _ in grp}:
                s.flush()
            rf = np.asarray(release_f.get(gi, []), np.int64).reshape(-1, 3)
            rc = np.asarray(release_c.get(gi, []), np.int64).reshape(-1, 3)
            self.fine_cache.drop(rf)
            self.coarse_cache.drop(rc)
            self.src.release(self.level, rf)
            self.src.release(self.level + 1, rc)
            with cond:
                state["resident"] -= len(rf) + len(rc)
                cond.notify_all()
            if (gi + 1) % 100 == 0 or gi + 1 == len(groups):
                el = time.time() - t0
                self.log.info(f"bands: {n_done}/{len(tiles)} tiles, {el:.0f} s, {n_done / max(el, 1e-6):.1f} tiles/s, "
                              f"waiting for downloads {100 * t_wait / max(el, 1e-6):.0f} %, resident chunks "
                              f"{state['resident']}")
        th.join(timeout=5)
        return {"tiles": n_done, "seconds": time.time() - t0, "fine_chunks": len(last_f), "coarse_chunks": len(last_c)}
