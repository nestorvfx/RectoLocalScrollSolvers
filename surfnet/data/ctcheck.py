"""`surfnet ctcheck`: cells without CT are excluded.

A mesh can extend beyond the scan it is registered to (field of view, scroll mask): PHerc1667's merged unwrap was
traced on another scan, and ~19 % of its cells read zero CT on the 2.4 um scan. Such cells carry no information and
would teach the network to predict surfaces in empty space. Per cell, the fine band at depths -8..8 vox (the cell's
centre pixel) is read; a cell with more than `max_zero` zeros there gets split EXCLUDED (never an input, a target
or a validation cell). Idempotent; the counts go to meta `no_ct_cells`."""
from __future__ import annotations

import numpy as np

from .store import SPLIT_EXCLUDED, BandStore


def no_ct_cells(store: BandStore, half: int = 8, max_zero: float = 0.5) -> np.ndarray:
    T = store.spec.tile
    p = store.spec.ppc
    h0 = store.spec.fine_half
    bad = np.zeros((store.nty * T, store.ntx * T), bool)
    for ty in range(store.nty):
        for tx in range(store.ntx):
            if not store.done[ty, tx]:
                continue
            f = store.fine[ty, tx, h0 - half:h0 + half + 1, p // 2::p, p // 2::p]          # [17, T, T] cell centres
            bad[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] = (f == 0).mean(0) > max_zero
    return bad[:store.H, :store.W]


def run(stores: list[BandStore], log, max_zero: float = 0.5) -> dict:
    out = {}
    for s in stores:
        g = s.geom
        bad = no_ct_cells(s, max_zero=max_zero) & g["V"]
        split = g["split"].copy()
        new = bad & (split != SPLIT_EXCLUDED)
        split[bad] = SPLIT_EXCLUDED
        if new.any():
            s.update_geom(split=split)
        s.update_meta(no_ct_cells=int(bad.sum()))
        frac = float(bad.sum()) / max(int(g["V"].sum()), 1)
        out[s.meta["id"]] = frac
        log.info(f"ctcheck {s.meta['id']}: {int(bad.sum()):,} cells without CT ({100 * frac:.1f} %) excluded")
    return out
