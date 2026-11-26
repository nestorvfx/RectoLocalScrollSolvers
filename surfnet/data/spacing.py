"""Winding spacing of an object (voxels): the median |offset| to the windings +-1 where the multi-winding GT gives
them. Objects without neighbours (single segments) get the median spacing of the GT objects (`gt_median`): CT
intensity alone does not give it reliably (on the built bands: the band autocorrelation reads ~20 vox and
layer-peak counting ~11 vox on 0139 windings whose GT spacing is 12.6-20.6 vox). The synthetic winding jumps vary
the spacing by +-20 % per sample (`data/errors.py`). `from_band` is kept for reference only."""
from __future__ import annotations

import numpy as np

from .store import BandStore


def from_neighbours(store: BandStore) -> float | None:
    g = store.geom
    if "nbr" not in g:
        return None
    ks = store.meta.get("winding_index_offsets", [])
    vals = []
    for k in (-1, 1):
        if k in ks:
            v = g["nbr"][ks.index(k)].astype(np.float32)
            v = np.abs(v[np.isfinite(v)])
            if v.size:
                vals.append(v)
    if not vals:
        return None
    return float(np.median(np.concatenate(vals)))


def from_band(store: BandStore, n: int = 4000, seed: int = 0) -> float | None:
    rng = np.random.default_rng(seed)
    T = store.spec.tile
    tiles = np.argwhere(store.done)
    if len(tiles) == 0:
        return None
    prof = []
    for _ in range(n):
        ty, tx = tiles[rng.integers(len(tiles))]
        r, c = rng.integers(T), rng.integers(T)
        p = store.coarse[ty, tx, :, r, c].astype(np.float32)
        if (p > 0).mean() < 0.9:
            continue
        k = np.ones(15) / 15
        p = p - np.convolve(np.pad(p, 7, mode="edge"), k, "valid")
        p = (p - p.mean()) / max(p.std(), 1e-6)
        prof.append(p)
    if len(prof) < 50:
        return None
    P = np.stack(prof)
    D = P.shape[1]
    ac = np.array([(P[:, :D - L] * P[:, L:]).mean() for L in range(D // 2)])
    step = store.spec.coarse_step
    lo, hi = int(8 / step), int(50 / step)
    seg = ac[lo:hi]
    if len(seg) < 3:
        return None
    i = int(np.argmax(seg)) + lo
    return float(i * step)


def estimate(store: BandStore) -> float | None:
    return from_neighbours(store) or from_band(store)


def gt_median(stores: list[BandStore]) -> float | None:
    vals = [s.meta["spacing_vox"] for s in stores if "nbr" in s.geom and s.meta.get("spacing_vox")]
    return float(np.median(vals)) if vals else None
