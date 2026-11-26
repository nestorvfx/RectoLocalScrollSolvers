"""Continuous angle about the scroll axis over a multi-turn surface (PHerc1667's merged official unwrap).

Copied in spirit from our own surface_extraction_main/code/validation/merged_gt.py (never edited there): each cell's
principal angle is unwrapped along each contiguous run of a grid row (np.unwrap is safe only between neighbouring
columns), then every run gets the 2 pi multiple that agrees with its vertically adjacent runs, by a maximum spanning
tree (Prim; edge weight = shared columns) from the longest run. Turn k of a cell = the cells at angle Theta + 2 pi k.
The sense of the spiral (does the radius grow with Theta?) decides which side is 'outward'."""
from __future__ import annotations

import heapq
import math

import numpy as np

from ..common.geometry import Axis

GAP = 2


def unwrap(P: np.ndarray, V: np.ndarray, axis: Axis) -> np.ndarray:
    """Theta [H, W] (NaN invalid), continuous over the surface."""
    H, W = V.shape
    theta = np.full((H, W), np.nan)
    theta[V] = axis.angle(P[V])
    Theta = np.full((H, W), np.nan)
    seg = np.full((H, W), -1, np.int64)
    segs = []
    for r in range(H):
        c = np.flatnonzero(V[r])
        if not len(c):
            continue
        for part in np.split(c, np.flatnonzero(np.diff(c) > GAP) + 1):
            Theta[r, part] = np.unwrap(theta[r, part])
            seg[r, part] = len(segs)
            segs.append((r, int(part[0]), int(part[-1]) + 1))
    ns = len(segs)
    adj = [[] for _ in range(ns)]
    for r in range(1, H):
        a, b = seg[r - 1], seg[r]
        cols = np.flatnonzero((a >= 0) & (b >= 0))
        if not len(cols):
            continue
        key = a[cols] * ns + b[cols]
        dl = Theta[r - 1, cols] - Theta[r, cols]
        o = np.argsort(key, kind="stable")
        key, dl = key[o], dl[o]
        uk, st, cn = np.unique(key, return_index=True, return_counts=True)
        for k_, s_, n_ in zip(uk.tolist(), st.tolist(), cn.tolist()):
            mult = int(round(float(np.median(dl[s_:s_ + n_])) / (2 * math.pi)))
            sa, sb = divmod(k_, ns)
            adj[sa].append((n_, sb, mult))
            adj[sb].append((n_, sa, -mult))
    off = np.full(ns, np.nan)
    lengths = np.array([s[2] - s[1] for s in segs])
    for root in np.argsort(-lengths):
        if not np.isnan(off[root]):
            continue
        off[root] = 0
        heap = [(-w_, nb, m_) for w_, nb, m_ in adj[root]]
        heapq.heapify(heap)
        while heap:
            w_, nb, val = heapq.heappop(heap)
            if not np.isnan(off[nb]):
                continue
            off[nb] = val
            for w2, nb2, m2 in adj[nb]:
                if np.isnan(off[nb2]):
                    heapq.heappush(heap, (-w2, nb2, val + m2))
    return np.where(seg >= 0, Theta + 2 * math.pi * off[np.maximum(seg, 0)], np.nan)


def outward_sign(P: np.ndarray, V: np.ndarray, Theta: np.ndarray, axis: Axis) -> int:
    """+1 if the radius grows with Theta (Theta + 2 pi is the next turn outward), else -1."""
    ay, ax = axis.at(P[V][:, 2])
    r = np.hypot(P[V][:, 0] - ax, P[V][:, 1] - ay)
    t = Theta[V]
    ok = np.isfinite(t)
    return 1 if np.corrcoef(t[ok], r[ok])[0, 1] >= 0 else -1
