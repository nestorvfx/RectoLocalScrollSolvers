"""Score any surface against a ground-truth surface, whatever its grid (a tracer's output, a corrected sheet).

For every ground-truth cell in the scored area: the nearest point of the other surface (densified `up` x per grid
axis, points <= 1.5 vox apart), split into the component along the ground-truth normal (the error, signed: + toward
the normal) and the lateral component. A cell is covered when the lateral component is below `lateral_max` (vox): the
other surface passes over it. Then, over the scored area:
  coverage      share of cells covered;
  p50 / p90     |error| of the covered cells (vox);
  within_1vox   share of all scored cells covered and within 1 vox;
  score         the mean credit over all scored cells (`credit`): 1 within FULL_VOX of the truth, falling linearly to 0
                at half the winding spacing, 0 beyond it and where not covered (so a method cannot score well by
                covering only the easy part);
  wrong_sheet   share of all scored cells not covered or at least half the winding spacing from the truth: closer to a
                neighbouring sheet than to the right one.

Speed: one k-d tree per surface, reused for every number; the other surface is cropped to the scored window (`box`)
before it is densified, and its points are merged within 0.25 vox (a tracer's collapsed cells would otherwise stack
identical points, where k-d tree queries degrade to brute force)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _bilinear(A: np.ndarray, up: int) -> np.ndarray:
    """A [H, W, ...] upsampled `up` x per axis with align-corners bilinear interpolation -> [(H-1)up+1, (W-1)up+1, ...]."""
    H, W = A.shape[:2]

    def axis(n):
        t = np.arange((n - 1) * up + 1) / up
        i0 = np.minimum(t.astype(int), max(n - 2, 0))
        return i0, np.minimum(i0 + 1, n - 1), t - i0
    r0, r1, fr = axis(H)
    c0, c1, fc = axis(W)
    sh = (-1, 1) + (1,) * (A.ndim - 2)
    sw = (1, -1) + (1,) * (A.ndim - 2)
    fr, fc = fr.reshape(sh), fc.reshape(sw)
    return ((1 - fr) * (1 - fc) * A[r0][:, c0] + fr * (1 - fc) * A[r1][:, c0]
            + (1 - fr) * fc * A[r0][:, c1] + fr * fc * A[r1][:, c1])


def densify(Q: np.ndarray, VQ: np.ndarray, up: int | None = None, box=None, cells: bool = False):
    """Points of a grid surface, bilinearly upsampled `up` x along both axes, valid quads only. -> [n, 3] (`cells`: also
    [n, 2] the grid cell each point is nearest to). `up` None: enough for points <= 1.5 vox apart (a 20-vox tracer grid
    gets 14 x). `box` (lo, hi) xyz: only the cells inside it are used."""
    r0 = c0 = 0
    if box is not None:
        VQ = VQ & np.all((Q >= box[0]) & (Q <= box[1]), -1)
        if VQ.any():
            rr, cc = np.nonzero(VQ)
            r0, c0 = rr.min(), cc.min()
            Q, VQ = Q[r0:rr.max() + 1, c0:cc.max() + 1], VQ[r0:rr.max() + 1, c0:cc.max() + 1]
    if not VQ.any():
        return (np.zeros((0, 3)), np.zeros((0, 2), int)) if cells else np.zeros((0, 3))
    if up is None:
        from ..common.geometry import edge_steps
        st = [s for s in edge_steps(Q, VQ) if np.isfinite(s)]
        up = max(2, int(np.ceil(max(st) / 1.5))) if st else 2
    w = _bilinear(VQ.astype(np.float64), up)
    vu = w > 0.999                                                      # every corner of the quad valid
    q = _bilinear(np.where(VQ[..., None], Q, 0.0).astype(np.float64), up)
    pts = q[vu] / w[vu][:, None]
    ii, jj = np.nonzero(vu)
    # one point per 0.25-vox voxel: collapsed quads (a tracer's degenerate cells) would otherwise put up^2 identical
    # points in one place, and k-d tree queries near such clusters degrade to brute force
    key = np.floor((pts - pts.min(0)) * 4).astype(np.int64)
    key = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    keep = np.sort(np.unique(key, return_index=True)[1])
    pts, ii, jj = pts[keep], ii[keep], jj[keep]
    if not cells:
        return pts
    return pts, np.stack([np.rint(ii / up).astype(int) + r0, np.rint(jj / up).astype(int) + c0], 1)


def surface_errors(P: np.ndarray, N: np.ndarray, mask: np.ndarray, Q: np.ndarray, VQ: np.ndarray, up: int | None = None,
                   lateral_max: float = 3.0, box_margin: float | None = 100.0, pts: np.ndarray | None = None,
                   tree: cKDTree | None = None, workers: int = 1, k: int = 8) -> dict:
    """P, N [H, W, 3] ground truth and its unit normals; mask [H, W] the cells to score; Q, VQ the other surface (any
    grid, same voxel frame). -> per-cell arrays (NaN outside the mask): signed normal error, lateral distance, covered,
    and the distance to the nearest point; plus the other surface's points and tree (reused by `two_way`).
    `box_margin`: the other surface is cropped to the scored cells' box widened by this (vox; None: not cropped)."""
    H, W = mask.shape
    err, lat, dist = np.full((H, W), np.nan), np.full((H, W), np.nan), np.full((H, W), np.nan)
    cov = np.zeros((H, W), bool)
    if pts is None:
        box = None if box_margin is None or not mask.any() else \
            (P[mask].min(0) - box_margin, P[mask].max(0) + box_margin)
        pts = densify(Q, VQ, up, box)
    out = {"error": err, "lateral": lat, "covered": cov, "distance": dist, "pts": pts, "tree": None}
    if len(pts) == 0 or not mask.any():
        return out
    tree = tree or cKDTree(pts)
    out["tree"] = tree
    g, n = P[mask], N[mask]
    kk = min(k, len(pts))
    d, idx = tree.query(g, k=kk, workers=workers)                       # several candidates: the nearest along the
    d, idx = d.reshape(len(g), kk), idx.reshape(len(g), kk)             # normal (least lateral) wins
    v = pts[idx] - g[:, None]
    e = (v * n[:, None]).sum(-1)
    lt = np.linalg.norm(v - e[..., None] * n[:, None], axis=-1)
    j = np.argmin(lt, 1)
    rows = np.arange(len(g))
    err[mask], lat[mask], dist[mask] = e[rows, j], lt[rows, j], d[:, 0]
    cov[mask] = lt[rows, j] <= lateral_max
    return out


FULL_VOX = 2.0                   # full credit within this distance (vox) of the truth: its own registration error is ~1 vox


def credit(error: np.ndarray, covered: np.ndarray, spacing: float, full: float = FULL_VOX) -> np.ndarray:
    """Per cell: 1 within `full` vox of the truth, falling linearly to 0 at half the winding spacing (as close to a
    neighbouring sheet as to the right one); 0 beyond it and where not covered."""
    half = spacing / 2.0
    a = np.where(covered, np.abs(np.nan_to_num(error, nan=np.inf)), np.inf)
    return np.clip((half - a) / (half - full), 0.0, 1.0)


def summary(res: dict, mask: np.ndarray, spacing: float = 13.0, full: float = FULL_VOX) -> dict:
    e, c = res["error"][mask], res["covered"][mask]
    ae = np.abs(e[c])
    n = max(int(mask.sum()), 1)
    return {"cells": int(mask.sum()), "coverage": round(float(c.mean()), 4) if c.size else 0.0,
            "p50": round(float(np.median(ae)), 3) if ae.size else None,
            "p90": round(float(np.quantile(ae, 0.9)), 3) if ae.size else None,
            "within_1vox": round(float((ae < 1).sum()) / n, 4),
            "score": round(float(credit(e, c, spacing, full).sum()) / n, 4),
            "wrong_sheet": round(1.0 - float((ae < spacing / 2.0).sum()) / n, 4)}


class Reference:
    """A ground-truth surface prepared once for every surface scored against it, in two k-d trees: its grid cells
    (coarse) and its points <= 1.5 vox apart (fine), each point with its grid cell. A nearest-point query far from a
    dense sheet must examine every point within that distance, so far queries use the coarse tree and only near ones
    (< `refine` vox) the fine one."""

    def __init__(self, P: np.ndarray, V: np.ndarray, up: int | None = None, refine: float = 15.0):
        self.pts, self.cells = densify(P, V, up, cells=True)
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        self.coarse_cells = np.argwhere(V)
        self.coarse = cKDTree(P[V]) if V.any() else None
        self.refine = refine

    def nearest(self, q: np.ndarray, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """q [n, 3] -> distance to the surface [n] and the grid cell of the nearest point [n, 2]."""
        d, j = self.coarse.query(q, workers=workers)
        cell = self.coarse_cells[j]
        near = d < self.refine
        if near.any():
            d2, j2 = self.tree.query(q[near], workers=workers)
            d[near], cell[near] = d2, self.cells[j2]
        return d, cell


def two_way(P: np.ndarray, V: np.ndarray, mask: np.ndarray, Q: np.ndarray, VQ: np.ndarray, up: int | None = None,
            near: float = 10.0, res: dict | None = None, ref: Reference | None = None, workers: int = 1,
            max_points: int = 50_000) -> dict:
    """The tracer-comparison convention (`vc_grow_seg_from_segments --sweep`): Euclidean distances ground truth ->
    method over the scored cells, and method -> ground truth over the method's points that lie over the scored area
    (their nearest ground-truth point belongs to a scored cell; at most `max_points` of them, evenly spaced). Mean, p95
    and the share within `near` vox, each way. `res` (from `surface_errors` on a mask covering `mask`) and `ref` are
    reused when given."""
    if res is None:
        res = surface_errors(P, np.zeros_like(P), mask, Q, VQ, up, workers=workers)
    ref = ref or Reference(P, V, up)
    out = {"gt_to_method": res["distance"][mask], "method_to_gt": np.zeros(0)}
    q = res["pts"]
    if len(q) and ref.tree is not None:
        q = q[::max(1, int(np.ceil(len(q) / max_points)))]
        d, cell = ref.nearest(q, workers)
        r, c = cell.T
        out["method_to_gt"] = d[mask[np.clip(r, 0, mask.shape[0] - 1), np.clip(c, 0, mask.shape[1] - 1)]]
    f = lambda d: None if d is None or len(d) == 0 or not np.isfinite(d).any() else {              # noqa: E731
        "mean": round(float(np.nanmean(d)), 3), "p95": round(float(np.nanquantile(d, 0.95)), 3),
        f"within_{near:g}vox": round(float(np.nanmean(d < near)), 4), "points": int(np.isfinite(d).sum())}
    return {k: f(v) for k, v in out.items()}
