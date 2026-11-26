"""Exact surface scoring on the GPU (torch; runs on the CPU too): a surface of any grid against a ground-truth grid.

Both surfaces are grids of quads (two triangles each), and they are compared exactly, without densifying either:
- **error**: for every scored ground-truth cell, the exact closest point of the other surface (point-triangle), split
  into its component along the ground-truth normal (the signed error) and the lateral component. The cell is covered
  when the lateral component is at most `lateral_max` vox: the other surface passes over it (the definition of
  `score.surface_errors`, without its densification error).
- **two-way distances** (`vc_grow_seg_from_segments --sweep`): the exact Euclidean distance from each scored cell to
  the other surface, and from the other surface's nodes over the scored area (their nearest ground-truth cell is a
  scored one) to the ground truth.

Search: one batched pass over all cell x node pairs (matrix products; its cost does not depend on how far apart the
surfaces are) finds each point's nearest node; the exact closest point is then found among the 32 triangles of the
4 x 4 quads around it.
The coarse pass is float32 (relative to the case's centre), the exact tests float64."""
from __future__ import annotations

import numpy as np
import torch

RING = (-2, -1, 0, 1)                                 # quad offsets around a node: a 4 x 4 block of quads


class Grid:
    """A grid surface on `device`: positions [H, W, 3] (float64), validity, its valid nodes (and their row / col),
    the validity of its quads, and the reach for the coarse search."""

    def __init__(self, P: np.ndarray, V: np.ndarray, device, box=None, origin=None):
        V = V.copy()
        if box is not None:
            V &= np.all((P >= box[0]) & (P <= box[1]), -1)
        self.dev = device
        self.P = torch.as_tensor(np.where(V[..., None], P, 0.0), device=device, dtype=torch.float64)
        self.V = torch.as_tensor(V, device=device)
        self.rc = torch.nonzero(self.V)                                          # [n, 2]
        self.nodes = self.P[self.V]                                              # [n, 3]
        self.origin = origin if origin is not None else (self.nodes.mean(0) if len(self.nodes) else
                                                         torch.zeros(3, device=device, dtype=torch.float64))
        self.n32 = (self.nodes - self.origin).float()
        Vq = self.V[:-1, :-1] & self.V[1:, :-1] & self.V[:-1, 1:] & self.V[1:, 1:]
        self.Vq = Vq
        if len(self.nodes) > 1:
            er = (self.P[1:] - self.P[:-1]).norm(dim=-1)[self.V[1:] & self.V[:-1]]
            ec = (self.P[:, 1:] - self.P[:, :-1]).norm(dim=-1)[self.V[:, 1:] & self.V[:, :-1]]
            e = torch.cat([er, ec])
            self.reach = max(3.0, 0.75 * float(torch.quantile(e.float(), 0.99))) if len(e) else 3.0
        else:
            self.reach = 3.0

    def triangles(self, j: torch.Tensor):
        """The triangles of the 4 x 4 quads around nodes j [m] -> corners [m, 32, 3, 3] (float64), valid [m, 32]."""
        H, W = self.V.shape
        r, c = self.rc[j, 0], self.rc[j, 1]
        off = torch.tensor([(a, b) for a in RING for b in RING], device=self.dev)               # [16, 2]
        qr, qc = r[:, None] + off[:, 0], c[:, None] + off[:, 1]                                  # quads' top-left
        ok = (qr >= 0) & (qc >= 0) & (qr < H - 1) & (qc < W - 1)
        qr, qc = qr.clamp(0, max(H - 2, 0)), qc.clamp(0, max(W - 2, 0))
        ok &= self.Vq[qr, qc] if H > 1 and W > 1 else torch.zeros_like(ok)
        p00, p10 = self.P[qr, qc], self.P[(qr + 1).clamp(max=H - 1), qc]
        p01, p11 = self.P[qr, (qc + 1).clamp(max=W - 1)], self.P[(qr + 1).clamp(max=H - 1), (qc + 1).clamp(max=W - 1)]
        tri = torch.stack([torch.stack([p00, p10, p11], -2), torch.stack([p00, p11, p01], -2)], 2)  # [m, 16, 2, 3, 3]
        return tri.reshape(len(j), 32, 3, 3), ok[:, :, None].expand(-1, -1, 2).reshape(len(j), 32)


def _nearest_node(g32, nodes32, budget: float = 1.2e8):
    """The nearest node of each query point, batched over all pairs (squared distances by matrix products) -> [m]."""
    m, n = len(g32), len(nodes32)
    jn = torch.zeros((m,), device=g32.device, dtype=torch.long)
    if n == 0 or m == 0:
        return jn
    q2 = (nodes32 * nodes32).sum(-1)                                          # [n]
    step = max(1, int(budget // n))
    for i in range(0, m, step):
        g = g32[i:i + step]
        jn[i:i + step] = (q2[None] - 2.0 * (g @ nodes32.T)).argmin(1)          # |g|^2 is constant per row
    return jn


def _seg_dist(p, a, b):
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-18)).clamp(0, 1)
    return (p - (a + t[..., None] * ab)).norm(dim=-1)


def _seg_point(p, a, b):
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-18)).clamp(0, 1)
    return a + t[..., None] * ab


def _tri_closest(p, tri, ok):
    """The exact closest point to p [m, 3] on triangles [m, T, 3, 3] -> (point [m, 3], distance [m]; inf if none)."""
    a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
    pp = p[:, None].expand_as(a)
    ab, ac = b - a, c - a
    nrm = torch.cross(ab, ac, dim=-1)
    nn = nrm.norm(dim=-1).clamp(min=1e-18)
    h = ((pp - a) * nrm).sum(-1) / nn
    proj = pp - h[..., None] * nrm / nn[..., None]
    v2 = proj - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (v2 * ab).sum(-1), (v2 * ac).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp(min=1e-18)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (nn > 1e-9) & (v >= 0) & (w >= 0) & (v + w <= 1)
    cands = torch.stack([_seg_point(pp, a, b), _seg_point(pp, b, c), _seg_point(pp, c, a)], -2)   # [m, T, 3, 3]
    dc = (cands - pp[..., None, :]).norm(dim=-1)
    k = dc.argmin(-1)
    edge = cands.gather(-2, k[..., None, None].expand(*k.shape, 1, 3))[..., 0, :]
    pt = torch.where(inside[..., None], proj, edge)
    d = (pt - pp).norm(dim=-1)
    d = torch.where(ok, d, torch.full_like(d, float("inf")))
    j = d.argmin(1)
    rows = torch.arange(len(p), device=p.device)
    return pt[rows, j], d[rows, j]


def _tri_dist(p, tri, ok):
    """Exact distance from points p [m, 3] to triangles [m, T, 3, 3] -> [m] (inf if no valid triangle)."""
    a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
    pp = p[:, None].expand_as(a)
    ab, ac = b - a, c - a
    nrm = torch.cross(ab, ac, dim=-1)
    nn = nrm.norm(dim=-1)
    h = ((pp - a) * nrm).sum(-1) / nn.clamp(min=1e-18)                       # signed distance to the plane
    proj = pp - h[..., None] * nrm / nn.clamp(min=1e-18)[..., None]
    v2 = proj - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (v2 * ab).sum(-1), (v2 * ac).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp(min=1e-18)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (nn > 1e-9) & (v >= 0) & (w >= 0) & (v + w <= 1)
    seg = torch.minimum(torch.minimum(_seg_dist(pp, a, b), _seg_dist(pp, b, c)), _seg_dist(pp, c, a))
    dist = torch.where(inside, h.abs(), seg)
    return torch.where(ok, dist, torch.full_like(dist, float("inf"))).min(1).values


def score_surface(P, N, V, mask, scored, Q, VQ, device="cuda", lateral_max: float = 3.0, box_margin: float = 100.0,
                  max_points: int = 50_000, chunk: int = 20_000) -> dict:
    """P, N [H, W, 3] ground truth and unit normals, V its valid cells; `mask` the cells whose error is computed (known
    + target), `scored` (within mask) the cells of the two-way distances; Q, VQ the other surface (any grid, the same
    voxel frame). -> {"error", "covered", "distance" [H, W] (NaN outside mask), "gt_to_method", "method_to_gt" (arrays)}."""
    H, W = mask.shape
    err, dist = np.full((H, W), np.nan), np.full((H, W), np.nan)
    cov = np.zeros((H, W), bool)
    out = {"error": err, "covered": cov, "distance": dist, "gt_to_method": np.zeros(0), "method_to_gt": np.zeros(0)}
    if not mask.any() or not VQ.any():
        return out
    box = (P[mask].min(0) - box_margin, P[mask].max(0) + box_margin)
    M = Grid(Q, VQ, device, box=box)
    if len(M.nodes) == 0:
        return out
    G = Grid(P, V, device, origin=M.origin)
    Pm = torch.as_tensor(P[mask], device=device, dtype=torch.float64)
    Nm = torch.as_tensor(N[mask], device=device, dtype=torch.float64)
    e_all, c_all, d_all = [], [], []
    for i in range(0, len(Pm), chunk):                                       # ground truth -> the other surface
        g, n = Pm[i:i + chunk], Nm[i:i + chunk]
        jn = _nearest_node((g - M.origin).float(), M.n32)
        tri, ok = M.triangles(jn)
        c, d = _tri_closest(g, tri, ok)
        v = c - g
        e = (v * n).sum(-1)
        lat = (v - e[:, None] * n).norm(dim=-1)
        covered = torch.isfinite(d) & (lat <= lateral_max)
        e_all.append(torch.where(torch.isfinite(d), e, torch.full_like(e, float("nan"))))
        c_all.append(covered)
        d_all.append(d)
    err[mask] = torch.cat(e_all).cpu().numpy()
    cov[mask] = torch.cat(c_all).cpu().numpy()
    dist[mask] = torch.cat(d_all).cpu().numpy()
    out["gt_to_method"] = dist[scored & mask]
    q = M.nodes[::max(1, int(np.ceil(len(M.nodes) / max_points)))]          # the other surface -> ground truth
    jg = _nearest_node((q - M.origin).float(), G.n32)
    tri, ok = G.triangles(jg)
    dq = _tri_dist(q, tri, ok)
    r, c = G.rc[jg, 0].cpu().numpy(), G.rc[jg, 1].cpu().numpy()
    out["method_to_gt"] = dq.cpu().numpy()[scored[r, c]]
    return out


def two_way_summary(res: dict, near: float = 10.0) -> dict:
    f = lambda d: None if d is None or len(d) == 0 or not np.isfinite(d).any() else {                  # noqa: E731
        "mean": round(float(np.nanmean(np.where(np.isfinite(d), d, np.nan))), 3),
        "p95": round(float(np.nanquantile(np.where(np.isfinite(d), d, np.nan), 0.95)), 3),
        f"within_{near:g}vox": round(float(np.mean(d < near)), 4), "points": int(np.isfinite(d).sum())}
    return {"gt_to_method": f(res["gt_to_method"]), "method_to_gt": f(res["method_to_gt"])}
