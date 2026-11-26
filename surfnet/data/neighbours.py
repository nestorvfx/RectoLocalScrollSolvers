"""Offsets along each cell's normal to the neighbouring windings of multi-winding ground truth (GPU).

For a cell p (normal n) of surface A and a neighbouring surface B (a grid Q(u, v)), the intersection of the ray
p + t n with B solves Q(u, v) - p - t n = 0 for (u, v, t): batched Gauss-Newton with the 3 x 3 Jacobian
[dQ/du, dQ/dv, -n] (bilinear Q and derivatives), started from a seed found by a raster lookup:
- separate per-winding meshes (PHerc0139): B = winding w + k, seeded by the cell of B with the nearest (theta, z)
  about the scroll axis;
- one multi-turn mesh (PHerc1667's merged unwrap): B = the same mesh, seeded at (Theta + 2 pi k, z) with the
  continuous angle Theta (`turns.py`).
Accepted where the residual is < `tol` vox and the bilinear stencil of (u, v) is valid. Result per object:
nbr [2K, H, W] float16, k = -K..-1, 1..K (k > 0 = outward), NaN where no intersection."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..common.geometry import as_t, valid_at


class Raster:
    """Lookup table from (angle, z) bins to the flat index of a cell of B (nearest-filled up to `fill` bins)."""

    def __init__(self, ang: torch.Tensor, z: torch.Tensor, valid: torch.Tensor, dang: float, dz: float, fill: int = 3):
        dev = ang.device
        a = ang[valid]
        zz = z[valid]
        idx = valid.flatten().nonzero()[:, 0]
        self.a0, self.z0 = float(a.min()), float(zz.min())
        self.da, self.dz = dang, dz
        na = int((float(a.max()) - self.a0) / dang) + 2
        nz = int((float(zz.max()) - self.z0) / dz) + 2
        ia = ((a - self.a0) / dang).round().long().clamp(0, na - 1)
        iz = ((zz - self.z0) / dz).round().long().clamp(0, nz - 1)
        tab = torch.full((nz * na,), -1, dtype=torch.long, device=dev)
        tab[iz * na + ia] = idx
        tab = tab.view(1, 1, nz, na)
        for _ in range(fill):                                   # fill empty bins from neighbours (index + 1 max-pool)
            grown = F.max_pool2d((tab + 1).float(), 3, stride=1, padding=1).long() - 1
            tab = torch.where(tab >= 0, tab, grown)
        self.tab = tab[0, 0]
        self.na, self.nz = na, nz

    def lookup(self, ang: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ia = ((ang - self.a0) / self.da).round().long()
        iz = ((z - self.z0) / self.dz).round().long()
        ok = (ia >= 0) & (ia < self.na) & (iz >= 0) & (iz < self.nz)
        out = self.tab[iz.clamp(0, self.nz - 1), ia.clamp(0, self.na - 1)]
        return torch.where(ok, out, torch.full_like(out, -1))


def _bil_and_grad(Q: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    H, W = Q.shape[:2]
    u = u.clamp(0, H - 1 - 1e-4)
    v = v.clamp(0, W - 1 - 1e-4)
    r0, c0 = u.floor().long().clamp(0, H - 2), v.floor().long().clamp(0, W - 2)
    fr, fc = (u - r0)[:, None], (v - c0)[:, None]
    q00, q10, q01, q11 = Q[r0, c0], Q[r0 + 1, c0], Q[r0, c0 + 1], Q[r0 + 1, c0 + 1]
    q = (1 - fr) * (1 - fc) * q00 + fr * (1 - fc) * q10 + (1 - fr) * fc * q01 + fr * fc * q11
    qu = (1 - fc) * (q10 - q00) + fc * (q11 - q01)
    qv = (1 - fr) * (q01 - q00) + fr * (q11 - q10)
    return q, qu, qv


def intersect(p: torch.Tensor, n: torch.Tensor, Q: torch.Tensor, VQ: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor,
              iters: int = 8, tol: float = 0.75, max_step: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays p + t n [N, 3] against grid surface Q [H, W, 3]; seeds (u0, v0) [N]. -> t [N], ok [N]."""
    u, v = u0.double(), v0.double()
    q, _, _ = _bil_and_grad(Q, u, v)
    t = ((q - p) * n).sum(-1)
    for _ in range(iters):
        q, qu, qv = _bil_and_grad(Q, u, v)
        r = q - p - t[:, None] * n
        J = torch.stack([qu, qv, -n], -1)                                   # [N, 3, 3]
        J = J + 1e-6 * torch.eye(3, dtype=J.dtype, device=J.device)
        d = torch.linalg.solve(J, -r[..., None])[..., 0]
        d[:, :2] = d[:, :2].clamp(-max_step, max_step)
        u, v, t = u + d[:, 0], v + d[:, 1], t + d[:, 2]
    q, _, _ = _bil_and_grad(Q, u, v)
    res = torch.linalg.vector_norm(q - p - t[:, None] * n, dim=-1)
    ok = (res < tol) & valid_at(VQ, u, v)
    return t, ok


def _surface(P, N, V, device):
    return as_t(P, device, torch.float64), as_t(N, device, torch.float64), as_t(V, device, torch.bool)


def windings(objs: dict[int, dict], axis, K: int = 3, device="cuda", dang_cells: float = 1.0, chunk: int = 1_000_000
             ) -> dict[int, np.ndarray]:
    """objs: winding number -> {P, N, V} (processing-level voxels; numbers grow outward). -> nbr per winding."""
    out = {}
    cache = {}

    def ang_z(w):
        if w not in cache:
            P, N, V = _surface(objs[w]["P"], objs[w]["N"], objs[w]["V"], device)
            Pn, Vn = objs[w]["P"], objs[w]["V"]
            ang = np.full(Vn.shape, np.nan)
            ang[Vn] = axis.angle(Pn[Vn])
            pv = Pn[Vn]
            ay, ax = axis.at(pv[:, 2])
            r_med = float(np.median(np.hypot(pv[:, 0] - ax, pv[:, 1] - ay)))
            cell = float(np.median(np.linalg.norm(Pn[1:] - Pn[:-1], axis=-1)[Vn[1:] & Vn[:-1]]))
            A = as_t(np.nan_to_num(ang), device, torch.float64)
            ras = Raster(A, P[..., 2], V, dang=dang_cells * cell / max(r_med, 1.0), dz=dang_cells * cell)
            cache[w] = (P, N, V, A, ras)
        return cache[w]

    for w in objs:
        P, N, V, A, _ = ang_z(w)
        H, W = V.shape
        nbr = np.full((2 * K, H, W), np.nan, np.float16)
        ks = list(range(-K, 0)) + list(range(1, K + 1))
        cells = V.flatten().nonzero()[:, 0]
        for i, k in enumerate(ks):
            if w + k not in objs:
                continue
            PB, _, VB, _, ras = ang_z(w + k)
            res = np.full(H * W, np.nan, np.float32)
            for s in range(0, len(cells), chunk):
                c = cells[s:s + chunk]
                p, n = P.view(-1, 3)[c], N.view(-1, 3)[c]
                seed = ras.lookup(A.view(-1)[c], P.view(-1, 3)[c, 2])
                has = seed >= 0
                if not bool(has.any()):
                    continue
                Wb = VB.shape[1]
                t, ok = intersect(p[has], n[has], PB, VB, (seed[has] // Wb).double(), (seed[has] % Wb).double())
                tt = torch.full((len(c),), float("nan"), dtype=torch.float64, device=device)
                tt[has] = torch.where(ok, t, torch.full_like(t, float("nan")))
                res[c.cpu().numpy()] = tt.cpu().numpy()
            nbr[i] = res.reshape(H, W).astype(np.float16)
        out[w] = nbr
    return out


def merged(P: np.ndarray, N: np.ndarray, V: np.ndarray, Theta: np.ndarray, outward: int, axis, K: int = 3,
           device="cuda", chunk: int = 1_000_000) -> np.ndarray:
    """One multi-turn mesh: winding k = the same mesh at Theta + 2 pi k * outward."""
    Pt, Nt, Vt = _surface(P, N, V, device)
    Th = as_t(np.nan_to_num(Theta), device, torch.float64)
    Vt = Vt & as_t(np.isfinite(Theta), device, torch.bool)
    H, W = V.shape
    cell = float(np.median(np.linalg.norm(P[1:] - P[:-1], axis=-1)[V[1:] & V[:-1]]))
    pv = P[V]
    ay, ax = axis.at(pv[:, 2])
    r_med = float(np.median(np.hypot(pv[:, 0] - ax, pv[:, 1] - ay)))
    ras = Raster(Th, Pt[..., 2], Vt, dang=cell / max(r_med, 1.0), dz=cell, fill=6)
    nbr = np.full((2 * K, H, W), np.nan, np.float16)
    ks = list(range(-K, 0)) + list(range(1, K + 1))
    cells = Vt.flatten().nonzero()[:, 0]
    for i, k in enumerate(ks):
        res = np.full(H * W, np.nan, np.float32)
        for s in range(0, len(cells), chunk):
            c = cells[s:s + chunk]
            seed = ras.lookup(Th.view(-1)[c] + 2 * math.pi * k * outward, Pt.view(-1, 3)[c, 2])
            has = seed >= 0
            if not bool(has.any()):
                continue
            t, ok = intersect(Pt.view(-1, 3)[c][has], Nt.view(-1, 3)[c][has], Pt, Vt,
                              (seed[has] // W).double(), (seed[has] % W).double())
            tt = torch.full((len(c),), float("nan"), dtype=torch.float64, device=device)
            tt[has] = torch.where(ok, t, torch.full_like(t, float("nan")))
            res[c.cpu().numpy()] = tt.cpu().numpy()
        nbr[i] = res.reshape(H, W).astype(np.float16)
    return nbr
