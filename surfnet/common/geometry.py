"""Grid geometry on the GPU (torch): masked smoothing, normals, bilinear / bicubic grid sampling, resampling, the scroll
axis.

Conventions:
- A surface is a grid P [H, W, 3] of xyz positions in processing-level voxels (tifxyz order) with a valid mask V.
- Normals come from the masked-Gaussian-smoothed grid (sigma 2 cells), as the cross product of the row and column
  derivatives. They are oriented with ONE global sign toward the scroll axis: the recto faces the axis in these scrolls.
  The same rule builds the training bands and the inference stacks, so both see identical sampling directions.
- Volume sampling uses zyx; `xyz_to_zyx` flips the last axis."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def as_t(a, device="cpu", dtype=torch.float32) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dtype)


def xyz_to_zyx(p: torch.Tensor) -> torch.Tensor:
    return p.flip(-1)


def gauss1d(sigma: float, device, dtype=torch.float64) -> tuple[torch.Tensor, int]:
    r = max(1, int(4 * sigma + 0.5))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum(), r


def blur2d(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [C, H, W] (zero padding)."""
    if sigma <= 0:
        return t
    g, r = gauss1d(sigma, t.device, t.dtype)
    x = F.conv2d(t[:, None], g.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


def masked_blur(f: torch.Tensor, v: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalised-convolution blur of f [C, H, W] over the valid cells v [H, W]."""
    vv = v.to(f.dtype)
    w = blur2d(vv[None], sigma)[0]
    return blur2d(f * vv[None], sigma) / w.clamp(min=1e-6)[None]


def grid_normals(P: torch.Tensor, V: torch.Tensor, sigma: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit normals [H, W, 3] (xyz, unoriented) of the smoothed grid, and where they are defined."""
    P = P.to(torch.float64)
    f = masked_blur(P.permute(2, 0, 1), V, sigma)                     # [3, H, W]
    du = torch.gradient(f, dim=1)[0]
    dv = torch.gradient(f, dim=2)[0]
    n = torch.stack([du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]], -1)
    nn_ = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    ok = V & (nn_[..., 0] > 1e-9)
    return (n / nn_.clamp(min=1e-12)), ok


class Axis:
    """The scroll axis as a polyline over z: (z, y, x) samples in processing-level voxels, linear in between."""

    def __init__(self, z, y, x):
        o = np.argsort(np.asarray(z, np.float64))
        self.z = np.asarray(z, np.float64)[o]
        self.y = np.asarray(y, np.float64)[o]
        self.x = np.asarray(x, np.float64)[o]

    @classmethod
    def from_json(cls, path: str | Path, scale: float = 1.0) -> "Axis":
        a = json.loads(Path(path).read_text())
        return cls(np.asarray(a["z"]) * scale, np.asarray(a["y"]) * scale, np.asarray(a["x"]) * scale)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"z": self.z.tolist(), "y": self.y.tolist(), "x": self.x.tolist()}))

    def at(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(z, self.z, self.y), np.interp(z, self.z, self.x)

    def angle(self, P_xyz: np.ndarray) -> np.ndarray:
        ay, ax = self.at(P_xyz[..., 2])
        return np.arctan2(P_xyz[..., 0] - ax, P_xyz[..., 1] - ay)


def orient_to_axis(N: torch.Tensor, P: torch.Tensor, ok: torch.Tensor, axis: Axis | None) -> torch.Tensor:
    """One global sign so that the majority of normals point toward the axis (recto side). Without an axis the
    normals are returned unchanged (the caller must know the convention)."""
    if axis is None or not bool(ok.any()):
        return N
    z = P[..., 2].detach().cpu().numpy()
    ay, ax = axis.at(z)
    inward = torch.stack([as_t(ax, P.device, P.dtype) - P[..., 0], as_t(ay, P.device, P.dtype) - P[..., 1],
                          torch.zeros_like(P[..., 0])], -1)
    s = (N.to(P.dtype) * inward).sum(-1)
    sign = 1.0 if float((s[ok] > 0).double().mean()) >= 0.5 else -1.0
    return N * sign


def bilinear(A: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Sample A [H, W, C] at float grid coordinates r, c (any equal shape) -> [..., C]. Clamped at the border."""
    H, W = A.shape[:2]
    r = r.clamp(0, H - 1)
    c = c.clamp(0, W - 1)
    r0 = r.floor().long().clamp(0, max(H - 2, 0))
    c0 = c.floor().long().clamp(0, max(W - 2, 0))
    r1 = (r0 + 1).clamp(max=H - 1)
    c1 = (c0 + 1).clamp(max=W - 1)
    fr = (r - r0).unsqueeze(-1).to(A.dtype)
    fc = (c - c0).unsqueeze(-1).to(A.dtype)
    return ((1 - fr) * (1 - fc) * A[r0, c0] + fr * (1 - fc) * A[r1, c0] + (1 - fr) * fc * A[r0, c1] + fr * fc * A[r1, c1])


def valid_at(V: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """True where all four bilinear corners are valid (and inside the grid)."""
    H, W = V.shape
    inside = (r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)
    r0 = r.floor().long().clamp(0, max(H - 2, 0))
    c0 = c.floor().long().clamp(0, max(W - 2, 0))
    r1 = (r0 + 1).clamp(max=H - 1)
    c1 = (c0 + 1).clamp(max=W - 1)
    return inside & V[r0, c0] & V[r1, c0] & V[r0, c1] & V[r1, c1]


def _cubic_w(t: torch.Tensor) -> list[torch.Tensor]:
    """Catmull-Rom weights for the 4 taps at offsets -1, 0, 1, 2."""
    t2, t3 = t * t, t * t * t
    return [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2]


def bicubic(A: torch.Tensor, V: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Catmull-Rom sampling of A [H, W, C]; falls back to bilinear where a 4 x 4 stencil is not fully valid.
    Returns values [..., C] and validity (bilinear rule)."""
    H, W = A.shape[:2]
    ok = valid_at(V, r, c)
    lin = bilinear(A, r, c)
    r0 = r.floor().long()
    c0 = c.floor().long()
    fr = (r - r0).to(A.dtype)
    fc = (c - c0).to(A.dtype)
    wr, wc = _cubic_w(fr), _cubic_w(fc)
    out = torch.zeros_like(lin)
    full = torch.ones_like(ok)
    for i, di in enumerate((-1, 0, 1, 2)):
        rr = r0 + di
        for j, dj in enumerate((-1, 0, 1, 2)):
            cc = c0 + dj
            ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rrc, ccc = rr.clamp(0, H - 1), cc.clamp(0, W - 1)
            full = full & ins & V[rrc, ccc]
            out = out + (wr[i] * wc[j]).unsqueeze(-1) * A[rrc, ccc]
    return torch.where(full.unsqueeze(-1), out, lin), ok


def resample_grid(P: torch.Tensor, V: torch.Tensor, step_r: float, step_c: float, cubic: bool = True,
                  origin: tuple[float, float] = (0.0, 0.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a surface grid with a new step (in old cells) along rows / columns: the output cell (i, j) lies at old
    coordinates (origin[0] + i * step_r, origin[1] + j * step_c)."""
    H, W = V.shape
    Ho = int(math.floor((H - 1 - origin[0]) / step_r)) + 1
    Wo = int(math.floor((W - 1 - origin[1]) / step_c)) + 1
    rr = origin[0] + torch.arange(Ho, device=P.device, dtype=P.dtype) * step_r
    cc = origin[1] + torch.arange(Wo, device=P.device, dtype=P.dtype) * step_c
    R, C = torch.meshgrid(rr, cc, indexing="ij")
    if cubic:
        Q, ok = bicubic(P, V, R, C)
    else:
        Q, ok = bilinear(P, R, C), valid_at(V, R, C)
    return Q, ok


def edge_steps(P: np.ndarray, V: np.ndarray) -> tuple[float, float]:
    """Median distance (voxels) between neighbouring valid cells along rows and along columns."""
    dr = np.linalg.norm(P[1:] - P[:-1], axis=-1)[V[1:] & V[:-1]]
    dc = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[V[:, 1:] & V[:, :-1]]
    return float(np.median(dr)) if dr.size else float("nan"), float(np.median(dc)) if dc.size else float("nan")


def grid_steps(P: np.ndarray, V: np.ndarray, known: np.ndarray) -> tuple[float, float]:
    """A sheet's grid step (voxels) along rows and columns, measured on its known cells: the part of the sheet known to
    be right (a draft's area, or an area bridged from the known cells, can be stretched or compressed). Along an axis
    on which no two known cells neighbour, measured on all valid cells V."""
    k, v = edge_steps(P, known), edge_steps(P, V)
    return tuple(a if np.isfinite(a) else b for a, b in zip(k, v))


def axis_direction(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """[2, H, W]: unit direction (rows, cols) of the scroll axis (+z) within the surface, per cell (0 where undefined).
    Solves z = a dP/dr + b dP/dc in the tangent plane (least squares with the grid's metric)."""
    Pr = np.gradient(P, axis=0)
    Pc = np.gradient(P, axis=1)
    g11 = (Pr * Pr).sum(-1)
    g12 = (Pr * Pc).sum(-1)
    g22 = (Pc * Pc).sum(-1)
    zr, zc = Pr[..., 2], Pc[..., 2]
    det = g11 * g22 - g12 * g12
    ok = V & (det > 1e-9)
    a = np.where(ok, (g22 * zr - g12 * zc) / np.where(ok, det, 1), 0)
    b = np.where(ok, (g11 * zc - g12 * zr) / np.where(ok, det, 1), 0)
    n = np.sqrt(a * a + b * b)
    out = np.stack([np.where(n > 1e-9, a / np.maximum(n, 1e-9), 0), np.where(n > 1e-9, b / np.maximum(n, 1e-9), 0)])
    return out.astype(np.float32)
