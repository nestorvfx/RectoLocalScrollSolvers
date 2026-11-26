"""Spatial side of training-sample synthesis: where each input sample is read from the band.

Lateral map (output cell -> region cell): flips, rotation by any angle, isotropic scale (grid cell size 4-6 vox),
anisotropic scale + shear (grid shear before SLIM), and a smooth elastic / sliding field (tangential errors).

Ray model. The current (erroneous) surface is S_e = p + T n (p, n: the true surface and its normal; T: the error
along n, band voxels). Inference builds stacks along the current mesh's smoothed normals (sigma 2 cells), so the
input is sampled along n_e ~ normalize(n - grad T): per voxel of ray the depth changes by c = 1 / sqrt(1 + |g|^2)
and the position moves laterally by -g / sqrt(1 + |g|^2), g = grad of the smoothed T in voxels per voxel. A random
smooth extra tilt (<= 15 deg) models imperfect normals. Input sample k (virtual units of `step` band voxels, scaled
by the voxel-size factor sigma) lies at ray parameter u(k) = step * sigma * (k + w(k - k*)), w = a monotone depth
warp with w(0) = 0 (compressed / expanded windings), band depth b = T + c u, lateral = base + u * lat.
The target is the exact crossing with the true surface (b = 0): k* = -T / (step * sigma * c), in input units."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import fields as R

DEFAULTS = {
    "scale_p": 0.3, "scale": [0.8, 1.25],
    "aniso_p": 0.2, "aniso": 0.1, "shear": 0.1,
    "elastic_p": 0.3, "elastic_amp": [0.5, 2.0], "elastic_scale": [16.0, 48.0],
    "flip_p": 0.5,
    "voxel_p": 0.3, "voxel": [0.85, 1.15],
    "tilt_p": 0.3, "tilt_deg": 15.0,
    "warp_p": 0.3, "warp_slope": 0.35, "warp_len": [8.0, 40.0],
    "normal_sigma": 2.0,
}


def _cfg(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


@dataclass
class LateralMap:
    A: torch.Tensor          # [B, 2, 2] region cells per output cell (rows: y, x)
    centre: torch.Tensor     # [B, 2] region cell coordinates of the crop centre
    elastic: torch.Tensor    # [B, 2, h, w] region-cell displacement at each output cell
    iso: torch.Tensor        # [B] isotropic scale (region cells per output cell)
    h: int
    w: int

    def map(self, yo: torch.Tensor, xo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output-cell coordinates [B, ...] -> region-cell coordinates [B, ...]."""
        B = self.A.shape[0]
        shp = yo.shape
        vy = (yo - (self.h - 1) / 2).reshape(B, -1)
        vx = (xo - (self.w - 1) / 2).reshape(B, -1)
        ry = self.A[:, 0, 0, None] * vy + self.A[:, 0, 1, None] * vx + self.centre[:, 0, None]
        rx = self.A[:, 1, 0, None] * vy + self.A[:, 1, 1, None] * vx + self.centre[:, 1, None]
        e = sample_cells(self.elastic, yo, xo)                                   # [B, 2, ...]
        return ry.reshape(shp) + e[:, 0], rx.reshape(shp) + e[:, 1]


def sample_cells(f: torch.Tensor, yo: torch.Tensor, xo: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of cell fields f [B, C, h, w] at output-cell coordinates yo, xo [B, ...] (border clamp)."""
    B, C, h, w = f.shape
    shp = yo.shape[1:]
    gy = 2 * yo.reshape(B, -1, 1) / max(h - 1, 1) - 1
    gx = 2 * xo.reshape(B, -1, 1) / max(w - 1, 1) - 1
    g = torch.stack([gx, gy], -1)                                                # [B, n, 1, 2]
    out = F.grid_sample(f, g.to(f.dtype), mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(B, C, *shp)


def lateral_map(B, h, w, region, gen, device, cfg=None, train=True, margin: float = 0.0) -> LateralMap:
    """margin: extra cells kept free around the footprint (the lateral travel of tilted rays)."""
    c = _cfg(cfg)
    if train:
        th = R.urand((B,), 0, 2 * math.pi, gen, device)
        iso = torch.where(R.coin((B,), c["scale_p"], gen, device), R.urand((B,), *c["scale"], gen, device),
                          torch.ones(B, device=device))
        an = R.coin((B,), c["aniso_p"], gen, device).float()
        sy = 1 + an * R.urand((B,), -c["aniso"], c["aniso"], gen, device)
        sx = 1 + an * R.urand((B,), -c["aniso"], c["aniso"], gen, device)
        sh = an * R.urand((B,), -c["shear"], c["shear"], gen, device)
        fy = torch.where(R.coin((B,), c["flip_p"], gen, device), -1.0, 1.0)
        fx = torch.where(R.coin((B,), c["flip_p"], gen, device), -1.0, 1.0)
    else:
        th = torch.zeros(B, device=device)
        iso = torch.ones(B, device=device)
        sy = sx = fy = fx = torch.ones(B, device=device)
        sh = torch.zeros(B, device=device)
    co, si = torch.cos(th), torch.sin(th)
    Rm = torch.stack([torch.stack([co, -si], -1), torch.stack([si, co], -1)], -2)          # [B, 2, 2]
    S = torch.stack([torch.stack([sy * fy, sh], -1), torch.stack([torch.zeros_like(sx), sx * fx], -1)], -2)
    A = iso[:, None, None] * (Rm @ S)
    E = torch.zeros((B, 2, h, w), device=device)
    if train:
        on = R.coin((B,), c["elastic_p"], gen, device)
        if bool(on.any()):
            amp = R.urand((B, 1, 1), *c["elastic_amp"], gen, device)
            sc = R.urand((B,), *c["elastic_scale"], gen, device)
            E[:, 0] = on[:, None, None] * amp * R.smooth_noise(B, h, w, sc, gen, device) / 2
            E[:, 1] = on[:, None, None] * amp * R.smooth_noise(B, h, w, sc, gen, device) / 2
    # centre: keep the footprint inside the region where possible
    half = 0.5 * (A.abs() @ torch.tensor([float(h), float(w)], device=device)) + E.abs().amax((2, 3)) + margin
    lo = half
    hi = (region - 1) - half
    u = torch.rand((B, 2), generator=gen, device=device) if train else torch.full((B, 2), 0.5, device=device)
    centre = torch.where(hi > lo, lo + u * (hi - lo), torch.full_like(lo, (region - 1) / 2))
    return LateralMap(A=A, centre=centre, elastic=E, iso=iso, h=h, w=w)


@dataclass
class Ray:
    c: torch.Tensor         # [B, h, w] depth change per ray voxel
    lat: torch.Tensor       # [B, 2, h, w] region cells per ray voxel
    sigma: torch.Tensor     # [B] voxel-size factor
    warp_a: torch.Tensor    # [B] depth-warp amplitude (input units)
    warp_lam: torch.Tensor  # [B] depth-warp wavelength (input units)
    warp_ph: torch.Tensor   # [B]


def ray(T: torch.Tensor, lm: LateralMap, cell_vox: torch.Tensor, gen, device, cfg=None, train=True) -> Ray:
    c_ = _cfg(cfg)
    B, h, w = T.shape
    Ts = R.blur(T, c_["normal_sigma"])
    gy, gx = R.grad2d(Ts)                                                       # vox per output cell
    L = (lm.iso * cell_vox)[:, None, None]                                      # band vox per output cell
    g = torch.stack([gy, gx], 1) / L[:, None]                                   # dimensionless, output frame
    den = torch.sqrt(1 + (g * g).sum(1))
    c = 1 / den
    lat_out = -g / den[:, None] / L[:, None]                                    # output cells per ray vox
    lat = torch.einsum("bij,bjhw->bihw", lm.A, lat_out)                         # region cells per ray vox
    if train:
        on = R.coin((B, 1, 1), c_["tilt_p"], gen, device)
        tmax = math.tan(math.radians(c_["tilt_deg"]))
        mag = R.urand((B, 1, 1), 0, tmax, gen, device) * on
        ny = R.smooth_noise(B, h, w, 32.0, gen, device)
        nx = R.smooth_noise(B, h, w, 32.0, gen, device)
        rms = torch.sqrt((ny * ny + nx * nx).mean((1, 2), keepdim=True)).clamp(min=1e-6)
        ty = mag * ny / rms                                                    # RMS tilt = mag (up to tan 15 deg)
        tx = mag * nx / rms
        lat = lat + torch.stack([ty, tx], 1) / cell_vox[:, None, None, None]
        c = c / torch.sqrt(1 + ty * ty + tx * tx)
        sigma = torch.where(R.coin((B,), c_["voxel_p"], gen, device), R.urand((B,), *c_["voxel"], gen, device),
                            torch.ones(B, device=device))
        won = R.coin((B,), c_["warp_p"], gen, device).float()
        lam = R.urand((B,), *c_["warp_len"], gen, device)
        a = won * c_["warp_slope"] * lam / (2 * math.pi) * torch.rand((B,), generator=gen, device=device)
        ph = R.urand((B,), 0, 2 * math.pi, gen, device)
    else:
        sigma = torch.ones(B, device=device)
        a = torch.zeros(B, device=device)
        lam = torch.full((B,), 20.0, device=device)
        ph = torch.zeros(B, device=device)
    return Ray(c=c, lat=lat, sigma=sigma, warp_a=a, warp_lam=lam, warp_ph=ph)


def target(T: torch.Tensor, ry: Ray, step: float) -> torch.Tensor:
    """k* [B, h, w]: input units along the ray from the current surface to the true one."""
    return -T / (step * ry.sigma[:, None, None] * ry.c)


def ray_parameter(k: torch.Tensor, kstar: torch.Tensor, ry: Ray, step: float) -> torch.Tensor:
    """u(k) in band voxels. k [D] input units; kstar [B, ...] -> [B, D, ...]."""
    shp = (-1,) + (1,) * (kstar.dim() - 1)
    x = k.view(1, -1, *([1] * (kstar.dim() - 1))) - kstar[:, None]
    a = ry.warp_a.view(shp)[:, None]
    lam = ry.warp_lam.view(shp)[:, None]
    ph = ry.warp_ph.view(shp)[:, None]
    wv = a * (torch.sin(2 * math.pi * x / lam + ph) - torch.sin(ph))
    return step * ry.sigma.view(shp)[:, None] * (k.view(1, -1, *([1] * (kstar.dim() - 1))) + wv)
