"""Renders of the CT on a surface, for figures: about one pixel per voxel, the mean over a few voxels along the normals,
with the known (context) cells tinted green and the solved region outlined."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

from ..common.geometry import as_t
from ..infer.surface import normals


@torch.no_grad()
def render(vol, P: np.ndarray, V: np.ndarray, axis=None, up: int = 5, depths=(-1.0, 0.0, 1.0), device="cuda"):
    """vol: an engine.Volume; P [H, W, 3] xyz at the processing level, V [H, W]. -> grey [H*up, W*up] float (NaN
    outside), sampled from the level-0 cache."""
    Pt, Vt = as_t(P, device, torch.float64), as_t(V, device, torch.bool)
    N = normals(Pt, Vt, axis)
    H, W = V.shape
    big = lambda a: F.interpolate(a.permute(2, 0, 1)[None].float(), size=(H * up, W * up), mode="bilinear",  # noqa: E731
                                  align_corners=True)[0].permute(1, 2, 0)
    Pu, Nu = big(Pt), big(N)
    Nu = Nu / Nu.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    Vu = F.interpolate(Vt[None, None].float(), size=(H * up, W * up), mode="nearest")[0, 0] > 0.5
    acc = torch.zeros((H * up, W * up), device=device)
    for d in depths:
        zyx = (Pu + d * Nu).flip(-1)[None].float()
        acc += vol.sample(vol.fine, zyx, Vu)[0]
    img = (acc / len(depths)).cpu().numpy()
    return np.where(Vu.cpu().numpy(), img, np.nan)


def to_rgb(img: np.ndarray, known: np.ndarray | None = None, solve: np.ndarray | None = None, up: int = 5,
           lo_hi=None, alpha: float = 0.22) -> Image.Image:
    """Grey render -> RGB with the known cells tinted green and the solve region outlined (yellow)."""
    ok = np.isfinite(img)
    lo, hi = lo_hi or np.percentile(img[ok], [0.5, 99.5])
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((g * 255)[..., None], 3, -1)
    if known is not None:
        k = np.kron(known.astype(np.uint8), np.ones((up, up), np.uint8)).astype(bool)[:img.shape[0], :img.shape[1]]
        rgb[k] = (1 - alpha) * rgb[k] + alpha * np.array([40, 220, 60])
    rgb[~ok] = 0
    im = Image.fromarray(rgb.astype(np.uint8))
    if solve is not None:
        s = np.kron(solve.astype(np.uint8), np.ones((up, up), np.uint8)).astype(bool)[:img.shape[0], :img.shape[1]]
        edge = s & ~ndimage.binary_erosion(s, iterations=2)
        a = np.asarray(im).copy()
        a[edge] = (255, 220, 0)
        im = Image.fromarray(a)
    return im


def error_rgb(err: np.ndarray, mask: np.ndarray, lim: float = 6.0, up: int = 5) -> Image.Image:
    """Signed error (vox) on `mask` cells: blue - / red +, white within 0.5 vox, black elsewhere."""
    e = np.clip(err / lim, -1, 1)
    rgb = np.zeros(err.shape + (3,), np.float64)
    rgb[..., 0] = np.where(e > 0, 1.0, 1.0 + e)
    rgb[..., 1] = 1.0 - np.abs(e)
    rgb[..., 2] = np.where(e < 0, 1.0, 1.0 - e)
    rgb[~mask] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((err.shape[1] * up, err.shape[0] * up), Image.NEAREST)


def legend(width: int, items: list[tuple[str, tuple]], size: int = 16) -> Image.Image:
    """A legend strip `width` px wide: a short line in each item's colour, then its label, left to right."""
    f = _font(size)
    h = int(size * 1.7)
    im = Image.new("RGB", (width, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 6
    for label, colour in items:
        d.line([(x, h // 2), (x + 18, h // 2)], fill=colour, width=3)
        d.text((x + 24, (h - size) // 2 - 1), label, fill=colour, font=f)
        x += 24 + int(f.getlength(label) if hasattr(f, "getlength") else 7 * len(label)) + 16
    return im


def _font(size: int):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(tiles: list[tuple[str, Image.Image]], path, pad: int = 8) -> None:
    """Tiles side by side, each with its title above it (the font scaled to the tile's width, and down until the
    longest title fits within its tile)."""
    tw = min(t.width for _, t in tiles)
    size = max(14, tw // 16)
    f = _font(size)
    while hasattr(f, "getlength") and size > 10 and max(f.getlength(tt) for tt, _ in tiles) > tw - 12:
        size -= 1
        f = _font(size)
    bar = int(f.size * 1.6) if hasattr(f, "size") else 26
    w = sum(t.width for _, t in tiles) + pad * (len(tiles) - 1)
    h = max(t.height for _, t in tiles) + bar
    out = Image.new("RGB", (w, h), (24, 24, 24))
    x = 0
    d = ImageDraw.Draw(out)
    for title, t in tiles:
        out.paste(t, (x, bar))
        d.text((x + 6, bar // 5), title, fill=(255, 230, 120), font=f)
        x += t.width + pad
    out.save(path, quality=92)


@torch.no_grad()
def cross_section(vol, P: np.ndarray, N: np.ndarray, V: np.ndarray, row: int, curves: dict, half: int = 40,
                  up: int = 5, device="cuda") -> Image.Image:
    """The CT along grid row `row` of a reference surface (P, N), +-half vox along its normals (up px per cell along
    the row, 1 px per vox in depth; the reference at the middle line), with each surface of `curves` drawn as its
    offset along the normals: {name: (offsets [W] vox, NaN where absent, (r, g, b)[, "dash"])}."""
    W = V.shape[1]
    p = torch.as_tensor(P[row], device=device, dtype=torch.float64)
    n = torch.as_tensor(N[row], device=device, dtype=torch.float64)
    p = F.interpolate(p.T[None], size=W * up, mode="linear", align_corners=True)[0].T
    n = F.interpolate(n.T[None], size=W * up, mode="linear", align_corners=True)[0].T
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ok = torch.as_tensor(np.repeat(V[row], up)[: W * up], device=device)
    t = torch.arange(-half, half + 1, device=device, dtype=torch.float64)
    zyx = (p[None] + t[:, None, None] * n[None]).flip(-1)[:, None].float()    # [D, 1, W*up, 3]
    img = vol.sample(vol.fine, zyx, ok[None].expand(1, -1))[:, 0].cpu().numpy()  # [D, W*up]
    okn = ok.cpu().numpy()
    lo, hi = np.percentile(img[:, okn], [0.5, 99.5]) if okn.any() else (0, 255)
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((g * 255)[..., None], 3, -1).astype(np.uint8)
    rgb[:, ~okn] = 0
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    x = (np.arange(W) + 0.5) * up
    for name, spec in curves.items():
        off, col, dash = spec[0], spec[1], len(spec) > 2 and spec[2] == "dash"
        y = half - np.asarray(off, float)          # + offsets (along the normal) drawn upward
        pts = [(float(xi), float(yi)) if np.isfinite(yi) else None for xi, yi in zip(x, y)]
        for i in range(len(pts) - 1):
            if pts[i] is None or pts[i + 1] is None or (dash and (i // 2) % 2):
                continue
            d.line([pts[i], pts[i + 1]], fill=col, width=2 if not dash else 1)
    return im
