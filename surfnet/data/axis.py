"""The scroll axis (centre of the spiral per height), in processing-level voxels.

- `from_meshes`: algebraic circle fits (Kasa 1976) of the GT surface points in z slabs, robustly re-fitted without the
  worst residuals, smoothed over z. Multi-winding GT (several full turns per slab) pins the spiral's centre to
  within ~spacing / 4 (a circle fit to a spiral is slightly biased). That is enough: the axis only sets the normals'
  global sign and a consistent angle for neighbour seeds and turn unwrapping.
- `from_volume`: the occupancy centroid per z of a coarse pyramid level of the masked CT (non-zero = inside the
  scroll mask); for objects without multi-turn geometry.
- `from_json`: a stored axis {z, y, x}."""
from __future__ import annotations

import numpy as np

from ..common.geometry import Axis


def circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = -(x * x + y * y)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = np.sqrt(max(cx * cx + cy * cy - sol[2], 0.0))
    return cx, cy, r


def from_meshes(meshes: list[tuple[np.ndarray, np.ndarray]], slab: float = 128.0, min_pts: int = 2000,
                smooth: int = 3, seed: int = 0) -> Axis:
    """meshes: (P [H, W, 3] xyz, V [H, W]) in processing-level voxels."""
    rng = np.random.default_rng(seed)
    pts = np.concatenate([P[V] for P, V in meshes])
    if len(pts) > 2_000_000:
        pts = pts[rng.choice(len(pts), 2_000_000, replace=False)]
    z = pts[:, 2]
    zs, cxs, cys = [], [], []
    for z0 in np.arange(z.min(), z.max() + 1, slab):
        m = (z >= z0) & (z < z0 + slab)
        if m.sum() < min_pts:
            continue
        x, y = pts[m, 0], pts[m, 1]
        ang = np.unwrap(np.sort(np.arctan2(y - y.mean(), x - x.mean())))
        if ang[-1] - ang[0] < np.pi:                       # less than half a turn: the centre is not determined
            continue
        cx, cy, _ = circle_fit(x, y)
        for _ in range(3):                                 # drop the worst 20 % and refit
            res = np.abs(np.hypot(x - cx, y - cy) - np.median(np.hypot(x - cx, y - cy)))
            k = res <= np.quantile(res, 0.8)
            cx, cy, _ = circle_fit(x[k], y[k])
        zs.append(z0 + slab / 2)
        cxs.append(cx)
        cys.append(cy)
    if not zs:
        raise ValueError("no slab with enough multi-turn points for an axis fit")
    cxs, cys = np.asarray(cxs), np.asarray(cys)
    if smooth > 1 and len(zs) >= smooth:
        k = np.ones(smooth) / smooth
        cxs = np.convolve(np.pad(cxs, smooth // 2, mode="edge"), k, "valid")
        cys = np.convolve(np.pad(cys, smooth // 2, mode="edge"), k, "valid")
    return Axis(zs, cys, cxs)


def from_volume(source, level: int, coarse_level: int = 5, smooth: int = 5) -> Axis:
    """Occupancy centroid per z of `coarse_level`, in `level` voxels."""
    from ..volume import CHUNK
    shp = source.shape(coarse_level)
    grid = source.grid(coarse_level)
    keys = np.array([(a, b, c) for a in range(grid[0]) for b in range(grid[1]) for c in range(grid[2])])
    source.fetch(coarse_level, keys)
    vol = np.zeros(shp, np.uint8)
    for k in keys:
        a = source.read(coarse_level, tuple(int(t) for t in k))
        if a is None:
            continue
        z, y, x = (int(t) * CHUNK for t in k)
        sz = vol[z:z + CHUNK, y:y + CHUNK, x:x + CHUNK].shape
        vol[z:z + CHUNK, y:y + CHUNK, x:x + CHUNK] = a[:sz[0], :sz[1], :sz[2]]
    occ = vol > 0
    zs, cy, cx = [], [], []
    yy, xx = np.mgrid[0:shp[1], 0:shp[2]]
    for zi in range(shp[0]):
        m = occ[zi]
        if m.sum() < 16:
            continue
        zs.append(zi)
        cy.append(yy[m].mean())
        cx.append(xx[m].mean())
    zs, cy, cx = map(np.asarray, (zs, cy, cx))
    if smooth > 1 and len(zs) >= smooth:
        k = np.ones(smooth) / smooth
        cy = np.convolve(np.pad(cy, smooth // 2, mode="edge"), k, "valid")
        cx = np.convolve(np.pad(cx, smooth // 2, mode="edge"), k, "valid")
    f = 2 ** (coarse_level - level)                        # coarse voxel i covers f fine voxels
    to = lambda q: q * f + (f - 1) / 2.0                   # noqa: E731
    return Axis(to(zs), to(cy), to(cx))
