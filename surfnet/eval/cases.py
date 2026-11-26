"""Test cases on a ground-truth surface: where to place them, which cells are known, which are to be recovered, and a
wrong draft to start from.

Layouts (on a square window; `size` = the target square in cells, `known` = the context width in cells):
  ring      the known cells surround the target (a hole to fill);
  side      a known strip along one side of the target (an extension, as a tracer grows);
  between   known strips on two opposite sides;
  patches   random known disks covering 30 % of a `known`-wide band around the target.
A wrong draft (`draft_offsets`): the truth displaced along its normals by a natural perturbation (`perturb.py`) that
grows from 0 at the known cells, |A| in `amp` vox; `perturb: winding` also moves one region of the target by one
winding spacing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..infer.perturb import FAMILIES, natural_field


def extent(case: dict, margin: int = 12) -> int:
    """The window side (cells) a case needs."""
    return int(case["size"]) + 2 * int(case["known"]) + 2 * margin


def layout(n_win: int, case: dict, rng) -> tuple[np.ndarray, np.ndarray]:
    """known, target masks on an n_win x n_win window, the target square centred."""
    n, k = int(case["size"]), int(case["known"])
    r0 = c0 = (n_win - n) // 2
    target = np.zeros((n_win, n_win), bool)
    target[r0:r0 + n, c0:c0 + n] = True
    known = np.zeros((n_win, n_win), bool)
    lay = case["layout"]
    if lay == "side":
        known[r0:r0 + n, c0 - k:c0] = True
    elif lay == "between":
        known[r0:r0 + n, c0 - k:c0] = True
        known[r0:r0 + n, c0 + n:c0 + n + k] = True
    elif lay == "ring":
        known[r0 - k:r0 + n + k, c0 - k:c0 + n + k] = True
        known &= ~target
    elif lay == "patches":
        band = np.zeros((n_win, n_win), bool)
        band[r0 - k:r0 + n + k, c0 - k:c0 + n + k] = True
        band &= ~target
        yy, xx = np.ogrid[:n_win, :n_win]
        while known.sum() < 0.3 * band.sum():
            y, x = rng.integers(r0 - k, r0 + n + k), rng.integers(c0 - k, c0 + n + k)
            rad = rng.uniform(3, 10)
            known |= ((yy - y) ** 2 + (xx - x) ** 2 <= rad * rad) & band
    else:
        raise ValueError(f"layout {lay}")
    return known, target


def place_blocks(valid: np.ndarray, size: int, count: int, rng, tries: int = 4000) -> list[tuple[int, int]]:
    """Up to `count` non-overlapping size x size blocks whose cells are all valid (eroded by 2). -> origins."""
    H, W = valid.shape
    if H < size or W < size:
        return []
    ok = ndimage.binary_erosion(valid, iterations=2)
    integ = np.pad(np.cumsum(np.cumsum(ok, 0), 1), ((1, 0), (1, 0)))
    taken = np.zeros_like(valid)
    out = []
    for _ in range(tries):
        if len(out) == count:
            break
        r, c = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
        s = integ[r + size, c + size] - integ[r, c + size] - integ[r + size, c] + integ[r, c]
        if s == size * size and not taken[r:r + size, c:c + size].any():
            out.append((r, c))
            taken[r:r + size, c:c + size] = True
    return out


def place_windows(valid: np.ndarray, size: int, count: int, centres: list, rng, tries: int = 4000) -> list:
    """Up to `count` non-overlapping size x size windows of valid cells (eroded by 2): first centred on `centres`
    (shared locations, so that cases of different sizes reuse the same CT), then anywhere. -> origins."""
    H, W = valid.shape
    ok = ndimage.binary_erosion(valid, iterations=2)
    integ = np.pad(np.cumsum(np.cumsum(ok, 0), 1), ((1, 0), (1, 0)))
    taken = np.zeros_like(valid)
    out = []

    def fits(r, c):
        if r < 0 or c < 0 or r + size > H or c + size > W:
            return False
        s = integ[r + size, c + size] - integ[r, c + size] - integ[r + size, c] + integ[r, c]
        return s == size * size and not taken[r:r + size, c:c + size].any()

    cands = [(int(cy) - size // 2, int(cx) - size // 2) for cy, cx in centres]
    cands += [(int(rng.integers(0, max(H - size, 0) + 1)), int(rng.integers(0, max(W - size, 0) + 1))) for _ in range(tries)]
    for r, c in cands:
        if len(out) == count:
            break
        if fits(r, c):
            out.append((r, c))
            taken[r:r + size, c:c + size] = True
    return out


def draft_offsets(known: np.ndarray, target: np.ndarray, case: dict, spacing: float, rng) -> np.ndarray:
    """Offsets (vox along the truth's normals) of a wrong draft on the target cells, 0 elsewhere."""
    H, W = known.shape
    lo, hi = case.get("amp", [4.0, 11.0])
    fam = FAMILIES[int(rng.integers(len(FAMILIES)))]
    A, ell, rough = natural_field(fam, H, W, rng, lo, hi, float(rng.choice([-1.0, 1.0])))
    dist = ndimage.distance_transform_edt(~known)
    T = (1 - np.exp(-dist / ell)) * (A + 0.6 * rough)
    if case.get("perturb") == "winding":                      # one region of the target on the next winding
        blob = ndimage.gaussian_filter(rng.standard_normal((H, W)), 12) > 0
        blob &= target & (dist > 8)
        T = np.where(blob, T + float(rng.choice([-1.0, 1.0])) * spacing, T)
    return np.where(target, T, 0.0)
