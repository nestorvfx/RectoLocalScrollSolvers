"""Natural perturbation fields: the displacements used to build test starts and the perturbed starts of a correction.

Five procedural families (never used as training perturbations):
- hybrid multifractal: smooth plains and rough peaks (Musgrave);
- domain-warped: two-level warped fBm, flowing marbled structure (Quilez);
- fragment plates: Worley cells with warped borders, each plate its own sign, size and tilt;
- fibre streaks: a multifractal stretched along a fibre direction, plus local bulges;
- ridged multifractal: sharp ridges, smooth valleys.
`natural_field` returns the full-strength signed field A (|A| in [dmin, dmax] vox), a divergence length ell (cells)
and an inner roughness. A start then diverges from the known cells as T = (1 - exp(-r / ell)) (A + noise rough),
r = the distance (cells) to the nearest known cell."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _octave(rng, H, W, scale, pad):
    gh, gw = int(np.ceil((H + 2 * pad) / scale)) + 3, int(np.ceil((W + 2 * pad) / scale)) + 3
    f = ndimage.zoom(rng.standard_normal((gh, gw)), scale, order=3)[:H + 2 * pad, :W + 2 * pad]
    return (f - f.mean()) / max(float(f.std()), 1e-6)


class Fbm:
    """Value-noise octaves on the cell grid (scale0 / 2^i cells), evaluable at warped coordinates."""

    def __init__(self, rng, H, W, scale0=25.0, octaves=5, pad=64):
        self.pad = pad
        self.oct = [_octave(rng, H, W, max(scale0 / 2 ** i, 1.0), pad) for i in range(octaves)]

    def n(self, i, Y, X):                                       # octave i in about [-1, 1]
        o = ndimage.map_coordinates(self.oct[i], [Y + self.pad, X + self.pad], order=1, mode="mirror")
        return np.clip(0.5 * o, -1, 1)

    def fbm(self, Y, X, h=0.5):
        w = [2.0 ** (-i * h) for i in range(len(self.oct))]
        return sum(wi * self.n(i, Y, X) for i, wi in enumerate(w)) / sum(w)

    def hybrid(self, Y, X, offset=0.7, h=0.25):
        """Musgrave's hybrid multifractal: smooth plains and rough peaks in one field."""
        res = self.n(0, Y, X) + offset
        weight = res.copy()
        for i in range(1, len(self.oct)):
            weight = np.minimum(weight, 1.0)
            signal = (self.n(i, Y, X) + offset) * 2.0 ** (-i * h)
            res = res + weight * signal
            weight = weight * signal
        return res

    def ridged(self, Y, X, offset=1.0, gain=2.0, h=0.9):
        """Musgrave's ridged multifractal: sharp ridges, smooth valleys."""
        signal = (offset - np.abs(self.n(0, Y, X))) ** 2
        res = signal.copy()
        for i in range(1, len(self.oct)):
            weight = np.clip(signal * gain, 0, 1)
            signal = (offset - np.abs(self.n(i, Y, X))) ** 2 * weight
            res = res + signal * 2.0 ** (-i * h)
        return res


def _span(f, lo, hi):
    a, b = np.percentile(f, [2, 98])
    return lo + (hi - lo) * np.clip((f - a) / max(b - a, 1e-9), 0, 1)


FAMILIES = ("hybrid multifractal", "domain-warped", "fragment plates", "fibre streaks", "ridged multifractal")


def natural_field(fam: str, H: int, W: int, rng, dmin: float, dmax: float, sign: float):
    """Full-strength signed field A, divergence-length map ell and inner roughness of one family, new parameters per
    call (cells grid)."""
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    F = [Fbm(rng, H, W, scale0=rng.uniform(15, 40)) for _ in range(3)]
    if fam == "hybrid multifractal":
        A = _span(F[0].hybrid(Y, X, offset=rng.uniform(0.5, 0.9), h=rng.uniform(0.15, 0.4)), dmin, dmax)
    elif fam == "domain-warped":
        w = rng.uniform(15, 45)
        Fw = [Fbm(rng, H, W, scale0=rng.uniform(30, 60), octaves=3) for _ in range(4)]
        q0, q1 = Fw[0].fbm(Y, X), Fw[1].fbm(Y, X)
        r0, r1 = Fw[2].fbm(Y + w * q0, X + w * q1), Fw[3].fbm(Y + w * q0, X + w * q1)
        A = _span(F[0].fbm(Y + w * r0, X + w * r1, h=rng.uniform(0.5, 1.0)), dmin, dmax)
    elif fam == "fragment plates":
        from scipy.spatial import cKDTree
        d = rng.uniform(8, 25)
        n = max(4, int(H * W / d ** 2))
        pts = np.c_[rng.uniform(0, H, n), rng.uniform(0, W, n)]
        wv = rng.uniform(10, 25)
        Yw, Xw = Y + wv * F[0].fbm(Y, X), X + wv * F[1].fbm(Y, X)
        lab = cKDTree(pts).query(np.c_[Yw.ravel(), Xw.ravel()])[1].reshape(H, W)
        mag = rng.uniform(dmin, dmax, n) * np.where(rng.random(n) < 0.7, sign, -sign)
        ty, tx = rng.uniform(-0.12, 0.12, n), rng.uniform(-0.12, 0.12, n)
        A = mag[lab] + ty[lab] * (Y - pts[lab, 0]) + tx[lab] * (X - pts[lab, 1])
        soft = rng.random(n) < rng.uniform(0.3, 0.7)
        A = np.where(soft[lab], ndimage.gaussian_filter(A, 2.0), A)
        A = np.sign(A) * np.clip(np.abs(A), dmin, dmax)
    elif fam == "fibre streaks":
        th = rng.choice([0.0, np.pi / 2]) + rng.uniform(-0.3, 0.3)
        st = rng.uniform(3, 8)
        u = (X * np.cos(th) + Y * np.sin(th)) / st
        v = -X * np.sin(th) + Y * np.cos(th)
        A = _span(F[0].hybrid(v, u, offset=rng.uniform(0.6, 0.9)), dmin, dmax)
        for _ in range(rng.integers(4, 12)):
            cy, cx, rad = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(3, 12)
            A = A + rng.uniform(-3, 3) * np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * rad ** 2))
        A = np.clip(A, dmin, dmax)
    else:
        A = _span(F[0].ridged(Y, X, offset=rng.uniform(0.8, 1.1), gain=rng.uniform(1.5, 2.5)), dmin, dmax)
    if fam != "fragment plates":
        A = sign * A
    ell = np.clip(np.exp(np.log(rng.uniform(3, 7)) + 1.8 * F[1].fbm(Y, X)), 2.0, 14.0)
    rough = Fbm(rng, H, W, scale0=rng.uniform(4, 9), octaves=4).fbm(Y, X, h=0.2)
    rough = rough / max(float(rough.std()), 1e-6) * np.exp(2.7 * F[2].fbm(Y, X) - 0.4)
    return A, ell, rough
