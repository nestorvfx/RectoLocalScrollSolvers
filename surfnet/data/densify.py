"""Dense geometry for meshes whose official copy on the processing volume is coarse.

PHerc0139's GT meshes registered to the 9.362 um volume have ~187 um cells (20 voxels); the same meshes registered to
the 2.399 um scan have ~48 um cells. If the 9.362 um grid is the 2.399 um grid subsampled by an integer factor (the
grid shapes say so), the official transform between the frames is recovered as the least-squares affine of the
corresponding cells and applied to the dense mesh. The fit is accepted only if its residual is < `max_rms` voxels
(else the caller falls back to the coarse mesh, upsampled)."""
from __future__ import annotations

import numpy as np


def correspondence_factor(dense_shape, coarse_shape) -> tuple[int, int] | None:
    """Integer factors f with coarse = dense[::f] along each axis (shape-consistent; the affine residual decides)."""
    out = []
    for n, m in zip(dense_shape, coarse_shape):
        fs = [f for f in range(2, 17) if (n - 1) // f + 1 == m or n // f == m]
        if not fs:
            return None
        out.append(fs[0])
    return out[0], out[1]


def fit_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares affine dst ~ A @ [src, 1]; returns A [3, 4] and the rms residual."""
    X = np.concatenate([src, np.ones((len(src), 1))], 1)
    A, *_ = np.linalg.lstsq(X, dst, rcond=None)
    res = X @ A - dst
    return A.T, float(np.sqrt((res ** 2).sum(1).mean()))


def densify(P_dense: np.ndarray, V_dense: np.ndarray, P_coarse: np.ndarray, V_coarse: np.ndarray,
            max_rms: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Dense mesh mapped into the coarse mesh's frame, or None when the grids do not correspond."""
    f = correspondence_factor(V_dense.shape, V_coarse.shape)
    if f is None:
        return None
    sub = P_dense[::f[0], ::f[1]]
    vs = V_dense[::f[0], ::f[1]]
    h = min(sub.shape[0], P_coarse.shape[0])
    w = min(sub.shape[1], P_coarse.shape[1])
    ok = vs[:h, :w] & V_coarse[:h, :w]
    if ok.sum() < 24:
        return None
    A, rms = fit_affine(sub[:h, :w][ok], P_coarse[:h, :w][ok])
    if rms > max_rms:
        return None
    Pd = P_dense @ A[:, :3].T + A[:, 3]
    return Pd, V_dense.copy(), {"factor": list(f), "rms_vox": rms, "affine": A.tolist()}
