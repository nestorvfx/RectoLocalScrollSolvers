"""`surfnet onsheet`: an untrained per-cell check that an official mesh lies ON the papyrus. The fine model's targets
and verified prompts use only cells that pass.

Per cell (3 x 3-pixel mean of the fine band, depths around the object's face offset):
  S0   = the CT mean over k -2..+2 at the mesh (what a render shows);
  Smax = the brightest 5-vox slab within k -6..+6;
  r    = S0 / Smax (~1 on the papyrus; low when the mesh sits in a gap while papyrus lies within 6 vox).
The ratio is averaged over a `window` x `window`-cell neighbourhood (the official meshes' offsets come as blotches,
not single cells), and a cell fails below a threshold calibrated on verified cells: their `fp` quantile (1 %: 1 % of
verified cells would fail; the recorded build calibrated on hand-verified PHerc0211 cells: 0.771).

Measured on the recorded build (5 x 5, fp 1 %):
- approved surfaces displaced by +3 / +4 vox (into the gap): 30 / 33 % of cells fail; by -3 / -4 (into the papyrus
  body): 4 / 8 %. A conservative filter: it removes the clear offsets, not every offset;
- flagged official cells have the papyrus 3-6 vox from the mesh (none are holes);
- official cells failing: PHerc0139 ~47 %, PHerc1667 ~49 %, segments 29-38 %.
Verified (tier-1) cells always pass. Writes geom `onsheet` (bool) and meta `onsheet`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from .store import BandStore

H = 8                    # depths k -8..+8 around the face offset: 5-vox slabs centred at -6..+6


def ratio_map(store: BandStore, workers: int = 8) -> np.ndarray:
    """Per-cell S0 / Smax [H, W] (NaN where the cell has no CT or its tile is not built)."""
    T, p, h0 = store.spec.tile, store.spec.ppc, store.spec.fine_half
    f0 = int(round(float(store.meta.get("face_offset_vox", 0.0))))
    lo, hi = h0 + f0 - H, h0 + f0 + H + 1
    out = np.full((store.nty * T, store.ntx * T), np.nan, np.float32)

    def tile(t):
        ty, tx = t
        f = np.asarray(store.fine[ty, tx, lo:hi], np.float32).reshape(2 * H + 1, T, p, T, p).mean((2, 4))
        cs = np.cumsum(np.pad(f, ((1, 0), (0, 0), (0, 0))), 0)
        sl = (cs[5:] - cs[:-5]) / 5                               # slab centres k = -H+2 .. H-2
        r = sl[H - 2] / np.maximum(sl.max(0), 1e-3)
        out[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] = np.where((f > 0).all(0), r, np.nan)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(tile, [tuple(t) for t in np.argwhere(store.done)]))
    return out[:store.H, :store.W]


def smoothed(r: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Mean of r over the valid cells of a window x window neighbourhood."""
    ok = valid & np.isfinite(r)
    num = ndimage.uniform_filter(np.where(ok, r, 0).astype(np.float64), window)
    den = ndimage.uniform_filter(ok.astype(np.float64), window)
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)


def calibrate(ref: BandStore, window: int, fp: float) -> float:
    """Threshold = the fp-quantile of the smoothed ratio on the reference's approved (tier-1, training) cells."""
    g = ref.geom
    green = g["V"] & (g["tier"] == 1) & (g["split"] == 0)
    rs = smoothed(ratio_map(ref), g["V"] & (g["split"] != 2), window)
    v = rs[green & np.isfinite(rs)]
    if v.size < 1000:
        raise SystemExit(f"onsheet: only {v.size} approved cells with bands on {ref.meta['id']}")
    return float(np.quantile(v, fp))


def run(stores: list[BandStore], ref_id: str, log, window: int = 5, fp: float = 0.01,
        thr: float | None = None) -> dict:
    if thr is None:
        ref = next((s for s in stores if s.meta["id"] == ref_id), None)
        if ref is None:
            raise SystemExit(f"onsheet: reference {ref_id} not found")
        thr = calibrate(ref, window, fp)
        log.info(f"onsheet: threshold {thr:.3f} = the {100 * fp:.0f} % quantile on the approved cells of {ref_id} "
                 f"({window} x {window} cells)")
    out = {}
    for s in stores:
        g = s.geom
        V = g["V"] & (g["split"] != 2)
        rs = smoothed(ratio_map(s), V, window)
        ok = ~(rs < thr) | (g["tier"] == 1)                    # NaN (no CT) passes here; ctcheck excludes those
        tgt = V & (g["tier"] > 0)
        share = float(ok[tgt].mean()) if tgt.any() else 1.0
        s.update_geom(onsheet=ok)
        s.update_meta(onsheet={"pass_share": round(share, 4), "threshold": round(thr, 4), "window": window,
                               "fp_on_approved": fp, "reference": ref_id})
        out[s.meta["id"]] = share
        log.info(f"onsheet {s.meta['id']}: {100 * share:.1f} % of target cells on the papyrus")
    return out
