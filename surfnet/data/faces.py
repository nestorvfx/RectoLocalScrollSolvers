"""Mesh-convention check (`surfnet faces`): do every object's normals point to the recto side, and where does its mesh
sit on the papyrus relative to a hand-verified reference sheet?

1. Orientation. A papyrus layer is a near-symmetric bright band, so the CT intensity profile cannot tell recto from
   verso (a reversed profile shifted by the band width matches as well). The official meshes follow VC3D's fixed grid
   handedness (rows along +z, text readable from the recto), so the orientation is learnt where it is certain and
   applied where it is not:
   - "trusted" objects (multi-turn GT windings, the merged unwrap, verified sheets: the scroll axis is known) give the sign
     between their raw grid normal (d/drow x d/dcol) and the recto direction; it must agree across all of them;
   - every other object (single segments, oriented at prep by a rough occupancy axis) gets that sign; if its normals
     disagree, its bands are reversed along depth in place and its normals negated (= reversed normals).
2. Face offset. With the orientation fixed, the mean z-scored fine-band profile (depth -12..12 vox) of each object is
   matched to the reference's verified-cell profile over shifts of +-5 vox (less than half a winding: unambiguous). The
   shift s (the reference recto face lies at band depth s in this object) goes to meta `face_offset_vox`; the sampler
   places the true surface there (`data/sampler.py`). Low correlations are reported for review."""
from __future__ import annotations

import numpy as np
import torch

from ..common.geometry import as_t, grid_normals
from .store import BandStore


def native_sign(P: np.ndarray, N: np.ndarray, V: np.ndarray) -> int:
    """+1 if the stored normals equal the raw grid normal (d/drow x d/dcol), -1 if opposite. Grids over 20 M cells are
    sub-sampled 4x per axis (the sign is the same; full float64 normals of a 71 M-cell grid exceed a 30 GB box)."""
    if P.shape[0] * P.shape[1] > 20_000_000:
        P, N, V = P[::4, ::4], N[::4, ::4], V[::4, ::4]
    raw, ok = grid_normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool))
    ok = ok.numpy()
    d = (raw.numpy() * N).sum(-1)[ok]
    return 1 if d.size == 0 or float(np.mean(d)) >= 0 else -1


def mean_profile(store: BandStore, half: int = 12, n: int = 4000, seed: int = 0, tier_max: int = 3,
                 only_tier: int | None = None) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    g = store.geom
    T = store.spec.tile
    p = store.spec.ppc
    h0 = store.spec.fine_half
    m = g["V"] & (g["tier"] > 0) & (g["tier"] <= tier_max)
    if only_tier is not None:
        m &= g["tier"] == only_tier
    cells = np.argwhere(m)
    if len(cells) < 50:
        return None
    pick = cells[rng.choice(len(cells), min(n, len(cells)), replace=False)]
    prof = []
    for r, c in pick:
        ty, tx = r // T, c // T
        if not store.done[ty, tx]:
            continue
        v = store.fine[ty, tx, h0 - half:h0 + half + 1, p * (r % T) + p // 2, p * (c % T) + p // 2].astype(np.float32)
        if (v > 0).mean() < 0.95 or v.std() < 1e-3:
            continue
        prof.append((v - v.mean()) / v.std())
    if len(prof) < 50:
        return None
    return np.mean(prof, 0)


def best_shift(prof: np.ndarray, ref: np.ndarray, max_shift: float = 5.0) -> dict:
    """Shift s (vox) with prof(d) ~ ref(d - s): the reference feature at depth 0 sits at depth s in this object."""
    half = (len(ref) - 1) // 2
    d = np.arange(-half, half + 1)
    best = {"corr": -2.0, "shift": 0.0}
    for s in np.arange(-max_shift, max_shift + 0.01, 0.25):
        r_s = np.interp(d - s, d, ref, left=np.nan, right=np.nan)
        ok = np.isfinite(r_s)
        if ok.sum() < len(d) // 2:
            continue
        c = np.corrcoef(prof[ok], r_s[ok])[0, 1]
        if c > best["corr"]:
            best = {"corr": float(c), "shift": float(s)}
    return best


def flip_store(store: BandStore) -> None:
    """Reverse the depth axis of the bands in place and negate normals / neighbour offsets (= reversed normals)."""
    for arr in (store.fine, store.coarse):
        for ty in range(store.nty):
            for tx in range(store.ntx):
                if store.done[ty, tx]:
                    arr[ty, tx] = arr[ty, tx, ::-1].copy()
        arr.flush()
    g = dict(store.geom)
    upd = {"N": -g["N"]}
    if "nbr" in g:
        upd["nbr"] = (-g["nbr"].astype(np.float32)).astype(np.float16)
    store.update_geom(**upd)


def orient(stores: list[BandStore], log, apply: bool = True) -> int | None:
    """Learn the native-handedness sign from trusted objects and apply it to the others. Returns the sign."""
    signs = {}
    for s in stores:
        g = s.geom
        sg = native_sign(g["P"].astype(np.float64), g["N"].astype(np.float64), g["V"])
        s.update_meta(native_sign=sg)
        signs[s.meta["id"]] = sg
    trusted = [s for s in stores if s.meta.get("axis_trusted")]
    if not trusted:
        log.info("orient: no trusted object; orientation left as built")
        return None
    ts = [signs[s.meta["id"]] for s in trusted]
    s0 = 1 if sum(ts) >= 0 else -1
    agree = sum(t == s0 for t in ts)
    log.info(f"orient: trusted objects agree on native sign {s0:+d} in {agree}/{len(ts)}")
    if agree < len(ts):
        log.info("orient: trusted objects DISAGREE: the handedness is not consistent; segments left as built (review)")
        return None
    for s in stores:
        if s.meta.get("axis_trusted"):
            continue
        if signs[s.meta["id"]] != s0:
            log.info(f"orient {s.meta['id']}: normals reversed to the recto side (native sign {s0:+d})")
            if apply:
                flip_store(s)
                s.update_meta(native_sign=s0, flipped=True)
        s.update_meta(oriented_by="native handedness of the trusted objects")
    return s0


def run(stores: list[BandStore], ref_id: str, log, min_corr: float = 0.6, apply: bool = True,
        ref_tier: int | None = 1) -> dict:
    orient(stores, log, apply)
    ref_store = next((s for s in stores if s.meta["id"] == ref_id), None)
    if ref_store is None:
        raise SystemExit(f"reference object {ref_id} has no bands")
    ref = mean_profile(ref_store, only_tier=ref_tier)
    if ref is None:
        raise SystemExit("reference profile: not enough cells with bands")
    # the recto face on the reference: the strongest bright-to-dark step toward +depth within +-6 vox (the growth
    # front's untrained gate compares the corrected surfaces with it)
    half = (len(ref) - 1) // 2
    d = np.arange(-half, half + 1)
    step = np.array([ref[max(0, i - 3):i].mean() - ref[i:i + 3].mean() if 3 <= i <= len(ref) - 3 else -9
                     for i in range(len(ref))])
    step[np.abs(d - 0.5) > 6] = -9
    face_pos = float(d[int(np.argmax(step))] - 0.5)
    log.info(f"faces reference {ref_id}: recto face at {face_pos:+.1f} vox of the approved surface")
    out = {}
    for s in stores:
        if s.meta["id"] == ref_id:
            s.update_meta(face_offset_vox=0.0, face_ref=True, face_pos_vox=face_pos)
            continue
        prof = mean_profile(s)
        if prof is None:
            log.info(f"faces {s.meta['id']}: no profile (bands missing?)")
            continue
        b = best_shift(prof, ref)
        out[s.meta["id"]] = b
        flag = "" if b["corr"] >= min_corr else "  << LOW CORRELATION: review"
        log.info(f"faces {s.meta['id']}: face offset {b['shift']:+.2f} vox, corr {b['corr']:.3f}{flag}")
        if apply and b["corr"] >= min_corr:
            s.update_meta(face_offset_vox=round(b["shift"] * 4) / 4, face_corr=b["corr"])
    return out
