"""Correct a region of a sheet from the cells known to be right (`surfnet correct`).

Inputs, all on the grid of one tifxyz surface:
- known cells: verified correct; pinned at their position and shown to the model as the green prompt;
- solve cells: the cells to correct (default: every valid cell that is not known);
- optional rejected cells: known to be wrong (never used as context).

The solve cells and the known cells within `margin` cells of them are solved together in one pass (the run region);
with the draft start, other valid cells within `ahead` cells are solved as unknown context but not kept.

Starts:
- draft (default): the sheet's own solve cells, smoothed (sigma 6 cells), plus one natural perturbation per run
  (`perturb.natural_field`, |A| dmin-dmax vox along the normals, growing from 0 at the known cells, clipped at
  +-clip); `runs` runs; the result is the per-cell median, the spread is max - min over the runs;
- bridge: the solve cells are emptied and bridged from the known cells only (`surface.thin_plate_fill`), then the same
  perturbed runs; nothing of the draft enters (ahead = 0).

Reach: from the known cells alone (bridge), SurfNet was reliable on the cells within 40 cells (1.9 mm) of a known cell,
e.g. a hole up to 3.8 mm across in a ring of known cells: there it beat GrowPatch in the benchmark. Farther cells can
follow a neighbouring winding (README, Which to use); the tool warns. Such areas are grown with a tracer first
(`surfnet grow`) and its surface corrected as the draft.

Each run is `engine.run_sheet`: locate (context stream, BP), then refinement passes (fine stream, BP), every frame a
low-pass of the current estimate; both average `tta` of the 8 flips / rotations of each window (2 by default).

Ways of running it (`apply`): `FAST` is one fast application (4 runs, 1 flip, 1 refinement pass, the BP iteration
compiled on a GPU, no confidence maps); `resolve` then solves again only the cells whose runs differ by more than half
a winding (`targeted`: 8 runs, the other cells pinned as known); `keep` then keeps the draft on the cells whose runs
still differ by more than half a winding (`keep_draft`).

The surface is resampled to the model's cell (47.46 um; its step measured on the known cells, resampled when more than
GRID_TOL off) at the processing level of the volume (the level closest to 9.4 um) and written back in the volume's
level-0 voxels, on the model's grid."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from ..common import frames as GM
from ..common import tifxyz
from ..common.geometry import Axis, as_t, axis_direction, grid_steps, resample_grid
from ..data.objects import level_coords
from ..data.sheet import CELL_UM, processing_level, read_mask
from ..model import bins_fine
from ..volume import LevelOffset, open_volume
from . import engine as ENG
from .engine import Inference, Volume, load_model, run_sheet
from .perturb import FAMILIES, natural_field
from .surface import normals, smooth_surface, thin_plate_fill


REACH = 40.0                     # cells (1.9 mm): from the known cells alone SurfNet was reliable within this distance of
                                 # them (holes up to 3.8 mm); farther it can follow a neighbouring winding (README)
FAST = {"runs": 4, "tta": 1, "passes": 1, "confidence": False, "bp_compiled": True}   # one fast application
HALF_WINDING = 6.25              # vox: the targeted re-solve takes the cells whose runs spread more than this
RESOLVE_RUNS = 8
GRID_TOL = 0.05                  # a sheet within 5 % of the model's cell is solved on its own grid: the benchmark's cases
                                 # lie up to 3.7 % from it (on their known cells), and training varied the scale by 15 %


@dataclass
class Settings:
    start: str = "draft"          # draft | bridge
    runs: int = 5                 # perturbed starts; the per-cell median is kept (an even count: the middle two's mean)
    tta: int = 2                  # flips / rotations averaged per window (the benchmark's default solve; 4 scored the same)
    passes: int = 3
    early_stop: bool = True       # stop a run's passes once its frame moves < 0.25 vox (p99)
    confidence: bool = True       # the confidence maps of the result and the input (two more passes; not the surface)
    bp_compiled: bool = False     # the BP iteration through torch.compile on a GPU (fused kernels; the same messages)
    margin: float = 64.0          # known cells within this many cells of the solve cells join the run
    ahead: float = 40.0           # other valid cells within this many cells: solved as context, not kept (draft only)
    dmin: float = 4.0             # perturbation amplitude range (vox)
    dmax: float = 20.0
    clip: float = 20.0
    noise: float = 0.6
    seed: int = 21


def frame_probs(inf, spec, Vr, known_t, P_known, P_s, axis):
    """The model's probabilities along surface P_s's own low-pass frame: fine [nf, H, W] (bins_fine) and context
    [nc, H, W] (bins_ctx); P_s's offset from that frame (vox); the cells the windows covered."""
    Fp = GM.blur_masked(P_s.permute(2, 0, 1).float(), Vr, 6.0).permute(1, 2, 0).double()
    Fp = torch.where(Vr[..., None], Fp, P_s)
    Fn = normals(Fp, Vr, axis)
    g = ((P_known - Fp) * Fn).sum(-1).float()
    nan = torch.full_like(g, float("nan"))
    so = [torch.where(known_t, g, nan), nan, nan]
    sm = [known_t, torch.zeros_like(Vr), torch.zeros_like(Vr)]
    ax = torch.as_tensor(axis_direction(P_s.cpu().numpy(), Vr.cpu().numpy()), device=Vr.device, dtype=torch.float32)
    pf, pc, cov = inf.refine(Fp, Fn, Vr, so, sm, ax)
    return pf[:-1], pc, ((P_s - Fp) * Fn).sum(-1).float(), cov


def confidence(inf, spec, Vr, known_t, P_known, P_s, axis):
    """Per cell: the fine head's probability mass within 1 vox of surface P_s, read along P_s's own low-pass frame."""
    pf, _, off, cov = frame_probs(inf, spec, Vr, known_t, P_known, P_s, axis)
    bf = bins_fine(spec, Vr.device).view(-1, 1, 1)
    return torch.where(cov, (((bf - off[None]).abs() <= 1.0).float() * pf).sum(0), torch.zeros_like(off))


def run_region(V0, known, solve, s: Settings):
    """The cells of a joint solve: the solve cells, the known cells within `margin` of them and other valid cells within
    `ahead`; and its bounding box (+4 cells; BP memory ~ its area)."""
    H, W = V0.shape
    dsol = ndimage.distance_transform_edt(~solve)
    run_np = V0 & (solve | (known & (dsol <= s.margin)) | (~known & ~solve & (dsol <= s.ahead)))
    rr_, cc_ = np.nonzero(run_np)
    return run_np, (slice(max(0, rr_.min() - 4), min(H, rr_.max() + 5)), slice(max(0, cc_.min() - 4), min(W, cc_.max() + 5)))


def confidence_maps(inf, spec, axis, V0, P_in, P_out, known, solve, s: Settings):
    """The confidence of the result P_out and of the input P_in over the run region (`confidence`)."""
    dev = inf.dev
    run_np, cs = run_region(V0, known, solve, s)
    Vr, known_r = as_t(run_np, dev, torch.bool), as_t(known & run_np, dev, torch.bool)
    P_ref, P_s = as_t(P_in, dev, torch.float64), as_t(P_out, dev, torch.float64)
    c_out, c_in = torch.zeros(V0.shape, device=dev), torch.zeros(V0.shape, device=dev)
    c_out[cs] = confidence(inf, spec, Vr[cs], known_r[cs], P_ref[cs], P_s[cs], axis).float()
    c_in[cs] = confidence(inf, spec, Vr[cs], known_r[cs], P_ref[cs], P_ref[cs], axis).float()
    return c_out.cpu().numpy(), c_in.cpu().numpy()


def joint_solve(inf, spec, axis, V0, P_np, known, solve, s: Settings, rng, log=None):
    """The solve cells (bool [H, W]) solved together, the known cells pinned at P_np. -> dict of per-cell maps (move
    along the sheet's normals in vox, spread, confidence of the result and of the input) and the result P_out."""
    dev = inf.dev
    H, W = V0.shape
    run_np, cs = run_region(V0, known, solve, s)
    unv = run_np & ~known
    V, Vr, known_r = as_t(V0, dev, torch.bool), as_t(run_np, dev, torch.bool), as_t(known & run_np, dev, torch.bool)
    P_ref = as_t(P_np, dev, torch.float64)
    N_ref = normals(P_ref, V, axis)
    S = as_t(np.where(known[..., None], P_np, smooth_surface(P_np, V0, 6.0)), dev, torch.float64)
    N_S = normals(S, V, axis)
    dist = ndimage.distance_transform_edt(~known) if known.any() else np.full((H, W), 1e6)
    ENG.TIMES.clear()
    t_all = time.time()
    # the CT the windows can need, downloaded in one batch per level before they run (not a few chunks per window):
    # level 0 within the perturbation, the locate step's reach and the fine stack of the start; level 1 within the
    # context stack's +-96 vox beyond that
    reach0 = s.clip + float(inf.sp.cout) + float(inf.sp.fk) + 8.0
    n_keys = inf.vol.prefetch(S, N_S, Vr, reach0, reach0 + 2.0 * float(inf.sp.ck)) if hasattr(inf.vol, "prefetch") \
        else 0
    outs = []
    # every run's perturbation field (CPU, ~0.7 s each on a 400-cell area) made in a background thread while the GPU
    # works; one worker, called in run order, draws from rng in the same sequence as calling them one by one
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)
    fields = [pool.submit(natural_field, FAMILIES[k % len(FAMILIES)], H, W, rng, s.dmin, s.dmax,
                          1.0 if k % 2 == 0 else -1.0) for k in range(s.runs)]
    pool.shutdown(wait=False)
    for k in range(s.runs):
        fam = FAMILIES[k % len(FAMILIES)]
        t0 = time.time()
        A, ell, rough = fields[k].result()
        T = np.where(unv, np.clip((1 - np.exp(-dist / ell)) * (A + s.noise * rough), -s.clip, s.clip), 0.0)
        st_ = S + as_t(T, dev, torch.float64)[..., None] * N_S
        r = run_sheet(inf, None, Vr[cs], known_r[cs], P_ref[cs], passes=s.passes, start=st_[cs],
                      early_stop=s.early_stop, bp_compiled=s.bp_compiled)
        o_ = torch.zeros((H, W), device=dev, dtype=torch.float64)
        o_[cs] = ((r["P"] - P_ref[cs]) * N_ref[cs]).sum(-1)
        outs.append(o_)
        if log:
            log(f"run {k + 1}/{s.runs} ({fam}) done in {time.time() - t0:.0f} s")
        if dev != "cpu":
            torch.cuda.empty_cache()
    O = torch.stack(outs)
    avg = O.median(0).values if len(outs) % 2 else torch.quantile(O, 0.5, dim=0)
    spread = O.max(0).values - O.min(0).values
    sol = as_t(solve, dev, torch.bool)
    P_out = torch.where(sol[..., None], P_ref + avg[..., None] * N_ref, P_ref)
    c_out = c_in = np.zeros((H, W), np.float32)
    if s.confidence:
        c_out, c_in = confidence_maps(inf, spec, axis, V0, P_np, P_out.cpu().numpy(), known, solve, s)
    f = lambda t: t.float().cpu().numpy()                                       # noqa: E731
    times = {k: round(v, 1) for k, v in ENG.TIMES.items()}
    times["total"] = round(time.time() - t_all, 1)
    times["prefetch_chunks"] = n_keys
    if log:
        log(f"times: {times}")
    return {"times": times, "move": np.where(solve, f(avg), 0.0), "spread": np.where(solve, f(spread), 0.0),
            "confidence": c_out, "confidence_input": c_in, "P_out": P_out.cpu().numpy(),
            "runs": f(O)}                                                  # every run's move along the input's normals


def targeted(inf, spec, axis, V0, P, spread, known, solve, s: Settings, rng, tau: float = HALF_WINDING, log=None):
    """The targeted re-solve: only the solve cells whose runs disagreed (spread > tau vox) solved again, the cells they
    agreed on pinned as known context around them. -> joint_solve's dict ("targeted": the cells solved again; the
    spread of the other cells kept)."""
    U = solve & (spread > tau)
    if not U.any():
        return {"P_out": P.copy(), "spread": spread, "times": {}, "targeted": 0}
    r = joint_solve(inf, spec, axis, V0, P, known | (solve & ~U), U, s, rng, log)
    r["targeted"] = int(U.sum())
    r["spread"] = np.where(U, r["spread"], spread)
    return r


def keep_draft(P, P_out, spread, solve, tau: float = HALF_WINDING):
    """The input's position kept on the solve cells whose runs still differ by more than tau vox (the spread as kept
    between steps, float16): where SurfNet's runs disagree, the draft (e.g. GrowPatch's surface) stays. -> P_out, the
    kept cells."""
    U = solve & (np.asarray(spread).astype(np.float16).astype(np.float64) > tau)
    return np.where(U[..., None], P, P_out), U


def apply(inf, spec, axis, V0, P, known, solve, s: Settings, resolve: bool = False, keep: bool = False,
          log=None) -> dict:
    """One application of the model, then (resolve) the targeted re-solve with RESOLVE_RUNS runs, then (keep) the
    input kept where the last step's runs still disagree (`keep_draft`); every step from the same seed. Between steps
    the surface is kept in float32 and the spread in float16, as the benchmark kept them, so the steps compute what it
    scored. -> joint_solve's dict for the whole: the move along the input's normals, the last spread, the confidence of
    the result and of the input (s.confidence)."""
    kept = lambda a, dt: a.astype(dt).astype(np.float64)                         # noqa: E731
    if not resolve and not keep:
        return joint_solve(inf, spec, axis, V0, P, known, solve, s, np.random.default_rng(s.seed), log)
    s1 = replace(s, confidence=False)
    r = joint_solve(inf, spec, axis, V0, P, known, solve, s1, np.random.default_rng(s.seed), log)
    times = [r["times"]]
    if resolve:
        r = targeted(inf, spec, axis, V0, kept(r["P_out"], np.float32), kept(r["spread"], np.float16), known, solve,
                     replace(s1, runs=RESOLVE_RUNS), np.random.default_rng(s.seed), log=log)
        times.append(r["times"])
    P_out, kept_cells = r["P_out"], None
    if keep:
        P_out, U = keep_draft(P, P_out, r["spread"], solve)
        kept_cells = int(U.sum())
    N = normals(as_t(P, "cpu", torch.float64), as_t(V0, "cpu", torch.bool), axis).numpy()
    c_out = c_in = np.zeros(V0.shape, np.float32)
    if s.confidence:
        c_out, c_in = confidence_maps(inf, spec, axis, V0, P, P_out, known, solve, s)
    return {"P_out": P_out, "move": np.where(solve, ((P_out - P) * N).sum(-1), 0.0).astype(np.float32),
            "spread": np.where(solve, r["spread"], 0.0).astype(np.float32), "confidence": c_out,
            "confidence_input": c_in, "times": times, "targeted": r.get("targeted"), "kept": kept_cells}


def bridge(P, known, solve):
    """The solve cells bridged from the known cells' positions only (surface.thin_plate_fill), kept in float32: the
    start of --start bridge."""
    return thin_plate_fill(P, known, solve).astype(np.float32).astype(np.float64)


def _to_model_grid(P, V, masks: dict, cell_vox: float, known=None, tol: float = 0.02):
    """Resample a grid (and its masks, nearest) to `cell_vox` voxels per cell; no-op within `tol` (2 %). The grid's step
    is measured on the known cells (`geometry.grid_steps`; on all valid cells if none are given): a draft's area may be
    stretched or compressed, and must not decide the resampling."""
    dr, dc = grid_steps(P, V, V if known is None else known)
    sr, sc = cell_vox / dr, cell_vox / dc
    if abs(sr - 1) <= tol and abs(sc - 1) <= tol:
        return P, V, masks, (1.0, 1.0)
    Pt, Vt = resample_grid(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), sr, sc)
    H, W = Vt.shape
    r = np.clip(np.rint(np.arange(H) * sr).astype(int), 0, V.shape[0] - 1)
    c = np.clip(np.rint(np.arange(W) * sc).astype(int), 0, V.shape[1] - 1)
    out = {k: (m[r[:, None], c[None, :]] if m is not None else None) for k, m in masks.items()}
    return Pt.numpy(), Vt.numpy(), out, (sr, sc)


def correct_sheet(ckpt: str, sheet: str, volume: str, pitch_um: float, known_mask: str, out: str, *,
                  solve_mask: str | None = None, level: int | None = None, axis: str | None = None,
                  flip: bool = False, settings: Settings | None = None, resolve: bool = False, keep: bool = False,
                  cache: str | None = None, device: str = "cuda", log=print) -> dict:
    s = settings or Settings()
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    P0, V0, meta = tifxyz.read(sheet)
    known0 = read_mask(known_mask, V0.shape) & V0
    solve0 = read_mask(solve_mask, V0.shape) if solve_mask else V0.copy()
    level = processing_level(pitch_um) if level is None else int(level)
    cell_vox = CELL_UM / (pitch_um * 2 ** level)
    P, V, m, steps = _to_model_grid(level_coords(P0, level), V0, {"known": known0, "solve": solve0}, cell_vox, known0,
                                    GRID_TOL)
    known = m["known"] & V
    # bridge: the solve cells need no position (a hole may be empty); known cells the resampling could not place are
    # solved with them. draft: the solve cells are the draft's valid cells.
    solve = ((m["solve"] | m["known"]) if s.start == "bridge" else m["solve"] & V) & ~known
    if not known.any() or not solve.any():
        raise SystemExit(f"known cells {int(known.sum())}, solve cells {int(solve.sum())}: both must be non-empty")
    ax = None
    if axis:
        a0 = Axis.from_json(axis)
        ax = Axis(*(level_coords(v, level) for v in (a0.z, a0.y, a0.x)))
    far = ndimage.distance_transform_edt(~known)[solve] > REACH
    if s.start == "bridge":                                    # the solve cells emptied, bridged from the known cells
        P = bridge(P, known, solve)
        V = known | solve
        s.ahead = 0.0
        if far.any():
            log(f"WARNING: {far.mean():.0%} of the solve cells are more than {REACH:.0f} cells ({REACH * CELL_UM / 1000:.1f} "
                f"mm) from every known cell, beyond the distance at which SurfNet alone was reliable: there it can follow "
                f"a neighbouring winding. Grow the area with a tracer (e.g. GrowPatch) first and correct its surface as "
                f"the draft (surfnet grow, then --start draft); see the README, Which to use.")
    log(f"sheet {P.shape[0]} x {P.shape[1]} cells (model grid; resampled {steps[0]:.2f} x {steps[1]:.2f}), level "
        f"{level}, known {int(known.sum())}, solve {int(solve.sum())}; start {s.start}, runs {s.runs}, tta {s.tta}, "
        f"passes {s.passes}{', then the targeted re-solve' if resolve else ''}"
        f"{', the draft kept where the runs disagree' if keep else ''}")
    model, spec = load_model(ckpt, device)
    src = open_volume(volume, cache or (Path(outd) / "cache"))
    if level:
        src = LevelOffset(src, level)
    inf = Inference(model, spec, Volume(src, device, 2500, 800), ax, device, tta=s.tta)
    if flip:                                                    # reversed grid handedness: flip the normals
        rev = lambda a: a[::-1].copy()                          # noqa: E731
        r = apply(inf, spec, ax, rev(V), rev(P), rev(known), rev(solve), s, resolve, keep=keep, log=log)
        r = {k: rev(v) if isinstance(v, np.ndarray) else v for k, v in r.items() if k != "runs"}   # (unused here)
    else:
        r = apply(inf, spec, ax, V, P, known, solve, s, resolve, keep=keep, log=log)
    f = 2 ** level
    P_out0 = r["P_out"] * f + (f - 1) / 2.0                     # back to the volume's level-0 voxels
    rep = {"ckpt": str(ckpt), "sheet": str(sheet), "volume": str(volume), "level": level, "pitch_um": pitch_um,
           "settings": asdict(s), "resolve": resolve, "resolved_cells": r.get("targeted"),
           "keep_draft": keep, "kept_cells": r.get("kept"),
           "known_cells": int(known.sum()), "solve_cells": int(solve.sum()),
           "solve_beyond_reach": round(float(far.mean()), 4),
           "move_p50_p90": [float(np.median(np.abs(r["move"][solve]))), float(np.quantile(np.abs(r["move"][solve]), 0.9))],
           "spread_p50_p90": [float(np.median(r["spread"][solve])), float(np.quantile(r["spread"][solve], 0.9))],
           "times": r["times"]}
    if s.confidence:
        rep.update(confidence_input=float(np.mean(r["confidence_input"][solve])),
                   confidence_output=float(np.mean(r["confidence"][solve])))
    tifxyz.write(outd / "surface", P_out0, V, dict({k: v for k, v in meta.items() if k not in ("bbox",)},
                                                     scale=[1.0 / (cell_vox * f)] * 2, surfnet=rep), overwrite=True)
    conf = {"confidence": r["confidence"], "confidence_input": r["confidence_input"]} if s.confidence else {}
    np.savez_compressed(outd / "maps.npz", known=known, solve=solve, move=r["move"], spread=r["spread"], **conf)
    (outd / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"move p50 / p90 {rep['move_p50_p90'][0]:.2f} / {rep['move_p50_p90'][1]:.2f} vox, spread p50 / p90 "
        f"{rep['spread_p50_p90'][0]:.2f} / {rep['spread_p50_p90'][1]:.2f} vox"
        + (f", confidence {rep['confidence_input']:.2f} -> {rep['confidence_output']:.2f}" if s.confidence else "")
        + (f", {rep['resolved_cells']} cells solved again" if resolve else "")
        + (f", the draft kept on {rep['kept_cells']} cells" if keep else "") + f"; written {outd / 'surface'}")
    return rep
