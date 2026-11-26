"""SurfNet against the official tracer on held-out ground truth (configs in cases.yaml; docs/SUPPLEMENTARY.md).

  python bench.py cases     --work W                        ground truth, cases                             (CPU)
  python bench.py growpatch --work W --out O --apprun APPRUN GrowPatch as a user runs it (`surfnet grow`)    (CPU)
  python bench.py surfnet   --work W --out O --ckpt M.pt --methods ...   SurfNet's ways (G...: --results with G) (GPU)
  python bench.py score     --work W --results R ... --out O   (--errors O2: from the per-cell errors O2 saved)
  python bench.py report    --scores O

Every stage writes its own results tree (O/<case>/<method>.npz | <method>/ + <method>.json): a stage's output is a
backup that later stages and post-processing read (--results), so nothing is recomputed. Stages skip what exists.

Every method gets what a user has, and nothing of the ground truth outside the known cells: the known cells K of a
ground-truth window (their positions), the target cells S as a mask, the CT and, for GrowPatch, the m7 surface
prediction and its normal grids. The ground truth is used only to score.
  E    thin-plate extrapolation of K (surface.thin_plate_fill: K's positions only; the start of SurfNet alone)
  D    the wrong draft itself (correct cases: the truth moved along its normals, a synthetic draft)
  G    GrowPatch as `surfnet grow` runs it (surfnet/infer/grow.py) from the case's user inputs: the surface it writes
       on the case's grid (grow/surface), as the user gets it
  S5   SurfNet alone: its default solve (5 runs, 2 flips, up to 3 passes) from E (`surfnet correct --start bridge`),
       or from the draft
  GF4Cmed       from G's surface on the case's grid (what `surfnet grow` writes for `surfnet correct`), one fast
                application (correct.FAST: 4 runs, 1 flip, 1 pass, compiled BP; `--mode fast`)
  GF4Cmedx1T    then the targeted re-solve (correct.targeted: 8 runs on the cells whose runs spread > half a winding;
                `--resolve`)
  GF4CmedK, GF4Cmedx1TK   then G's surface kept where the last step's runs still spread > half a winding
                (correct.keep_draft; `--keep-draft`)
The ways are applied one link per method, each from the previous link's saved result (the surface in float32, the
spread in float16, as surfnet correct keeps them between steps), with the functions and settings of surfnet correct,
so every link is scored."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surfnet.common import tifxyz  # noqa: E402
from surfnet.common.geometry import as_t  # noqa: E402
from surfnet.data.sheet import CELL_UM  # noqa: E402
from surfnet.eval.cases import draft_offsets, extent, layout, place_blocks, place_windows  # noqa: E402
from surfnet.eval.score import summary, surface_errors  # noqa: E402
from surfnet.infer.correct import (FAST, RESOLVE_RUNS, Settings, _to_model_grid, bridge, joint_solve,  # noqa: E402
                                   keep_draft, targeted)
from surfnet.infer.surface import normals, thin_plate_fill  # noqa: E402

METHODS = {"fill": ("E", "G", "S5"), "extend": ("E", "G", "S5"), "context": ("E", "G", "S5"),
           "correct": ("D", "E", "S5")}
PREV = {"GF4Cmed": None, "GF4Cmedx1T": "GF4Cmed", "GF4CmedK": "GF4Cmed", "GF4Cmedx1TK": "GF4Cmedx1T"}   # each link's input
WAYS = tuple(PREV)                                        # from GrowPatch's surface: see the module's docstring
RUN_ORDER = ("S5",) + WAYS


# ------------------------------------------------------------------------------------------------ helpers
def load_cfg(work: Path) -> dict:
    return yaml.safe_load((work / "cases.yaml").read_text())


def fetch(cfg, prefix: str, dst: Path, sync: bool = False) -> Path:
    """A bucket prefix (s3://: s5cmd) or a local directory standing in for the bucket (tests) -> dst."""
    src = f"{cfg['bucket']}/{prefix.rstrip('/')}"
    dst.mkdir(parents=True, exist_ok=True)
    if src.startswith("s3://"):
        subprocess.run(["s5cmd", "--no-sign-request", "sync" if sync else "cp", f"{src}/*", f"{dst}/"], check=True,
                       capture_output=True)
    else:
        import shutil
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def cases_of(work: Path, only=None):
    for d in sorted((work / "cases").iterdir()):
        if (d / "case.json").exists() and (not only or d.name.split("__")[0] in only or d.name in only):
            yield d, json.loads((d / "case.json").read_text())


def find_result(results: list[Path], case: str, m: str):
    """A method's output for a case in the first results tree that has it: (P, V) or None. G: the surface `surfnet
    grow` wrote on the case's grid; empty where GrowPatch gave no surface (G.json ok false: the area is then only
    bridged, and the G ways start from that)."""
    for r in results:
        f = r / case / f"{m}.npz"
        if f.exists():
            z = np.load(f)
            return z["P"].astype(np.float64), z["V"]
        if m == "G" and (r / case / "G.json").exists():
            if not json.loads((r / case / "G.json").read_text())["ok"]:
                return np.full((1, 1, 3), -1.0), np.zeros((1, 1), bool)
            Q, VQ, _ = tifxyz.read(r / case / "grow" / "surface")
            return Q, VQ
    return None


def done(results: list[Path], case: str, m: str) -> bool:
    return any((r / case / f"{m}.json").exists() for r in results)


def write_json(p: Path, d: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))


def arrays_of(d: Path) -> dict:
    z = dict(np.load(d / "arrays.npz"))
    for k in ("P", "N", "E", "D"):
        if k in z:
            z[k] = z[k].astype(np.float64)
    return z


# ------------------------------------------------------------------------------------------------ cases
def cmd_cases(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "cases.yaml").exists():
        src = Path(a.config) if a.config else Path(__file__).parent / "cases.yaml"
        (work / "cases.yaml").write_text(src.read_text())
    cfg = load_cfg(work)
    rng = np.random.default_rng(cfg.get("seed", 0))
    for sf in cfg["surfaces"]:
        if a.only and sf["name"] not in a.only:
            continue
        sd = work / "surfaces" / sf["name"]
        gt = sd / "gt"
        if not (gt / "x.tif").exists():
            fetch(cfg, sf["mesh"], gt)
        P0, V0, _ = tifxyz.read(gt)
        cell = CELL_UM / sf["pitch_um"]
        P, V, _, steps = _to_model_grid(P0, V0, {}, cell)                 # every test volume is read at level 0
        N = normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), None).numpy()
        spacing = sf.get("spacing_vox")
        if sf.get("spacing_from"):                                         # median distance to a neighbour winding
            nb = sd / "neighbour"
            if not (nb / "x.tif").exists():
                fetch(cfg, sf["spacing_from"]["mesh"], nb)
            Qn, VQn, _ = tifxyz.read(nb)
            sub = np.zeros_like(V)
            sub[::8, ::8] = True
            res = surface_errors(P, N, V & sub, Qn, VQn, lateral_max=4.0)
            spacing = float(np.median(np.abs(res["error"][res["covered"]]))) / sf["spacing_from"]["windings"]
        spacing = spacing or 13.0 * 9.4 / sf["pitch_um"]
        count = cfg["locations"][sf["set"]]
        big = max(extent(c) for c in cfg["cases"])                       # shared locations: where the largest fits
        centres = [tuple(x) for x in sf["centres"]] if sf.get("centres") else \
            [(r + big // 2, c + big // 2) for r, c in place_blocks(V, big, count, rng)]
        placed = {}
        for ci, case in enumerate(cfg["cases"]):
            wins = place_windows(V, extent(case), count, centres, rng)
            placed[ci] = len(wins)
            for li, o in enumerate(wins):
                build_case(work, sf, spacing, ci, case, li, o, P, V, N, rng)
        write_json(sd / "surface.json", {**sf, "spacing_vox": spacing, "model_grid": list(V.shape),
                                         "resampled": list(steps), "centres": centres, "locations_per_case": placed})
        print(f"{sf['name']}: model grid {V.shape}, spacing {spacing:.2f} vox, {len(centres)} shared locations; "
              f"locations per case {sorted(set(placed.values()))}", flush=True)


def build_case(work, sf, spacing, ci, case, li, o, Pg, Vg, Ng, rng):
    kind = case["kind"]
    name = f"{sf['name']}__{ci:02d}_{case['kind']}_{case['layout']}_k{case['known']}_s{case['size']}" + \
           (f"_{case.get('perturb', 'nat')}{case.get('amp', [4, 11])[1]:g}" if case["kind"] == "correct" else "") + f"__L{li}"
    d = work / "cases" / name
    if (d / "case.json").exists():
        return
    n = extent(case)
    sl = (slice(o[0], o[0] + n), slice(o[1], o[1] + n))
    P, V, N = Pg[sl], Vg[sl], Ng[sl]
    K, S = layout(n, case, rng)
    K &= V
    S &= V
    E = thin_plate_fill(P, K, S)
    arrays = {"P": P.astype(np.float32), "V": V, "N": N.astype(np.float32), "K": K, "S": S, "E": E.astype(np.float32)}
    if case["kind"] == "correct":
        T = draft_offsets(K, S, case, spacing, rng)
        arrays["D"] = np.where(S[..., None], P + T[..., None] * N, P).astype(np.float32)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "arrays.npz", **arrays)
    info = {"name": name, "surface": sf["name"], "set": sf["set"], "kind": kind, "case": case, "location": li,
            "origin": list(o), "extent": n, "spacing_vox": spacing, "pitch_um": sf["pitch_um"],
            "known_cells": int(K.sum()), "target_cells": int(S.sum()),
            "target_mm2": round(float(S.sum()) * (CELL_UM / 1000) ** 2, 2), "methods": list(METHODS[kind])}
    write_json(d / "case.json", info)


# ------------------------------------------------------------------------------------------------ GrowPatch
def user_inputs(d: Path, sf: dict, u: Path):
    """What a user has for a case, as the tools read it: a sheet with the known cells' positions only (every other
    cell empty), the known cells and the area as PNG masks."""
    from PIL import Image
    z = arrays_of(d)
    u.mkdir(parents=True, exist_ok=True)
    tifxyz.write(u / "sheet", np.where(z["K"][..., None], z["P"], -1.0), z["K"], {"scale": [sf["pitch_um"] / CELL_UM] * 2},
                 overwrite=True)
    Image.fromarray(z["K"].astype(np.uint8) * 255).save(u / "known.png")
    Image.fromarray(z["S"].astype(np.uint8) * 255).save(u / "area.png")


def cmd_growpatch(a):
    """GrowPatch as a user runs it: `surfnet grow` (surfnet/infer/grow.py) on each case's user inputs ->
    out/<case>/grow (the inputs; GrowPatch's own surface `growpatch`; its surface on the case's grid `surface`, scored
    as G and the G ways' start) and out/<case>/G.json (end-to-end seconds; ok false where GrowPatch gave no surface: G
    then scores 0 and the G ways start from the bridged area, as `surfnet grow` leaves it)."""
    from surfnet.infer.grow import grow_sheet
    work, out = Path(a.work), Path(a.out)
    cfg = load_cfg(work)
    surf = {s["name"]: s for s in cfg["surfaces"]}
    i, n = map(int, a.part.split("/"))
    for d, c in list(cases_of(work, a.only))[i::n]:
        if "G" not in c["methods"] or done([out], c["name"], "G"):
            continue
        if a.until and time.time() > a.until:
            print("deadline: stopping", flush=True)
            break
        sf = surf[c["surface"]]
        ng = Path(a.data) / Path(sf["normal_grids"]).name
        if not (ng / "metadata.json").exists():
            fetch(cfg, sf["normal_grids"], ng, sync=True)
        od = out / c["name"]
        u = od / "grow"
        user_inputs(d, sf, u)
        t0 = time.time()
        try:
            rep = grow_sheet(str(u / "sheet"), str(u / "known.png"), str(u / "area.png"), f"{cfg['bucket']}/{sf['m7']}",
                             str(ng), sf["pitch_um"], a.apprun, str(u), threads=a.threads, timeout=a.case_timeout,
                             log=lambda *_: None)
            meta = {"seconds": round(time.time() - t0, 1), "ok": not rep["growpatch_failed"],
                    "error": rep["growpatch_failed"], "generations": rep["generations"], "covered": rep["covered"]}
        except SystemExit as e:                                            # the inputs themselves were unusable
            meta = {"seconds": round(time.time() - t0, 1), "ok": False, "error": str(e)[-2000:]}
        write_json(od / "G.json", meta)
        print(f"G {c['name']}: {'ok' if meta['ok'] else 'FAILED'} {meta['seconds']:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ surfnet
class Runner:
    def __init__(self, a, cfg):
        from surfnet.infer.engine import Inference, Volume, load_model
        from surfnet.volume import open_volume
        self.a, self.cfg = a, cfg
        self.model, self.spec = load_model(a.ckpt, a.device)
        self.infs = {}
        self._mk = lambda vol: Inference(self.model, self.spec, vol, None, a.device, tta=a.tta, amp=a.amp)
        self._vol = lambda url, cache: Volume(open_volume(url, cache), a.device, 2500, 800)

    def inf(self, sf):
        if sf["volume"] not in self.infs:
            cache = Path(self.a.cache) / Path(sf["volume"]).name
            self.infs[sf["volume"]] = self._mk(self._vol(f"{self.cfg['bucket']}/{sf['volume']}", cache))
        return self.infs[sf["volume"]]

    def settings(self, m: str) -> Settings:
        if m == "S5":
            return Settings(runs=self.a.runs, ahead=0.0, tta=self.a.tta, passes=self.a.passes)
        return Settings(ahead=0.0, **FAST)

    def run(self, d, c, sf, m, results):
        """Method m on case c -> (joint_solve's dict, seconds), or None while its input is missing."""
        z = arrays_of(d)                                 # of the truth only the known cells' positions are read
        K, S = z["K"], z["S"]
        s, spread0 = self.settings(m), None
        prev = PREV.get(m)
        if prev:                                                           # the previous link's result
            q = find_result(results, c["name"], prev)
            if q is None:
                return None
            Pin = q[0]
            spread0 = next(np.load(rr / c["name"] / f"{prev}.npz")["spread"].astype(np.float64) for rr in results
                           if (rr / c["name"] / f"{prev}.npz").exists())
            if m.endswith("K"):                                            # the draft kept where the runs disagree
                Pd = grown(results, c["name"])
                if Pd is None:
                    return None
                t0 = time.time()
                P_out, U = keep_draft(Pd, Pin, spread0, S)
                return {"P_out": P_out, "spread": spread0, "times": {}, "kept": int(U.sum())}, time.time() - t0
        elif m in WAYS:                                                    # GrowPatch's surface on the case's grid
            Pin = grown(results, c["name"])
            if Pin is None:
                return None
        elif "D" in z:                                                     # a wrong draft to correct
            Pin = z["D"]
        else:                                                              # --start bridge, as surfnet correct does it
            Pin = bridge(np.where(K[..., None], z["P"], -1.0), K, S)
        inf = self.inf(sf)
        inf.tta = s.tta                                                    # flips / rotations averaged per window
        rng = np.random.default_rng(self.a.seed)
        t0 = time.time()
        if m.endswith("T"):
            r = targeted(inf, self.spec, None, K | S, Pin, spread0, K, S, replace(s, runs=RESOLVE_RUNS), rng)
        else:
            r = joint_solve(inf, self.spec, None, K | S, Pin, K, S, s, rng)
        return r, time.time() - t0


def grown(results: list[Path], case: str):
    """The case's GrowPatch surface on its own grid, as `surfnet grow` writes it for `surfnet correct`, or None."""
    for r in results:
        f = r / case / "grow" / "surface"
        if (f / "x.tif").exists():
            return tifxyz.read(f)[0]
    return None


def methods_of(c) -> list:
    """A case's methods: those it was built with and, where it has G, the ways from GrowPatch's surface."""
    return list(c["methods"]) + (list(WAYS) if "G" in c["methods"] else [])


def cmd_surfnet(a):
    work, out = Path(a.work), Path(a.out)
    cfg = load_cfg(work)
    surf = {s["name"]: s for s in cfg["surfaces"]}
    results = [out] + [Path(r) for r in a.results]
    run = Runner(a, cfg)
    i, n = map(int, a.part.split("/"))
    allc = list(cases_of(work, a.only))
    methods = [x for x in RUN_ORDER if x in a.methods.split(",")]
    if a.split == "case":              # whole cases (their CT downloaded once), balanced over the workers by size x work
        load, jobs = [0] * n, []
        units = [(d, c, [m for m in methods if m in methods_of(c)]) for d, c in allc]
        cost = lambda d, ms: int((np.load(d / "arrays.npz")["S"] | np.load(d / "arrays.npz")["K"]).sum()) * \
            sum(2 if m.endswith("T") else 1 for m in ms)                   # noqa: E731
        for d, c, ms in sorted(units, key=lambda u: (-cost(u[0], u[2]), u[1]["name"])):
            w = int(np.argmin(load))
            load[w] += cost(d, ms)
            if w == i:
                jobs += [(d, c, m) for m in ms]
    else:                              # whole locations alternate between the workers: each fetches its CT once
        locs = sorted({(c["surface"], c["location"]) for _, c in allc})
        jobs = [(d, c, m) for m in methods for d, c in allc if locs.index((c["surface"], c["location"])) % n == i]
    for d, c, m in jobs:
        if m not in methods_of(c) or done(results, c["name"], m):
            continue
        if a.until and time.time() > a.until:
            print("deadline: stopping", flush=True)
            return
        res = run.run(d, c, surf[c["surface"]], m, results)
        if res is None:
            continue
        r, sec = res
        s = run.settings(m)
        meta = {"seconds": round(sec, 1), "kept": r["kept"]} if "kept" in r else \
            {"seconds": round(sec, 1), "amp": a.amp, "runs": RESOLVE_RUNS if m.endswith("T") else s.runs,
             "tta": s.tta, "passes": s.passes, "times": r.get("times"),
             **({"targeted": r.get("targeted")} if m.endswith("T") else {})}
        save_run(out / c["name"], m, r, c_valid(d), meta)
        print(f"{m} {c['name']}: {sec:.0f} s {r.get('times')}", flush=True)
        if a.device != "cpu":
            torch.cuda.empty_cache()


def c_valid(d):
    z = np.load(d / "arrays.npz")
    return z["K"] | z["S"]


def save_run(od: Path, m: str, r: dict, valid, meta: dict):
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"{m}.npz", P=r["P_out"].astype(np.float32), V=valid,
                        spread=r["spread"].astype(np.float16))
    write_json(od / f"{m}.json", meta)


# ------------------------------------------------------------------------------------------------ score
def _score_case(d, c, results, out, device, errors=None):
    """One case, every method (exact closest points on the GPU, `eval/score_gpu.py`) -> the scores row; the per-cell
    errors saved (err_<m>.npy). errors: a folder of such saved errors (<case>/err_<m>.npy) to score from instead of
    recomputing the closest points (then without the two-way distances)."""
    from surfnet.eval.score_gpu import score_surface, two_way_summary
    bins = (0, 10, 20, 40, 80, 160, 10_000)
    z = arrays_of(d)
    P, V, N, K, S = z["P"], z["V"], z["N"], z["K"], z["S"]
    sp, dist = c["spacing_vox"], ndimage.distance_transform_edt(~K)
    row = {"case": c, "methods": {}}
    for m in methods_of(c):
        if errors is not None:
            f = Path(errors) / c["name"] / f"err_{m}.npy"
            if not f.exists():
                continue
            err = np.load(f).astype(np.float64)
            res = {"error": np.nan_to_num(err), "covered": np.isfinite(err)}
        else:
            q = (z["E"], K | S) if m == "E" else (z["D"], K | S) if m == "D" else find_result(results, c["name"], m)
            if q is None:
                continue
            Q, VQ = q
            res = score_surface(P, N, V, S | K, S, Q, VQ, device=device)
            err = np.where(res["covered"], res["error"], np.nan)
            (out / c["name"]).mkdir(parents=True, exist_ok=True)
            np.save(out / c["name"] / f"err_{m}.npy", err.astype(np.float16))
        r = summary(res, S, sp)
        if errors is None:
            r["two_way"] = two_way_summary(res)
        r["by_distance"] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = S & (dist >= lo) & (dist < hi)
            if mm.sum() >= 20:
                s2 = summary(res, mm, sp)
                r["by_distance"][f"{lo}-{hi if hi < 10_000 else 'inf'}"] = [s2["score"], s2["p50"]]
        ek = np.abs(res["error"][res["covered"] & K])
        r["known_drift_p50_p95"] = [round(float(np.median(ek)), 3), round(float(np.quantile(ek, 0.95)), 3)] \
            if ek.size else None
        meta = next((json.loads((rr / c["name"] / f"{m}.json").read_text()) for rr in results
                     if (rr / c["name"] / f"{m}.json").exists()), {})
        r["seconds"] = meta.get("seconds")
        row["methods"][m] = r
    return row


def cmd_score(a):
    """Every case scored (this worker's share with --part i/n, one per GPU) -> out/scores.jsonl (n > 1:
    scores_part<i>.jsonl; `score --merge` joins them in case order) and the err maps."""
    work, out = Path(a.work) if a.work else None, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        rows = [json.loads(l) for f in sorted(out.glob("scores_part*.jsonl")) for l in open(f)]
        rows.sort(key=lambda r: r["case"]["name"])
        with open(out / "scores.jsonl", "w") as fo:
            fo.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"merged {len(rows)} cases", flush=True)
        return
    device = a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu"
    i, n = map(int, a.part.split("/"))
    todo = list(cases_of(work, a.only))[i::n]
    results = [Path(r) for r in a.results]
    t0 = time.time()
    with open(out / ("scores.jsonl" if n == 1 else f"scores_part{i}.jsonl"), "w") as fo:
        for k, (d, c) in enumerate(todo):
            row = _score_case(d, c, results, out, device, errors=a.errors)
            fo.write(json.dumps(row) + "\n")
            print(f"[{k + 1}/{len(todo)} {time.time() - t0:.0f} s]", c["name"],
                  {m: (v["score"], v["wrong_sheet"], v["p50"]) for m, v in row["methods"].items()}, flush=True)
    print(f"scored {len(todo)} cases on {device} in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ report
def cmd_report(a):
    rows = [json.loads(l) for l in open(Path(a.scores) / "scores.jsonl")]
    groups = {}
    for r in rows:
        c = r["case"]
        key = (c["set"], c["kind"], c["case"]["layout"], c["case"]["known"], c["case"]["size"],
               c["case"].get("perturb", ""), str(c["case"].get("amp", "")))
        for m, v in r["methods"].items():
            groups.setdefault(key, {}).setdefault(m, []).append(v)
    with open(Path(a.scores) / "results.csv", "w") as f:
        f.write("set,kind,layout,known,size_cells,size_mm,perturb,amp,method,n,score,wrong_sheet,p50,p90,coverage,"
                "gt_to_method_mean,gt_to_method_p95,method_to_gt_mean,known_drift_p50,seconds\n")
        for key in sorted(groups):
            for m, vs in groups[key].items():
                mean = lambda k: float(np.nanmean([v[k] if v.get(k) is not None else np.nan for v in vs]))  # noqa: E731
                tw = lambda w, k: float(np.nanmean([v["two_way"][w][k] if v.get("two_way", {}).get(w) else np.nan  # noqa: E731
                                                    for v in vs]))
                kd = float(np.nanmean([v["known_drift_p50_p95"][0] if v.get("known_drift_p50_p95") else np.nan for v in vs]))
                st, kind, lay, kn, size, pert, amp = key
                f.write(f"{st},{kind},{lay},{kn},{size},{size * CELL_UM / 1000:.1f},{pert},\"{amp}\",{m},{len(vs)},"
                        f"{mean('score'):.4f},{mean('wrong_sheet'):.4f},{mean('p50'):.3f},{mean('p90'):.3f},{mean('coverage'):.4f},"
                        f"{tw('gt_to_method', 'mean'):.2f},{tw('gt_to_method', 'p95'):.2f},{tw('method_to_gt', 'mean'):.2f},"
                        f"{kd:.3f},{mean('seconds'):.1f}\n")
    print(open(Path(a.scores) / "results.csv").read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("cases", "growpatch", "surfnet", "score", "report"))
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--scores")
    ap.add_argument("--only", nargs="*", default=None, help="surface names or case names")
    ap.add_argument("--split", default="location", choices=("location", "case"),
                    help="surfnet: how the cases are shared by the workers (--part i/n): whole locations, or whole "
                         "cases balanced by size")
    ap.add_argument("--ckpt")
    ap.add_argument("--amp", default=None, help="fp16 (default) | bf16 | fp32")
    ap.add_argument("--methods", default="S5")
    ap.add_argument("--part", default="0/1", help="i/n: this worker's share")
    ap.add_argument("--merge", action="store_true", help="score: join the parts' scores_part<i>.jsonl")
    ap.add_argument("--errors", default=None, help="score: from the per-cell errors an earlier score stage saved in this "
                    "folder (<case>/err_<m>.npy), not recomputed")
    ap.add_argument("--config", help="cases: the case set (default benchmark/cases.yaml)")
    ap.add_argument("--until", type=float, default=0, help="unix time after which no new run starts")
    ap.add_argument("--runs", type=int, default=5, help="S5: runs")
    ap.add_argument("--tta", type=int, default=2, help="S5: flips / rotations per window")
    ap.add_argument("--passes", type=int, default=3, help="S5: refinement passes at most")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--data", default="data", help="growpatch: where the normal grids are kept")
    ap.add_argument("--apprun")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--case-timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        a.device = "cpu"                      # scores also run on the CPU (slower)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
