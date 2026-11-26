"""One benchmark case (a case folder bench.py built) solved by the recto local solver as its command line solves a
sheet (recto_local_solver.solve.solve_area), and scored as bench.py scores every method (its own _score_case). The
solver gets the case's user inputs only: the known cells' positions, the area, the CT; the ground truth only scores.

  RF   the frame (no horizontal fibres)          RFS   the frame + the fibre step
-> OUT/<case>/RF.npz, RFS.npz (P, V on the case's grid, as bench.py stores a method), RF.json, RFS.json (seconds, the
   solve's record), err_RF.npy, err_RFS.npy (the error maps), scores.json (the benchmark's score row), work/calib.npz

  python recto_case.py --case WORK/cases/<name> --out OUT [--cache CT_CACHE] [--volume URL] [--axis AXIS.json]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]                       # the packages; bench.py (the cases, the scorer)
VOLUME = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0"
AXIS = ROOT / "recto_local_solver" / "data" / "axis_PHerc0139.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="a case folder (case.json, arrays.npz) of a benchmark work tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="the CT chunk cache (default OUT/cache)")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--axis", default=str(AXIS))
    ap.add_argument("--gpu-cache-gb", type=float, default=None)
    a = ap.parse_args()
    import torch
    import bench
    from recto_local_solver import volume
    from recto_local_solver.solve import FOCUS, solve_area

    d = Path(a.case)
    c = json.loads((d / "case.json").read_text())
    out = Path(a.out)
    od = out / c["name"]
    od.mkdir(parents=True, exist_ok=True)
    volume.configure(a.volume, a.cache or out / "cache", a.axis, a.gpu_cache_gb)
    z = bench.arrays_of(d)
    P, V, K, S = z["P"], z["V"], z["K"], z["S"]
    known, area = K & V, S & V
    t0 = time.time()
    r = solve_area(np.where(known[..., None], P, 0.0), known, area, od / "work", focus=FOCUS, ctx=80, fibre_step=True)
    rec = r["record"]
    keep = known | r["solved"]
    secs = {"RF": rec["s_calib"] + rec["s_frame"], "RFS": round(time.time() - t0, 1)}
    for m, Q in (("RF", r["frame"]), ("RFS", r["fibre_step"])):
        Q = np.where(known[..., None], P, np.where(r["solved"][..., None], Q, 0.0))
        np.savez_compressed(od / f"{m}.npz", P=np.where(keep[..., None], Q, -1.0).astype(np.float32), V=keep)
        bench.write_json(od / f"{m}.json", {"seconds": round(secs[m], 1), "record": rec})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = bench._score_case(d, dict(c, methods=["RF", "RFS"]), [out], out, device)
    bench.write_json(od / "scores.json", row)
    print(f"RECTO {c['name']}: " + "  ".join(f"{m} score {100 * v['score']:.1f} ({v['seconds']} s)"
                                            for m, v in row["methods"].items()), flush=True)


if __name__ == "__main__":
    main()
