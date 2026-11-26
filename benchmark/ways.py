"""The benchmark's methods against GrowPatch (docs/SUPPLEMENTARY.md): SurfNet's ways (bench.py) and the recto local
solver's two results (recto_case.py). Post-processing only: it reads the stages' score files and results trees (the
methods' own json: seconds and their parts) and recomputes nothing.

  python ways.py analyse   --scores S.jsonl ... --trees T ... --out ways.json
      score, share on the wrong sheet, p50 and end-to-end seconds per case and way, over the cases GrowPatch ran on
  python ways.py report    ways.json [--methods M,...] [--pairs X:Y ...] [--exclude CASES.txt]
      per family: the mean score, the failures (more than 10 % of the area on the wrong sheet) and the mean seconds
      (all methods and cases: -> ways_families.json, [score, seconds, cases, failures]); X against Y case by case
  python ways.py robust    ways.json [--ways M,...] [--thresholds 10,20] [--exclude CASES.txt]
      failures per way (more than T % of the area on the wrong sheet), and how many of GrowPatch's failures each way
      fixes and how many of its passes it fails
  python ways.py locations ways.json [--ways M,...]
      by physical location (3 per winding, shared by all case kinds): means, failures, a location-clustered bootstrap
      interval of each way's mean gain over GrowPatch, and validation (w056-w057) against test (w058-w059) windings
--exclude: the cases listed in a file left out (e.g. docs/results/recto_used_cases.txt: the held-out cases only)

Seconds of a way: the measured seconds of its chain of links (`surfnet grow` for the G ways, then every surfnet step up
to it), each with the CT download it made: GrowPatch on one CPU core, SurfNet on one GPU. The recto local solver's: its
own run on one GPU (RF: calibration + frame; RFS: the whole run)."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

FAMILIES = ["holes up to 3.8 mm", "holes 7.6-11.9 mm", "extensions 1.9-15.2 mm",
            "7.6 mm area, known on two sides / patches", "7.6 mm area, known on one side"]


def family(case: dict) -> str:
    """A case's family from its spec (case.json "case": kind, layout, size)."""
    k, s = case["kind"], case["size"]
    if k == "fill":
        return FAMILIES[0] if s <= 80 else FAMILIES[1]
    if k == "extend":
        return FAMILIES[2]
    if k == "context":
        return FAMILIES[4] if case["layout"] == "side" else FAMILIES[3]
    return k


def read_scores(files):
    """The score files' rows by case; a case in several files gets every file's methods."""
    rows = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            row = rows.setdefault(r["case"]["name"], {"case": dict(r["case"]), "methods": {}})
            row["methods"].update(r["methods"])
            row["case"]["methods"] = list(dict.fromkeys(list(row["case"]["methods"]) + list(r["case"]["methods"])))
    return rows


def chain(m: str) -> list:
    """The links whose times add up to way m (m itself last): GrowPatch, then the ways from its surface."""
    return {"GF4Cmed": ["G", "GF4Cmed"], "GF4Cmedx1T": ["G", "GF4Cmed", "GF4Cmedx1T"],
            "GF4CmedK": ["G", "GF4Cmed", "GF4CmedK"],
            "GF4Cmedx1TK": ["G", "GF4Cmed", "GF4Cmedx1T", "GF4Cmedx1TK"]}.get(m, [m])


def cmd_analyse(a):
    rows = {n: r for n, r in read_scores(a.scores).items() if "G" in r["case"]["methods"]}   # the areas to grow
    sec = defaultdict(dict)
    for t in a.trees:
        for f in glob.glob(f"{t}/*/*.json"):
            n, m = f.split("/")[-2], f.split("/")[-1][:-5]
            if n not in rows or m in sec[n]:
                continue
            try:
                d = json.load(open(f))
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("seconds") is not None:
                sec[n][m] = float(d["seconds"])
    res = {}
    for n in sorted(rows):
        out = {}
        for m, v in rows[n]["methods"].items():
            ts = [sec[n].get(x) for x in chain(m)]
            out[m] = {"score": round(100 * v["score"], 2), "wrong": round(100 * v["wrong_sheet"], 2), "p50": v["p50"],
                      "coverage": v["coverage"],
                      "seconds": round(sum(ts), 1) if all(t is not None for t in ts) else None}
        res[n] = {"case": rows[n]["case"], "methods": out}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{len(res)} cases -> {a.out}")


def drop(d: dict, path) -> dict:
    """The analysis without the cases listed in `path` (a case name first on each line; # comments), or all of it."""
    if not path:
        return d
    skip = {ln.split()[0] for ln in open(path) if ln.strip() and not ln.startswith("#")}
    return {n: r for n, r in d.items() if n not in skip}


def cmd_report(a):
    d = drop(json.load(open(a.analysis)), a.exclude)
    fam = defaultdict(list)
    for n, r in d.items():
        fam[family(r["case"]["case"])].append(n)
        fam["all"].append(n)
    allm = a.methods.split(",") if a.methods else sorted({m for r in d.values() for m in r["methods"]})
    rows = {}
    for f in FAMILIES + ["all"]:
        ns = fam.get(f, [])
        if not ns:
            continue
        rows[f] = {}
        for m in allm:
            v = [d[n]["methods"][m] for n in ns if m in d[n]["methods"]]
            if len(v) < len(ns):
                continue
            t = [x["seconds"] for x in v if x["seconds"] is not None]
            rows[f][m] = (float(np.mean([x["score"] for x in v])), float(np.mean(t)) if len(t) == len(v) else None,
                          len(v), sum(x["wrong"] > 10 for x in v))
        print(f"\n== {f} ({len(ns)} cases): score, failures, seconds")
        for m, (rs, ts, n, nf) in sorted(rows[f].items(), key=lambda kv: -kv[1][0]):
            print(f"  {m:12s} {rs:6.1f}   {nf:3d} ({100 * nf / n:3.0f} %)   {'-' if ts is None else round(ts):>6} s")
    if not a.exclude and not a.methods:                # the full table only
        json.dump(rows, open(a.analysis.replace(".json", "_families.json"), "w"), indent=1)
    for pr in a.pairs or []:                       # better / worse: by more than 0.5 points
        x, y = pr.split(":")
        print(f"\n{x} against {y}")
        by = defaultdict(list)
        for n, r in d.items():
            if x in r["methods"] and y in r["methods"]:
                dd = r["methods"][x]["score"] - r["methods"][y]["score"]
                by[family(r["case"]["case"])].append((dd, n))
                by["all"].append((dd, n))
        for f in FAMILIES + ["all"]:
            v = by.get(f)
            if v:
                print(f"  {f:42s} n={len(v):2d} better {sum(dd > 0.5 for dd, _ in v):2d} "
                      f"worse {sum(dd < -0.5 for dd, _ in v):2d}  mean {np.mean([dd for dd, _ in v]):+5.1f}  "
                      f"largest drop {min(v)[0]:+5.1f} ({min(v)[1]})")


def cmd_robust(a):
    """Failures (more than T % of the area on the wrong sheet) per way: all of its cases; of GrowPatch's failures, how
    many it fixes; of GrowPatch's passes, how many it fails; its worst score and 10th percentile."""
    d = drop(json.load(open(a.analysis)), a.exclude)
    g = {n: r["methods"]["G"]["wrong"] for n, r in d.items() if "G" in r["methods"]}
    for T in [float(t) for t in a.thresholds.split(",")]:
        gfail = {n for n, x in g.items() if x > T}
        print(f"\n== more than {T:g} % on the wrong sheet: GrowPatch in {len(gfail)} of {len(g)} cases")
        print(f"  {'way':12s} | failed of n | GrowPatch's failures: still failed, fixed | GrowPatch's passes: failed |"
              f" worst score, p10")
        for m in a.ways.split(","):
            v = {n: r["methods"][m]["wrong"] for n, r in d.items() if m in r["methods"]}
            if not v:
                continue
            x = np.array(list(v.values()))
            sc = np.array([d[n]["methods"][m]["score"] for n in v])
            b = [n for n in v if n in gfail]
            c = [n for n in v if n in g and n not in gfail]
            bb, cc = sum(v[n] > T for n in b), sum(v[n] > T for n in c)
            print(f"  {m:12s} | {int((x > T).sum()):3d} of {len(x):3d} | {bb:3d} {len(b) - bb:3d} | {cc:3d} of {len(c):3d} |"
                  f" {sc.min():5.1f} {np.percentile(sc, 10):5.1f}")


def cmd_locations(a):
    d = json.load(open(a.analysis))
    loc = lambda r: f"{r['case']['surface'][5:]} L{r['case']['location']}"            # noqa: E731
    ways = a.ways.split(",")
    rng = np.random.default_rng(0)
    print(f"{len(d)} cases at {len({loc(r) for r in d.values()})} locations")
    print(f"{'location':10s} {'n':>3s} " + " ".join(f"{m:>11s}" for m in ways) + "   failures (> 10 % wrong sheet)")
    for L in sorted({loc(r) for r in d.values()}):
        rs = [r for r in d.values() if loc(r) == L]
        mean = [np.mean([r["methods"][m]["score"] for r in rs if m in r["methods"]] or [np.nan]) for m in ways]
        wf = [sum(r["methods"][m]["wrong"] > 10 for r in rs if m in r["methods"]) for m in ways]
        print(f"{L:10s} {len(rs):3d} " + " ".join(f"{x:11.1f}" for x in mean) + "   "
              + " ".join(f"{m} {f}" for m, f in zip(ways, wf)))
    for f in ["all"] + FAMILIES:
        print(f"-- {f}")
        for m in ways[1:]:
            pairs = [(loc(r), r["methods"][m]["score"] - r["methods"][ways[0]]["score"]) for r in d.values()
                     if m in r["methods"] and ways[0] in r["methods"] and f in ("all", family(r["case"]["case"]))]
            if not pairs:
                continue
            locs = sorted({L for L, _ in pairs})
            by = {L: [x for k, x in pairs if k == L] for L in locs}
            boot = [np.mean([x for L in rng.choice(locs, len(locs)) for x in by[L]]) for _ in range(10000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            split = {s: np.mean([x for L, x in pairs if L[:4] in w] or [np.nan])
                     for s, w in (("validation", ("w056", "w057")), ("test", ("w058", "w059")))}
            print(f"  {m} - {ways[0]}: mean {np.mean([x for _, x in pairs]):+.2f} over {len(pairs)} cases at "
                  f"{len(locs)} locations, 95 % interval clustered by location {lo:+.2f} to {hi:+.2f}; validation "
                  f"windings {split['validation']:+.2f}, test windings {split['test']:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyse")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--trees", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("analysis")
    p.add_argument("--methods", default=None)
    p.add_argument("--pairs", nargs="*")
    p.add_argument("--exclude", default=None)
    p = sub.add_parser("robust")
    p.add_argument("analysis")
    p.add_argument("--ways", default="G,S5,GF4Cmed,GF4Cmedx1T,GF4CmedK,GF4Cmedx1TK,RF,RFS")
    p.add_argument("--thresholds", default="10,20", help="% of the area on the wrong sheet")
    p.add_argument("--exclude", default=None)
    p = sub.add_parser("locations")
    p.add_argument("analysis")
    p.add_argument("--ways", default="G,GF4CmedK,GF4Cmedx1TK,S5", help="the first is the reference (the recto local "
                   "solver's: --ways G,RFS,RF; the intervals depend on the ways' order, a seeded resampling)")
    a = ap.parse_args()
    {"analyse": cmd_analyse, "report": cmd_report, "robust": cmd_robust, "locations": cmd_locations}[a.cmd](a)


if __name__ == "__main__":
    main()
