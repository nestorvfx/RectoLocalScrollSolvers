"""The benchmark's 80 cases (docs/results/subset80.txt), of the 240 cases with an area to grow in the two case sets
(cases.yaml, cases_ctx64.yaml), drawn at random before any result; only the case specs are read, no scores.

  subset50.txt  50 drawn at random within each kind of area (ways.family), in proportion to the kinds (largest
                remainder; the seat left over to a kind drawn at random), seed 0
  subset80.txt  those 50 and 30 more (seed 1) from the other cases, so that the 80 are in proportion to the kinds;
                within a kind the 30 are spread evenly over its amounts of context (known cells: holes 20 / 64,
                extensions 40 / 64, known on one side 10 / 20 / 80, chosen from an earlier test as the three amounts at
                which SurfNet alone did best; two sides or patches: the two layouts)

  python subset.py main=CASES_DIR ctx=CASES_DIR OUT_DIR     (a set's work/cases: one case.json per case)"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ways import family  # noqa: E402


def seats_for(n, sizes, rng):
    """n seats over the kinds in proportion to sizes (largest remainder, ties at random)."""
    quota = {k: n * v / sum(sizes.values()) for k, v in sorted(sizes.items())}
    seats = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - seats[k]), rng.random()))[:n - sum(seats.values())]:
        seats[k] += 1
    return seats


def main():
    cases = {}
    for arg in sys.argv[1:-1]:
        label, d = arg.split("=", 1)
        for f in sorted(Path(d).glob("*/case.json")):
            c = json.loads(f.read_text())
            if "G" in c["methods"]:
                cases[c["name"]] = (label, family(c["case"]), c["case"])
    out = Path(sys.argv[-1])
    kinds = {}
    for n, (_, k, _) in sorted(cases.items()):
        kinds.setdefault(k, []).append(n)
    sizes = {k: len(v) for k, v in kinds.items()}
    rng = random.Random(0)                                     # the 50 (as drawn first: the same calls, in order)
    s50 = seats_for(50, sizes, rng)
    pick50 = sorted(n for k, ns in sorted(kinds.items()) for n in rng.sample(ns, s50[k]))
    rng = random.Random(1)                                     # 30 more: the 80 in proportion to the kinds
    s80 = seats_for(80, sizes, rng)
    more = []
    for k, ns in sorted(kinds.items()):                       # the kind's seats spread evenly over its amounts of context
        strata = {}
        for n in ns:
            c = cases[n][2]
            if n in pick50 or (c["kind"] == "context" and c["layout"] == "side" and c["known"] == 5):
                continue
            strata.setdefault(c["layout"] if c["layout"] in ("between", "patches") else c["known"], []).append(n)
        seats = seats_for(s80[k] - s50[k], {a: 1 for a in strata}, rng)
        more += [n for a in sorted(strata) for n in rng.sample(strata[a], seats[a])]
    more = sorted(more)
    for name, pick, seats in (("subset50", pick50, s50), ("subset80", sorted(pick50 + more), s80)):
        with open(out / f"{name}.txt", "w") as f:
            f.write(f"# {len(pick)} of {len(cases)} cases, stratified by kind at random (subset.py): "
                    + ", ".join(f"{k} {seats[k]}" for k in sorted(seats)) + "\n")
            f.writelines(f"{cases[n][0]} {n}\n" for n in pick)


if __name__ == "__main__":
    main()
