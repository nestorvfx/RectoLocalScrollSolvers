"""Surfaces scored against a benchmark case's ground truth, as bench.py scores every method (its _score_case): the
score (per cell, full credit within 2 vox of the truth, falling linearly to 0 at half the winding spacing; the mean over
the area), the share of the area on the wrong sheet, the median error. For the README's quick start (examples/) or any
case folder bench.py built; on the GPU if there is one, else the CPU.

  python score_surface.py --case ../examples/hole_1.9mm ../out/recto/frame.tifxyz ../out/surfnet/surface ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]                # the packages; bench.py (the cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="a case folder (case.json, arrays.npz)")
    ap.add_argument("surfaces", nargs="+", help="tifxyz directories in the volume's level-0 voxels")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: the GPU if there is one)")
    a = ap.parse_args()
    import torch

    import bench
    from surfnet.common import tifxyz
    from surfnet.eval.score import summary
    from surfnet.eval.score_gpu import score_surface
    d = Path(a.case)
    c = json.loads((d / "case.json").read_text())
    z = bench.arrays_of(d)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{c['name']}: {int(z['S'].sum())} area cells, winding spacing {c['spacing_vox']:.1f} vox")
    for s in a.surfaces:
        Q, VQ, _ = tifxyz.read(s)
        res = score_surface(z["P"], z["N"], z["V"], z["S"] | z["K"], z["S"], Q, VQ, device=device)
        r = summary(res, z["S"], c["spacing_vox"])
        print(f"  {s}: score {100 * r['score']:.1f}, {100 * r['wrong_sheet']:.1f} % of the area on the wrong sheet, "
              f"median error {r['p50']} vox")


if __name__ == "__main__":
    main()
