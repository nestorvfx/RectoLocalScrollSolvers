"""The benchmark end to end on a synthetic scroll (benchmark/bench.py through its command line, on CPU): cases,
GrowPatch through `surfnet grow` (a stand-in VC3D), SurfNet alone (S5) and the ways from GrowPatch's surface, scores,
the results table; and the product (`surfnet correct`) on the benchmark's user inputs gives what the benchmark scored."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import phantom as PH  # noqa: E402
from test_cpu import write_model, write_zarr  # noqa: E402

from surfnet.common import tifxyz  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "benchmark" / "bench.py"


def bench(*args):
    r = subprocess.run([sys.executable, str(BENCH), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


def test_benchmark(tmp_path):
    b = tmp_path / "bucket" / "phantom"
    spec = PH.PhantomSpec(shape=(256, 208, 208))
    write_zarr(PH.make_volume(spec), b / "volume.zarr")
    for n in (2, 3):
        P, V = PH.winding_mesh(spec, n)
        tifxyz.write(b / f"gt_w{n}", P, V, {"scale": [1 / spec.cell_vox] * 2})
    write_model(tmp_path / "model.pt")
    work = tmp_path / "work"
    work.mkdir()
    (work / "cases.yaml").write_text(json.dumps({
        "bucket": str(tmp_path / "bucket"), "seed": 0, "locations": {"primary": 1},
        "surfaces": [{"name": "ph", "set": "primary", "mesh": "phantom/gt_w2", "volume": "phantom/volume.zarr",
                      "pitch_um": 9.362, "spacing_from": {"mesh": "phantom/gt_w3", "windings": 1},
                      "m7": "phantom/volume.zarr", "normal_grids": "phantom/grids"}],
        "cases": [{"kind": "fill", "layout": "ring", "known": 4, "size": 10},
                  {"kind": "extend", "layout": "side", "known": 4, "size": 10},
                  {"kind": "context", "layout": "patches", "known": 4, "size": 10},
                  {"kind": "correct", "layout": "ring", "known": 4, "size": 10, "perturb": "winding"}]}))
    bench("cases", "--work", work)
    cases = sorted((work / "cases").iterdir())
    assert len(cases) == 4 and all((c / "arrays.npz").exists() for c in cases)
    sp = json.loads((work / "surfaces" / "ph" / "surface.json").read_text())["spacing_vox"]
    assert abs(sp - spec.spacing) < 0.5
    (b / "grids").mkdir()
    (b / "grids" / "metadata.json").write_text("{}")
    fake = tmp_path / "apprun"                                    # a stand-in GrowPatch: like VC3D it keeps the seed's
    fake.write_text("#!" + sys.executable + "\n" + (            # grid, padded by 2 cells, and grows a smooth surface
        "import json, sys\nfrom pathlib import Path\nimport numpy as np, tifffile\n"     # over it (thin-plate here)
        f"sys.path.insert(0, {str(BENCH.parents[1])!r})\nfrom surfnet.infer.surface import thin_plate_fill\n"
        "a = sys.argv\nt, s = Path(a[a.index('-t') + 1]), Path(a[a.index('--resume') + 1])\n"
        "X = np.pad(np.stack([tifffile.imread(s / f'{k}.tif') for k in 'xyz'], -1).astype(np.float64), "
        "((2, 2), (2, 2), (0, 0)), constant_values=-1.0)\n"
        "K = X[..., 0] > 0\nX = thin_plate_fill(X, K, ~K)\n"
        "t.mkdir(parents=True, exist_ok=True)\n"
        "for i, k in enumerate('xyz'):\n"
        "    tifffile.imwrite(t / f'{k}.tif', X[..., i].astype(np.float32))\n"
        "(t / 'meta.json').write_text(json.dumps({'format': 'tifxyz', 'scale': [0.05, 0.05], 'grid_offset': [2, 2]}))\n"))
    fake.chmod(0o755)
    grow = tmp_path / "grow"                                      # GrowPatch on the fill, extend and context cases,
    for part in ("0/2", "1/2"):                                   # over two workers
        bench("growpatch", "--work", work, "--out", grow, "--apprun", fake, "--data", tmp_path / "data", "--part", part)
    for c in cases:
        has = "G" in json.loads((c / "case.json").read_text())["methods"]
        assert (grow / c.name / "grow" / "surface" / "x.tif").exists() == has, c.name
        if has:
            assert json.loads((grow / c.name / "G.json").read_text())["covered"] > 0.9, c.name
    gpu = tmp_path / "gpu"                                       # SurfNet alone, split in two as on the two T4s
    for part in ("0/2", "1/2"):
        bench("surfnet", "--work", work, "--out", gpu, "--ckpt", tmp_path / "model.pt", "--device", "cpu", "--runs", 1,
              "--tta", 1, "--passes", 1, "--part", part, "--cache", tmp_path / "cache")
    assert all((gpu / c.name / "S5.npz").exists() for c in cases)
    ways = tmp_path / "ways"                                      # the ways from GrowPatch's surface, whole cases
    for part in ("0/2", "1/2"):                                   # over two workers
        bench("surfnet", "--work", work, "--out", ways, "--methods", "GF4Cmed,GF4Cmedx1T,GF4CmedK,GF4Cmedx1TK",
              "--split", "case", "--part", part, "--ckpt", tmp_path / "model.pt", "--device", "cpu", "--results", grow,
              "--cache", tmp_path / "cache")
    for c in cases:
        for m in ("GF4Cmed", "GF4Cmedx1T", "GF4CmedK", "GF4Cmedx1TK"):  # where GrowPatch has an output
            assert (ways / c.name / f"{m}.npz").exists() == (grow / c.name / "G.json").exists(), (c.name, m)
        if (grow / c.name / "G.json").exists():
            assert "kept" in json.loads((ways / c.name / "GF4CmedK.json").read_text())
            j = json.loads((ways / c.name / "GF4Cmedx1T.json").read_text())
            assert j["runs"] == 8 and j["tta"] == 1 and "targeted" in j
    # the product on the benchmark's user inputs gives exactly what the benchmark scored: SurfNet alone (correct
    # --start bridge, the default mode) and GrowPatch, then fast with GrowPatch's surface kept where the runs disagree
    # (correct on grow/surface --mode fast --keep-draft)
    for c in cases[:1]:                                                 # the fill case
        u = grow / c.name / "grow"
        for m, res, args in (("S5", gpu, ["--sheet", u / "sheet", "--start", "bridge", "--runs", "1", "--tta", "1",
                                          "--passes", "1"]),
                             ("GF4CmedK", ways, ["--sheet", u / "surface", "--mode", "fast", "--keep-draft"])):
            out = tmp_path / "product" / c.name / m
            r = subprocess.run([sys.executable, "-m", "surfnet.cli", "--device", "cpu", "correct", "--ckpt",
                                tmp_path / "model.pt", "--volume", b / "volume.zarr", "--pitch-um", "9.362", "--known",
                                u / "known.png", "--solve", u / "area.png", "--out", out, "--cache", tmp_path / "cache",
                                *args], capture_output=True, text=True, cwd=BENCH.parents[1])
            assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
            Q, VQ, _ = tifxyz.read(out / "surface")
            z = np.load(res / c.name / f"{m}.npz")
            assert (VQ == z["V"]).all() and np.abs(Q[VQ] - z["P"][VQ]).max() < 1e-3, (c.name, m)
    sc = tmp_path / "scores"
    bench("score", "--work", work, "--results", grow, gpu, ways, "--out", sc)
    rows = [json.loads(l) for l in open(sc / "scores.jsonl")]
    fill = next(r for r in rows if r["case"]["kind"] == "fill")
    assert set(fill["methods"]) == {"E", "G", "S5", "GF4Cmed", "GF4Cmedx1T", "GF4CmedK", "GF4Cmedx1TK"}
    assert fill["methods"]["G"]["coverage"] > 0.9 and fill["methods"]["G"]["seconds"] is not None
    assert fill["methods"]["E"]["known_drift_p50_p95"][0] < 1e-3
    corr = next(r for r in rows if r["case"]["kind"] == "correct")
    assert set(corr["methods"]) == {"D", "E", "S5"}
    out = bench("report", "--scores", sc)
    assert "fill,ring,4,10" in out and (sc / "results.csv").exists()
    assert np.isfinite(fill["methods"]["S5"]["p50"])
