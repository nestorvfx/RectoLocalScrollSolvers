"""CPU tests on a synthetic rolled scroll (tests/phantom.py): the user-facing paths run end to end.

  python -m pytest -q tests          (about 5 min on 4 CPU cores; no GPU needed)"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import phantom as PH  # noqa: E402

from surfnet.common import tifxyz  # noqa: E402
from surfnet.volume import ArraySource, LocalZarrSource  # noqa: E402


def write_zarr(vol: np.ndarray, root: Path, levels: int = 2, chunk: int = 64) -> None:
    """An uncompressed OME-Zarr v2 pyramid ('/' separator), as VC3D volumes are laid out."""
    src = ArraySource.from_volume(vol, n_levels=levels, chunk=chunk)
    for lv in range(levels):
        a = src.levels[lv]
        d = root / str(lv)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": list(a.shape), "chunks": [chunk] * 3,
                                               "dtype": "|u1", "compressor": None, "fill_value": 0, "order": "C",
                                               "filters": None, "dimension_separator": "/"}))
        for key in np.ndindex(*src.grid(lv)):
            blk = src.read(lv, key)
            if blk is not None:
                p = d.joinpath(*map(str, key))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(blk.tobytes())


def write_model(path: Path) -> None:
    """A randomly initialised model with the trained model's architecture and heads but narrow layers (4-8 channels,
    one block per stage): every code path of the full model at a small fraction of its CPU time."""
    from surfnet.common.config import load
    from surfnet.model import NetSpec, TwoStream
    torch.manual_seed(0)
    cfg = load("train.yaml")
    sd = dict(cfg["model"], **{k: cfg["crops"]["spec"][k] for k in ("fk", "fout", "ck", "cout", "ppc")})
    sd.update(f_stem=4, f_ch=[4, 8, 8, 8, 8], c_stem=4, c_ch=[4, 4, 8, 8, 8], blocks=[1, 1, 1, 1, 1])
    torch.save({"ema": TwoStream(NetSpec.from_dict(sd)).state_dict(), "spec": sd, "cfg": cfg, "step": 0}, path)


@pytest.fixture(scope="module")
def scroll(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scroll")
    spec = PH.PhantomSpec()
    vol = PH.make_volume(spec)
    write_zarr(vol, tmp / "volume.zarr")
    P, V = PH.winding_mesh(spec, 2)
    P, V = P[:, :60], V[:, :60]
    tifxyz.write(tmp / "sheet", P, V, {"scale": [1 / spec.cell_vox] * 2})
    known = np.zeros(V.shape, np.uint8)
    known[:, :20] = 255
    from PIL import Image
    Image.fromarray(known).save(tmp / "known.png")
    write_model(tmp / "model.pt")
    return {"dir": tmp, "vol": vol, "spec": spec}


def test_local_zarr_reads_as_array(scroll):
    a = ArraySource.from_volume(scroll["vol"], n_levels=2, chunk=64)
    z = LocalZarrSource(scroll["dir"] / "volume.zarr")
    for lv in (0, 1):
        assert z.shape(lv) == a.shape(lv)
        for key in [(0, 0, 0), (1, 1, 1), tuple(g - 1 for g in z.grid(lv))]:
            x, y = a.read(lv, key), z.read(lv, key)
            assert (x is None and y is None) or np.array_equal(x, y)


@pytest.mark.parametrize("start", ["draft", "bridge"])
def test_correct(scroll, start):
    from surfnet.infer.correct import Settings, correct_sheet
    d = scroll["dir"]
    out = d / f"out_{start}"
    rep = correct_sheet(str(d / "model.pt"), str(d / "sheet"), str(d / "volume.zarr"), 9.362, str(d / "known.png"),
                        str(out), settings=Settings(start=start, runs=1, tta=1, passes=1), device="cpu",
                        log=lambda *_: None)
    P, V, meta = tifxyz.read(out / "surface")
    assert V.sum() > 0 and np.isfinite(P[V]).all()
    assert rep["solve_cells"] > 0 and rep["known_cells"] > 0
    P0, V0, _ = tifxyz.read(d / "sheet")
    k = np.zeros(V0.shape, bool)
    k[:, :20] = True
    assert np.allclose(P[k & V], P0[k & V], atol=1e-3)          # known cells stay where they are
    maps = np.load(out / "maps.npz")
    assert set(maps.files) >= {"known", "solve", "move", "spread", "confidence", "confidence_input"}


@pytest.mark.parametrize("flip", [False, True])
def test_correct_fast_resolve_keep(scroll, flip):
    """The fast mode, then the targeted re-solve, then the draft kept where the runs disagree (and with the grid
    flipped)."""
    from surfnet.infer.correct import FAST, Settings, correct_sheet
    d = scroll["dir"]
    out = d / f"out_fast_{flip}"
    rep = correct_sheet(str(d / "model.pt"), str(d / "sheet"), str(d / "volume.zarr"), 9.362, str(d / "known.png"),
                        str(out), settings=Settings(start="bridge", **{**FAST, "runs": 2}), resolve=True, keep=True,
                        flip=flip, device="cpu", log=lambda *_: None)
    P, V, _ = tifxyz.read(out / "surface")
    P0, V0, _ = tifxyz.read(d / "sheet")
    k = np.zeros(V0.shape, bool)
    k[:, :20] = True
    assert V.sum() > 0 and np.isfinite(P[V]).all() and np.allclose(P[k & V], P0[k & V], atol=1e-3)
    assert rep["resolve"] and rep["resolved_cells"] is not None and rep["kept_cells"] is not None and \
        len(rep["times"]) == 2
    assert "confidence_output" not in rep and set(np.load(out / "maps.npz").files) == {"known", "solve", "move",
                                                                                      "spread"}


def test_correct_bridge_fills_an_empty_hole(scroll, tmp_path):
    """A hole with no positions in a 10-vox sheet (resampled to the model's cell): bridged and solved, no gap left."""
    from PIL import Image
    from surfnet.infer.correct import Settings, correct_sheet
    d = scroll["dir"]
    P, V = PH.winding_mesh(PH.PhantomSpec(cell_vox=10.0), 2)
    hole = np.zeros(V.shape, bool)
    hole[4:8, 10:18] = True
    tifxyz.write(tmp_path / "sheet", np.where(hole[..., None], -1.0, P), V & ~hole, {"scale": [0.1, 0.1]})
    Image.fromarray((V & ~hole).astype(np.uint8) * 255).save(tmp_path / "known.png")
    Image.fromarray(hole.astype(np.uint8) * 255).save(tmp_path / "hole.png")
    rep = correct_sheet(str(d / "model.pt"), str(tmp_path / "sheet"), str(d / "volume.zarr"), 9.362,
                        str(tmp_path / "known.png"), str(tmp_path / "out"), solve_mask=str(tmp_path / "hole.png"),
                        settings=Settings(start="bridge", runs=1, tta=1, passes=1), device="cpu", log=lambda *_: None)
    Q, VQ, _ = tifxyz.read(tmp_path / "out" / "surface")
    m = np.load(tmp_path / "out" / "maps.npz")
    assert rep["solve_cells"] >= hole.sum() * 4 and (VQ == (m["known"] | m["solve"])).all()
    r, c = np.nonzero(m["solve"])
    assert VQ[int(r.mean()), int(c.mean())] and np.isfinite(Q[VQ]).all()


def test_targeted_pins_the_agreed_cells(monkeypatch):
    """The re-solve takes only the cells whose runs spread more than half a winding; the others become known."""
    from surfnet.infer import correct as C
    seen = {}

    def fake(inf, spec, axis, V0, P, known, solve, s, rng, log=None):
        seen.update(known=known.copy(), solve=solve.copy(), runs=s.runs)
        return {"P_out": np.where(solve[..., None], P + 1.0, P), "spread": np.where(solve, 1.0, 0.0), "times": {}}

    monkeypatch.setattr(C, "joint_solve", fake)
    H, W = 8, 10
    K = np.zeros((H, W), bool)
    K[:, :3] = True
    S = ~K
    spread = np.zeros((H, W))
    spread[:, 6:] = 9.0
    P = np.zeros((H, W, 3))
    r = C.targeted(None, None, None, K | S, P, spread, K, S, C.Settings(runs=C.RESOLVE_RUNS), None)
    U = S & (spread > C.HALF_WINDING)
    assert (seen["solve"] == U).all() and (seen["known"] == (K | (S & ~U))).all() and seen["runs"] == 8
    assert r["targeted"] == int(U.sum()) and (r["P_out"][U] == 1).all() and (r["P_out"][~U] == 0).all()
    assert (r["spread"][U] == 1).all() and (r["spread"][S & ~U] == 0).all()


def test_keep_draft(monkeypatch):
    """--keep-draft: after the last step, the solve cells whose runs still spread more than half a winding take the
    input's position (the draft); the others keep the result; the known cells are untouched."""
    from surfnet.infer import correct as C

    def fake(inf, spec, axis, V0, P, known, solve, s, rng, log=None):
        spread = np.where(solve, 1.0, 0.0)
        spread[:, 6:] = 9.0
        return {"P_out": np.where(solve[..., None], P + 1.0, P), "spread": np.where(solve, spread, 0.0), "times": {}}

    monkeypatch.setattr(C, "joint_solve", fake)
    H, W = 8, 10
    K = np.zeros((H, W), bool)
    K[:, :3] = True
    S = ~K
    P = np.stack([*np.meshgrid(np.arange(W), np.arange(H)), np.full((H, W), 5)], -1).astype(np.float64)
    r = C.apply(None, None, None, K | S, P, K, S, C.Settings(confidence=False), keep=True)
    U = S & (np.arange(W)[None] >= 6)
    assert r["kept"] == int(U.sum()) and (r["P_out"][U] == P[U]).all() and (r["P_out"][S & ~U] == P[S & ~U] + 1).all()
    assert (r["P_out"][K] == P[K]).all() and (r["move"][U] == 0).all()


def test_grow(scroll, tmp_path):
    """`surfnet grow` with a stand-in VC3D: like GrowPatch it keeps the seed's grid (padded by 2 cells, its
    grid_offset), and it writes the winding 2 vox off. The seed, GrowPatch's parameters, and its surface back on the
    sheet's grid over the hole, the known cells unchanged; a VC3D that fails leaves the hole bridged, with a warning."""
    import subprocess
    from PIL import Image
    from surfnet.common.geometry import as_t, resample_grid
    from surfnet.infer.grow import write_seed
    from surfnet.infer.surface import normals, thin_plate_fill
    spec = scroll["spec"]
    P, V = PH.winding_mesh(spec, 2)
    P, V = P[:, :60], V[:, :60]
    hole = np.zeros(V.shape, bool)
    hole[8:-8, 22:40] = True
    known = V & ~hole
    tifxyz.write(tmp_path / "sheet", np.where(known[..., None], P, -1.0), known, {"scale": [1 / spec.cell_vox] * 2})
    Image.fromarray(known.astype(np.uint8) * 255).save(tmp_path / "known.png")
    Image.fromarray(hole.astype(np.uint8) * 255).save(tmp_path / "hole.png")
    N = normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), None).numpy()
    _, (sr, sc), origin, _ = write_seed(tmp_path / "probe", np.where(known[..., None], P, -1.0), known, hole)
    Pt, Vt = resample_grid(as_t(P + 2.0 * N, "cpu", torch.float64), as_t(V, "cpu", torch.bool), sr, sc, origin=origin)
    G = np.full((Pt.shape[0] + 4, Pt.shape[1] + 4, 3), -1.0)
    VG = np.zeros(G.shape[:2], bool)
    G[2:-2, 2:-2], VG[2:-2, 2:-2] = Pt.numpy(), Vt.numpy()
    tifxyz.write(tmp_path / "fakeG", G, VG, {"scale": [0.05, 0.05], "grid_offset": [2, 2]})
    fake = tmp_path / "AppRun"
    fake.write_text("#!" + sys.executable + "\nimport shutil, sys\nt = sys.argv[sys.argv.index('-t') + 1]\n"
                    f"shutil.copytree({str(tmp_path / 'fakeG')!r}, t, dirs_exist_ok=True)\n")
    fake.chmod(0o755)
    (tmp_path / "grids").mkdir()
    out = tmp_path / "grown"
    r = subprocess.run([sys.executable, "-m", "surfnet.cli", "grow", "--sheet", tmp_path / "sheet", "--known",
                        tmp_path / "known.png", "--solve", tmp_path / "hole.png", "--m7", "m7.zarr", "--normal-grids",
                        tmp_path / "grids", "--pitch-um", "9.362", "--vc3d", fake, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "seed" / "generations.tif").exists()
    prm = json.loads((out / "growpatch_params.json").read_text())
    assert prm["generations"] > 4 and prm["normal_grid_path"] == str(tmp_path / "grids")
    Q, VQ, _ = tifxyz.read(out / "surface")
    assert VQ.shape == V.shape and (VQ == V).all()
    assert np.allclose(Q[known], P[known])
    off = np.abs(((Q - P) * N).sum(-1))[hole]
    # the phantom's winding has a radius of ~46 vox: between GrowPatch's 20-vox cells the interpolation sags up to ~1 vox
    assert json.loads((out / "report.json").read_text())["covered"] > 0.9 and np.abs(off - 2.0).max() < 1.5
    fail = tmp_path / "AppRunFails"                              # GrowPatch gives no surface: the area is bridged
    fail.write_text("#!" + sys.executable + "\nimport sys\nsys.exit(3)\n")
    fail.chmod(0o755)
    r = subprocess.run([sys.executable, "-m", "surfnet.cli", "grow", "--sheet", tmp_path / "sheet", "--known",
                        tmp_path / "known.png", "--solve", tmp_path / "hole.png", "--m7", "m7.zarr", "--normal-grids",
                        tmp_path / "grids", "--pitch-um", "9.362", "--vc3d", fail, "--out", out],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "WARNING" in r.stdout + r.stderr, r.stdout[-2000:] + r.stderr[-2000:]
    rp = json.loads((out / "report.json").read_text())
    Q, VQ, _ = tifxyz.read(out / "surface")
    assert rp["covered"] == 0 and rp["growpatch_failed"] == "exit 3" and not (out / "growpatch").exists()
    assert np.allclose(Q[hole], thin_plate_fill(np.where(known[..., None], P, -1.0), known, hole)[hole], atol=1e-3)


def test_grid_step_from_known_cells(tmp_path):
    """The grid step comes from the known cells. GrowPatch's seed: known on one side of a doubly curved sheet (a sphere of
    150 vox radius), the bridge beyond them is 5 % shorter along the known edge (the benchmark's one-sided areas: up to
    7.5 %), and the seed must still lie at GrowPatch's 20 vox. surfnet correct: a draft whose area is stretched by 10 %
    is not resampled."""
    from surfnet.common.geometry import edge_steps
    from surfnet.infer.correct import _to_model_grid
    from surfnet.infer.grow import STEP, write_seed
    from surfnet.infer.surface import thin_plate_fill
    c, R = PH.PhantomSpec().cell_vox, 150.0
    I, J = np.mgrid[0:40, 0:80].astype(np.float64)
    th, ph = (I - 20) * c / R, (J - 10) * c / R
    P = np.stack([200 + R * np.cos(th) * np.sin(ph), 200 + R * np.sin(th), 200 + R * np.cos(th) * np.cos(ph)], -1)
    V = np.ones(I.shape, bool)
    known = V.copy()
    known[:, 20:] = False
    area = V & ~known
    Pk = np.where(known[..., None], P, -1.0)
    bridge = edge_steps(thin_plate_fill(Pk, known, area), V)[0] / edge_steps(P, known)[0]
    write_seed(tmp_path / "seed", Pk, known, area)
    Q, VQ, _ = tifxyz.read(tmp_path / "seed")
    assert bridge < 0.96 and np.allclose(edge_steps(Q, VQ), STEP, rtol=0.01)
    for w in (2, 3, 4):                                           # a known strip 2-4 cells wide, wherever it starts on
        for c0 in range(20, 25):                                  # the 3.9-cell seed grid, is in the seed
            k = np.zeros(V.shape, bool)
            k[:, c0:c0 + w] = True
            assert write_seed(tmp_path / f"s{w}_{c0}", np.where(k[..., None], P, -1.0), k, V & ~k)[3] > 0, (w, c0)
    I, J = np.mgrid[0:30, 0:40].astype(np.float64)                # a flat sheet on the model's cell, known left of
    J[:, 12:] = 12 + 1.1 * (J[:, 12:] - 12)                       # column 12, the draft beyond it stretched 10 %
    D = np.stack([100 + c * J, 100 + c * I, np.full(I.shape, 100.0)], -1)
    VD = np.ones(I.shape, bool)
    kD = VD.copy()
    kD[:, 12:] = False
    assert _to_model_grid(D, VD, {}, c)[3] != (1.0, 1.0)            # measured on every cell: resampled
    _, V2, _, steps = _to_model_grid(D, VD, {}, c, kD)
    assert steps == (1.0, 1.0) and V2.shape == VD.shape


def test_add_sheet_and_sample(scroll, tmp_path, monkeypatch):
    monkeypatch.setenv("SURFNET_DATA", str(tmp_path))
    from surfnet.common.config import load
    from surfnet.data.bands import BandBuilder
    from surfnet.data.sheet import add_sheet
    from surfnet.data.store import BandSpec, BandStore
    from surfnet.train.pool import Pool
    from surfnet.train.sampler import Sampler
    d = scroll["dir"]
    s = add_sheet(tmp_path, "mysheet", d / "sheet", d / "known.png", str(d / "volume.zarr"), 9.362,
                  spec=BandSpec(fine_half=32, coarse_half=96, tile=16), device="cpu")
    assert (s.geom["tier"] == 1).sum() > 0 and (s.geom["tier"] == 0).sum() > 0
    BandBuilder(LocalZarrSource(d / "volume.zarr"), 0, device="cpu", fine_capacity=256, coarse_capacity=64,
                group=8).build([s])
    s = BandStore(s.root, "r")
    assert s.done.any() and s.fine.max() > 0
    cfg = load("finetune.yaml")
    pool = Pool([s], region=320, slots=2, device="cpu", seed=0, prefetch=False, min_usable=16)
    sm = Sampler(pool, cfg, "cpu", seed=0)
    L = sm.layout(1)
    bt = sm.build(L, L["T_A"])
    assert bt["x_ctx"].shape[1] == 13 and bt["x_fine"].shape[1] == 13


def test_score_any_grid():
    from surfnet.common.geometry import as_t
    from surfnet.eval.score import summary, surface_errors
    from surfnet.infer.surface import normals
    from surfnet.common.geometry import Axis
    spec = PH.PhantomSpec()
    ax = PH.axis_json(spec)
    axis = Axis(ax["z"], ax["y"], ax["x"])
    P, V = PH.winding_mesh(spec, 2)
    N = normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), axis).numpy()
    m = V.copy()
    m[:3], m[-3:], m[:, :3], m[:, -3:] = False, False, False, False
    coarse = PH.PhantomSpec(cell_vox=7.3)                        # another grid, the same sheet moved 3 vox
    Q, VQ = PH.winding_mesh(coarse, 2)
    NQ = normals(as_t(Q, "cpu", torch.float64), as_t(VQ, "cpu", torch.bool), axis).numpy()
    h = spec.spacing / 2                                        # 3 vox off: the right sheet, partial credit
    s = summary(surface_errors(P, N, m, Q + 3.0 * NQ, VQ), m, spacing=spec.spacing)
    assert s["coverage"] > 0.95 and abs(s["p50"] - 3.0) < 0.4 and s["wrong_sheet"] < 0.05
    assert abs(s["score"] - (h - 3.0) / (h - 2.0)) < 0.1
    Q3, V3 = PH.winding_mesh(coarse, 3)                         # the next winding: covered, but the wrong sheet
    s3 = summary(surface_errors(P, N, m, Q3, V3), m, spacing=spec.spacing)
    assert s3["coverage"] > 0.9 and s3["wrong_sheet"] > 0.95 and s3["score"] < 0.05
    half = VQ.copy()
    half[:, : VQ.shape[1] // 2] = False                         # half missing: uncovered cells count as failures
    sh = summary(surface_errors(P, N, m, Q, half), m, spacing=spec.spacing)
    assert 0.3 < sh["coverage"] < 0.7 and sh["wrong_sheet"] > 0.3 and sh["score"] < 0.7


def test_credit():
    from surfnet.eval.score import credit
    e = np.array([0.0, -2.0, 4.25, 6.5, 9.0, np.nan])
    c = credit(e, np.isfinite(e), 13.0)                         # full within 2 vox, 0 at half the spacing (6.5)
    assert np.allclose(c, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0])


def test_two_way():
    from surfnet.eval.score import two_way
    spec = PH.PhantomSpec()
    P, V = PH.winding_mesh(spec, 2)
    m = np.zeros(V.shape, bool)
    m[5:15, 10:40] = True
    Q, VQ = PH.winding_mesh(PH.PhantomSpec(cell_vox=7.3), 2)
    r = two_way(P, V, m, Q, VQ)
    assert r["gt_to_method"]["mean"] < 1.0 and r["method_to_gt"]["mean"] < 1.0
    Q3, V3 = PH.winding_mesh(PH.PhantomSpec(cell_vox=7.3), 3)
    r3 = two_way(P, V, m, Q3, V3)
    assert r3["gt_to_method"]["mean"] > 8 and r3["gt_to_method"]["within_10vox"] < 0.5


def test_score_gpu():
    """The exact scorer (eval/score_gpu.py, here on the CPU): a sheet 3 vox off on another grid, the next winding,
    half of the sheet missing, and the two-way distances."""
    from surfnet.common.geometry import Axis, as_t
    from surfnet.eval.score import summary
    from surfnet.eval.score_gpu import score_surface, two_way_summary
    from surfnet.infer.surface import normals
    spec = PH.PhantomSpec()
    ax = PH.axis_json(spec)
    axis = Axis(ax["z"], ax["y"], ax["x"])
    P, V = PH.winding_mesh(spec, 2)
    N = normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), axis).numpy()
    m = V.copy()
    m[:3], m[-3:], m[:, :3], m[:, -3:] = False, False, False, False
    coarse = PH.PhantomSpec(cell_vox=7.3)
    Q, VQ = PH.winding_mesh(coarse, 2)
    NQ = normals(as_t(Q, "cpu", torch.float64), as_t(VQ, "cpu", torch.bool), axis).numpy()
    r = score_surface(P, N, V, m, m, Q + 3.0 * NQ, VQ, device="cpu")
    s = summary(r, m, spacing=spec.spacing)
    h = spec.spacing / 2
    assert s["coverage"] > 0.95 and abs(s["p50"] - 3.0) < 0.3 and s["wrong_sheet"] < 0.05
    assert abs(s["score"] - (h - 3.0) / (h - 2.0)) < 0.1
    tw = two_way_summary(r)
    assert abs(tw["gt_to_method"]["mean"] - 3.0) < 0.3 and abs(tw["method_to_gt"]["mean"] - 3.0) < 0.3
    Q3, V3 = PH.winding_mesh(coarse, 3)                         # the next winding: covered, but the wrong sheet
    s3 = summary(score_surface(P, N, V, m, m, Q3, V3, device="cpu"), m, spacing=spec.spacing)
    assert s3["coverage"] > 0.9 and s3["wrong_sheet"] > 0.95 and s3["score"] < 0.05
    half = VQ.copy()
    half[:, : VQ.shape[1] // 2] = False                         # half missing: uncovered cells count as failures
    sh = summary(score_surface(P, N, V, m, m, Q, half, device="cpu"), m, spacing=spec.spacing)
    assert 0.3 < sh["coverage"] < 0.7 and sh["wrong_sheet"] > 0.3 and sh["score"] < 0.7


def test_cross_section():
    from surfnet.common.geometry import Axis, as_t
    from surfnet.eval.render import cross_section
    from surfnet.infer.engine import Volume
    from surfnet.infer.surface import normals
    spec = PH.PhantomSpec()
    vol = Volume(ArraySource.from_volume(PH.make_volume(spec), n_levels=2), "cpu", 256, 64)
    ax = PH.axis_json(spec)
    P, V = PH.winding_mesh(spec, 2)
    N = normals(as_t(P, "cpu", torch.float64), as_t(V, "cpu", torch.bool), Axis(ax["z"], ax["y"], ax["x"])).numpy()
    W = V.shape[1]
    im = cross_section(vol, P, N, V, 10, {"truth": (np.zeros(W), (0, 255, 0)), "off": (np.full(W, 5.0), (255, 0, 0))},
                       device="cpu")
    assert im.size == (W * 5, 81)
    a = np.asarray(im)
    assert (a[40, :, 1] == 255).mean() > 0.5 and (a[35, :, 0] == 255).mean() > 0.5


def _sample_reference(cache, pts):
    """GPUChunkCache.sample as it was before the faster gather (corner by corner): the reference it must equal bit for
    bit."""
    shp = pts.shape[:-1]
    q = pts.reshape(-1, 3)
    base = torch.floor(q)
    frac = (q - base).float()
    base = base.long()
    C = cache.C
    nz, ny, nx = cache.grid
    out = torch.zeros(q.shape[0], dtype=torch.float32)
    for dz in (0, 1):
        wz = frac[:, 0] if dz else 1 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1 - frac[:, 1]
            for dx in (0, 1):
                wx = frac[:, 2] if dx else 1 - frac[:, 2]
                c = base + torch.tensor((dz, dy, dx))
                inb = ((c >= 0) & (c < torch.as_tensor(cache.shape))).all(-1)
                ci = torch.div(c, C, rounding_mode="floor")
                slot = cache.table[ci[:, 0].clamp(0, nz - 1), ci[:, 1].clamp(0, ny - 1), ci[:, 2].clamp(0, nx - 1)]
                ok = inb & (slot >= 0)
                li = (c - ci * C).clamp(0, C - 1)
                v = cache.slots[slot.clamp(min=0).long(), li[:, 0], li[:, 1], li[:, 2]].float()
                out += torch.where(ok, wz * wy * wx * v, torch.zeros_like(v))
    return out.reshape(shp)


def test_chunk_cache_sample_and_keys_match_reference():
    """The faster CT gather and chunk-key lookup give exactly what the earlier implementations gave: chunks missing
    from the cache, points outside the volume and on chunk borders included."""
    from surfnet.volume import GPUChunkCache
    rng = np.random.default_rng(0)
    src = ArraySource.from_volume(rng.integers(0, 256, (150, 170, 140), dtype=np.uint8), 1, chunk=32)
    cache = GPUChunkCache(src, 0, "cpu", capacity=200)
    keys = np.array([k for k in np.ndindex(*src.grid(0))])
    cache.ensure(keys[::2])                                       # every other chunk resident
    pts = torch.as_tensor(rng.uniform(-4, [154, 174, 144], (200_000, 3)), dtype=torch.float64)
    pts[:5000] = torch.round(pts[:5000])                          # exactly on voxel and chunk borders
    assert torch.equal(cache.sample(pts), _sample_reference(cache, pts))
    q = pts[::7]
    lo = torch.div(torch.floor(q - 9.0).long(), 32, rounding_mode="floor")
    hi = torch.div(torch.floor(q + 10.0).long(), 32, rounding_mode="floor")
    g = torch.as_tensor(cache.grid)
    lo, hi = torch.minimum(lo.clamp(min=0), g - 1), torch.minimum(hi.clamp(min=0), g - 1)
    ref = torch.unique(torch.cat([torch.stack([(lo, hi)[i][:, 0], (lo, hi)[j][:, 1], (lo, hi)[k][:, 2]], 1)
                                  for i in (0, 1) for j in (0, 1) for k in (0, 1)]), dim=0).numpy()
    assert np.array_equal(cache.keys_for(q, margin=9.0), ref)
