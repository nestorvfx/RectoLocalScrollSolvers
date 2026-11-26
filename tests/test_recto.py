"""CPU tests of the recto local solver on the synthetic scroll (tests/phantom.py): its command line end to end on a hole
(the PyTorch code, a local CT), and the checks of the sheet's grid."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phantom as PH  # noqa: E402


def write_level0(vol: np.ndarray, d: Path) -> Path:
    """vol as an OME-Zarr level of uint8, 128^3 raw chunks, '/' separator (as the open-data volumes are stored)."""
    C = 128
    d.mkdir(parents=True)
    (d / ".zarray").write_text(json.dumps({"chunks": [C] * 3, "compressor": None, "dtype": "|u1", "fill_value": 0,
                                           "filters": None, "order": "C", "shape": list(vol.shape), "zarr_format": 2,
                                           "dimension_separator": "/"}))
    for z in range(0, vol.shape[0], C):
        for y in range(0, vol.shape[1], C):
            for x in range(0, vol.shape[2], C):
                b = np.zeros((C, C, C), np.uint8)
                part = vol[z:z + C, y:y + C, x:x + C]
                b[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                f = d / str(z // C) / str(y // C) / str(x // C)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b.tobytes())
    return d


def sheet_inputs(spec, d: Path, hole):
    """winding 3's recto face as a user's inputs: the sheet (the hole's cells empty), the known cells, the hole."""
    from PIL import Image

    from recto_local_solver.cli import write_tifxyz
    P, V = PH.winding_mesh(spec, 3)
    area = np.zeros(V.shape, bool)
    area[hole] = True
    known = V & ~area
    write_tifxyz(d / "sheet.tifxyz", P, known, {"format": "tifxyz"})
    Image.fromarray(known.astype(np.uint8) * 255).save(d / "known.png")
    Image.fromarray(area.astype(np.uint8) * 255).save(d / "area.png")
    (d / "axis.json").write_text(json.dumps(PH.axis_json(spec)))
    return P, known, area


def test_solve_a_hole(tmp_path, monkeypatch):
    """the frame of a hole 10 x 12 cells, from the cells around it, keeps to the winding (the right sheet: less than
    half the winding spacing from the truth, as the benchmark counts it); a rerun into the same folder replaces the
    result."""
    import tifffile

    from recto_local_solver import cli
    monkeypatch.setenv("RECTO_KERNELS", "0")
    spec = PH.PhantomSpec()
    vol = write_level0(PH.make_volume(spec), tmp_path / "volume.zarr" / "0")
    P, known, area = sheet_inputs(spec, tmp_path, (slice(6, 16), slice(30, 42)))
    args = ["solve", "--sheet", str(tmp_path / "sheet.tifxyz"), "--known", str(tmp_path / "known.png"), "--solve",
            str(tmp_path / "area.png"), "--volume", str(vol), "--axis", str(tmp_path / "axis.json"), "--out",
            str(tmp_path / "out"), "--frame-only"]
    for _ in range(2):
        cli.main(args)
    rec = json.loads((tmp_path / "out" / "solve.json").read_text())
    assert rec["resampled"] == [1.0, 1.0] and rec["solved"] == int(area.sum())
    Q = np.stack([tifffile.imread(tmp_path / "out" / "frame.tifxyz" / f"{a}.tif") for a in "xyz"], -1)
    err = np.linalg.norm(Q - P, axis=-1)[area]
    assert np.mean(err < 0.5 * spec.spacing) > 0.9, np.quantile(err, [0.5, 0.9])
    assert np.array_equal(Q[known], P[known].astype(np.float32))
    assert not (tmp_path / "out" / "frame_fibre_step.tifxyz").exists()


def test_solver_grid():
    """a sheet on another step is resampled to the solver's (a 2x coarser grid: about twice the cells per axis); one
    within 10 % of it is left as it is; one whose rows run around the scroll is refused."""
    from recto_local_solver.cli import CELL_VOX, solver_grid
    spec = PH.PhantomSpec()
    P, V = PH.winding_mesh(spec, 3)
    known = V.copy()
    known[6:16, 30:42] = False
    area = V & ~known
    out = solver_grid(P, V, known, area, {}, log=lambda *_: None)
    assert out[-1] == (1.0, 1.0) and out[0] is P
    P2, V2, k2, a2, meta, steps = solver_grid(P[::2, ::2], V[::2, ::2], known[::2, ::2], area[::2, ::2], {},
                                              log=lambda *_: None)
    assert steps[0] == pytest.approx(0.5, abs=0.01) and steps[1] == pytest.approx(0.5, abs=0.01)
    assert all(abs(n2 - 2 * n) <= 2 for n2, n in zip(V2.shape, V[::2, ::2].shape))
    assert meta["scale"] == [1.0 / CELL_VOX] * 2 and k2.any() and a2.any() and not (k2 & a2).any()
    with pytest.raises(SystemExit):
        solver_grid(P.transpose(1, 0, 2), V.T, known.T, area.T, {}, log=lambda *_: None)
