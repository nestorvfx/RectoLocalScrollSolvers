"""SurfNet's command line (`surfnet`).

Correct a sheet
  surfnet correct   --ckpt M.pt --sheet S.tifxyz --volume V --pitch-um P --known K.png --out DIR [options]
  surfnet grow      --sheet S.tifxyz --known K.png --solve A.png --m7 M7 --normal-grids G --pitch-um P --vc3d APPRUN
                    --out DIR       GrowPatch over the area first; then correct DIR/surface with the same masks

Train on your own verified sheet (fine-tuning)
  surfnet add-sheet --id NAME --sheet S.tifxyz --verified G.png --volume V --pitch-um P [--rejected R.png] [--heldout H.png]
  surfnet bands     --only NAME
  surfnet train     --config finetune.yaml --init BASE.pt --run DIR --objects NAME

Train the base model on the official ground truth
  surfnet sources [--catalog configs/dataset/catalog.json] | fetch | prep | bands --transient | ctcheck | onsheet
  surfnet faces     --recorded configs/dataset/objects_meta.json
  surfnet train     --config train.yaml --run DIR

Test
  surfnet test      --ckpt M.pt --out R.json         the held-out suite (synthetic starts on held-out ground truth)

  surfnet export    --ckpt RUN/last.pt --out M.pt    inference weights only (EMA) + spec + config

Data root: $SURFNET_DATA (default ./surfnet_data). Configs named without a path are read from configs/."""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

# the on-sheet threshold of the recorded build: the 1 % quantile of the ratio on hand-verified PHerc0211 cells
RECORDED_ONSHEET_THR = 0.771


def _device(arg):
    import torch
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def _stores(only=None, mode="r+"):
    from .common.config import data_root
    from .data.store import BandStore
    root = data_root() / "objects"
    return [BandStore(p, mode) for p in sorted(root.iterdir()) if BandStore.exists(p)
            and (not only or any(fnmatch.fnmatch(p.name, q) for q in only))]


# ------------------------------------------------------------------------------------------------ use
def cmd_correct(a):
    from .infer.correct import FAST, Settings, correct_sheet
    run = dict(FAST) if a.mode == "fast" else {}
    run.update({k: v for k, v in (("runs", a.runs), ("tta", a.tta), ("passes", a.passes)) if v is not None})
    s = Settings(start=a.start, margin=a.margin, ahead=a.ahead, dmin=a.dmin, dmax=a.dmax, clip=a.clip, noise=a.noise,
                 seed=a.seed, **run)
    correct_sheet(a.ckpt, a.sheet, a.volume, a.pitch_um, a.known, a.out, solve_mask=a.solve, level=a.level,
                  axis=a.axis, flip=a.flip, settings=s, resolve=a.resolve, keep=a.keep_draft,
                  cache=a.cache, device=_device(a.device))


def cmd_grow(a):
    from .infer.grow import grow_sheet
    grow_sheet(a.sheet, a.known, a.solve, a.m7, a.normal_grids, a.pitch_um, a.vc3d, a.out, threads=a.threads,
               timeout=a.timeout)


# ------------------------------------------------------------------------------------------------ data
def cmd_add_sheet(a):
    from .common.config import data_root, load
    from .data.sheet import add_sheet
    from .data.store import BandSpec
    spec = BandSpec.from_dict(load(a.bands)["spec"])
    s = add_sheet(data_root(), a.id, a.sheet, a.verified, a.volume, a.pitch_um, rejected=a.rejected,
                  heldout=a.heldout, level=a.level, axis=a.axis, flip=a.flip, spacing_vox=a.spacing_vox, spec=spec,
                  device=_device(a.device), overwrite=a.force)
    g = s.geom
    print(f"object {a.id}: {s.H} x {s.W} cells, verified {int((g['tier'] > 0).sum())}, held out "
          f"{int((g['split'] == 1).sum())}; next: surfnet bands --only {a.id}")


def cmd_data(a):
    from .data import build
    if a.cmd == "sources" and a.catalog:              # the recorded build's exact bucket keys instead of a new listing
        import shutil

        from .common.config import data_root
        data_root().mkdir(parents=True, exist_ok=True)
        shutil.copy(a.catalog, data_root() / "catalog.json")
        print(f"catalog {a.catalog} -> {data_root() / 'catalog.json'}")
        return
    getattr(build, f"cmd_{a.cmd}")(a)


def cmd_ctcheck(a):
    from .common.log import get_logger
    from .data import ctcheck
    ctcheck.run(_stores(a.only), get_logger("ctcheck"), max_zero=a.max_zero)


def cmd_onsheet(a):
    from .common.log import get_logger
    from .data import onsheet
    thr = a.thr if a.thr is not None or a.ref else RECORDED_ONSHEET_THR
    onsheet.run(_stores(a.only), a.ref, get_logger("onsheet"), window=a.window, fp=a.fp, thr=thr)


def cmd_faces(a):
    import json

    from .common.log import get_logger
    from .data import faces
    log = get_logger("faces")
    if a.recorded:                    # the recorded build's face offsets and winding spacings, by object id
        rec = {m["id"]: m for m in json.loads(Path(a.recorded).read_text())}
        for s in _stores(a.only):
            r = rec.get(s.meta["id"])
            if r is None:
                log.info(f"faces {s.meta['id']}: not in {a.recorded}, unchanged")
                continue
            s.update_meta(face_offset_vox=float(r.get("face_offset_vox") or 0.0), spacing_vox=float(r["spacing_vox"]))
            log.info(f"faces {s.meta['id']}: face offset {s.meta['face_offset_vox']:+.2f}, spacing {s.meta['spacing_vox']:.1f}")
        return
    if not a.ref:
        raise SystemExit("faces: give --recorded FILE, or --ref OBJECT (an object with verified cells, tier 1)")
    faces.run(_stores((a.only or []) + [a.ref] if a.only else None), a.ref, log, min_corr=a.min_corr, apply=not a.dry)


# ------------------------------------------------------------------------------------------------ train / test
def cmd_train(a):
    from .common.config import data_root, load, set_dotted
    from .train.trainer import Trainer
    cfg = load(a.config, a.overrides)
    if not cfg["data"].get("root"):
        set_dotted(cfg, "data.root", str(data_root()))
    if a.init:
        set_dotted(cfg, "train.init", a.init)
    if a.objects:                     # fine-tuning: train on these objects, and monitor the fit on them ("fit" set)
        cfg["data"]["include"] = list(a.objects)
        for v in cfg.get("val", {}).get("sets", []):
            if v.get("name") == "fit":
                v["include"] = list(a.objects)
    Trainer(cfg, Path(a.run), _device(a.device), resume=a.resume).train()


def cmd_test(a):
    from .common.config import data_root
    from .common.log import get_logger
    from .eval.synthetic import Shim, run_tests
    log = get_logger("test")
    sh = Shim(a.ckpt, _device(a.device))
    sh.cfg["data"]["root"] = a.data or str(data_root())
    sets = [{"name": n, "include": i.split(","), "split": "any"} for n, i in
            (x.split("=", 1) if "=" in x else (x, x) for x in a.sets)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    run_tests(sh, sh.model, sets, Path(a.out), batches=a.batches, log=log)


def cmd_export(a):
    import torch
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    out = {"ema": st["ema"] if "ema" in st else st["model"], "spec": st["spec"], "cfg": st.get("cfg"),
           "step": st.get("step")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"{a.out}: step {out['step']}, {sum(v.numel() for v in out['ema'].values()) / 1e6:.1f} M values")


# ------------------------------------------------------------------------------------------------ parser
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="surfnet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("correct", help="correct a region of a sheet from its known-correct cells")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sheet", required=True, help="tifxyz directory (positions in the volume's level-0 voxels)")
    p.add_argument("--volume", required=True, help="s3:// URL or local OME-Zarr directory of the CT volume")
    p.add_argument("--pitch-um", type=float, required=True, help="voxel size of the volume's level 0 (um)")
    p.add_argument("--known", required=True, help="mask on the sheet's grid: cells known to be correct")
    p.add_argument("--solve", default=None, help="mask: cells to correct (default: every valid cell not known; with "
                   "--start bridge they may be empty)")
    p.add_argument("--out", required=True)
    p.add_argument("--start", choices=("draft", "bridge"), default="draft")
    p.add_argument("--mode", choices=("default", "fast"), default="default",
                   help="default: 5 runs, 2 flips, up to 3 refinement passes, confidence maps; fast: 4 runs, 1 flip, "
                        "1 pass, the BP iteration compiled (on a GPU), no confidence maps")
    p.add_argument("--resolve", action="store_true",
                   help="then solve again the cells whose runs differ by more than half a winding (6.25 vox), with 8 "
                        "runs, the other cells pinned")
    p.add_argument("--keep-draft", action="store_true",
                   help="then keep the draft (e.g. GrowPatch's surface from surfnet grow) on the cells whose runs still "
                        "differ by more than half a winding")
    p.add_argument("--runs", type=int, default=None, help="runs from perturbed starts; the median is kept (mode's)")
    p.add_argument("--tta", type=int, default=None, choices=(1, 2, 4, 8), help="flips / rotations per window (mode's)")
    p.add_argument("--passes", type=int, default=None, help="refinement passes at most (mode's)")
    p.add_argument("--margin", type=float, default=64.0)
    p.add_argument("--ahead", type=float, default=40.0)
    p.add_argument("--dmin", type=float, default=4.0)
    p.add_argument("--dmax", type=float, default=20.0)
    p.add_argument("--clip", type=float, default=20.0)
    p.add_argument("--noise", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--level", type=int, default=None, help="processing level (default: the one closest to 9.4 um)")
    p.add_argument("--axis", default=None, help="scroll axis json {z, y, x} in level-0 voxels (orients the normals)")
    p.add_argument("--flip", action="store_true", help="the sheet's grid has the opposite handedness")
    p.add_argument("--cache", default=None, help="chunk cache for S3 volumes (default OUT/cache)")

    p = sub.add_parser("grow", help="grow the area with GrowPatch (VC3D) along the m7, on the sheet's own grid")
    p.add_argument("--sheet", required=True, help="tifxyz directory (positions in the volume's level-0 voxels)")
    p.add_argument("--known", required=True, help="mask on the sheet's grid: cells known to be correct (the seed)")
    p.add_argument("--solve", required=True, help="mask on the sheet's grid: the area to grow over")
    p.add_argument("--m7", required=True, help="s3:// URL or local OME-Zarr of the m7 surface prediction")
    p.add_argument("--normal-grids", required=True, help="the m7's normal grids: local directory or s3:// prefix")
    p.add_argument("--pitch-um", type=float, required=True, help="voxel size of the volume's level 0 (um)")
    p.add_argument("--vc3d", required=True, help="VC3D's AppRun (benchmark/vc3d_setup.sh)")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=None, help="GrowPatch's CPU threads (default: all)")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds")

    p = sub.add_parser("add-sheet", help="your verified sheet as a training object")
    p.add_argument("--id", required=True)
    p.add_argument("--sheet", required=True)
    p.add_argument("--verified", required=True, help="mask on the sheet's grid: cells verified correct")
    p.add_argument("--volume", required=True)
    p.add_argument("--pitch-um", type=float, required=True)
    p.add_argument("--rejected", default=None)
    p.add_argument("--heldout", default=None)
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--axis", default=None)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--spacing-vox", type=float, default=13.0, help="winding spacing (vox at the processing level)")
    p.add_argument("--bands", default="bands.yaml")
    p.add_argument("--force", action="store_true")

    for name in ("sources", "fetch", "prep", "bands"):
        p = sub.add_parser(name, help="official dataset build" if name != "bands" else "CT bands of the objects")
        p.add_argument("--config", default="data.yaml")
        p.add_argument("--only", nargs="*", default=None, help="object id patterns")
        p.add_argument("--transient", action="store_true", help="delete downloaded chunks after their last use")
        p.add_argument("--force", action="store_true")
        if name == "sources":
            p.add_argument("--catalog", default=None, help="use a recorded catalog (configs/dataset/catalog.json)")
    p = sub.add_parser("ctcheck", help="exclude cells without CT")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-zero", type=float, default=0.5)
    p = sub.add_parser("onsheet", help="per cell: does the mesh lie on the papyrus? (targets only where it does)")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--thr", type=float, default=None, help=f"default: the recorded calibration {RECORDED_ONSHEET_THR}")
    p.add_argument("--ref", default=None, help="recalibrate on this object's verified (tier 1) cells instead")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--fp", type=float, default=0.01)
    p = sub.add_parser("faces", help="orientation and face offset of each object")
    p.add_argument("--recorded", default=None, help="apply a recorded build's offsets (configs/dataset/objects_meta.json)")
    p.add_argument("--ref", default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--min-corr", type=float, default=0.6)
    p.add_argument("--dry", action="store_true")

    p = sub.add_parser("train", help="train or fine-tune")
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--init", default=None, help="start from these weights (fine-tuning; fresh optimiser)")
    p.add_argument("--objects", nargs="*", default=None, help="fine-tuning: the object ids (from add-sheet) to train on")
    p.add_argument("--resume", default=None, help="auto | PATH: continue a run exactly")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    p = sub.add_parser("test", help="the held-out suite on objects with bands")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="*", default=["w058_w059=PHerc0139_w058,PHerc0139_w059", "PHerc0841=PHerc0841*"],
                   help="name=object-id patterns (comma-separated)")
    p.add_argument("--data", default=None)
    p.add_argument("--batches", type=int, default=2)
    p = sub.add_parser("export", help="inference weights only")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    fn = {"correct": cmd_correct, "grow": cmd_grow, "add-sheet": cmd_add_sheet, "sources": cmd_data, "fetch": cmd_data,
          "prep": cmd_data, "bands": cmd_data, "ctcheck": cmd_ctcheck, "onsheet": cmd_onsheet, "faces": cmd_faces,
          "train": cmd_train, "test": cmd_test, "export": cmd_export}[a.cmd]
    fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
