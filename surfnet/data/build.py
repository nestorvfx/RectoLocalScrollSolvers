"""The training dataset from the official open data: sources -> fetch -> prep -> bands. Everything lands under
$SURFNET_DATA:

    catalog.json                 resolved official objects (keys in the bucket, volumes, levels, splits, tiers)
    meshes/<oid>/{mesh,dense,labels}/   downloaded tifxyz meshes and label zarrs
    objects/<oid>/               band stores (geometry + bands)
    cache/<volume>/<level>/...   chunk cache (transient with --transient: chunks deleted after their last use)

Each step is idempotent (existing outputs are kept unless --force)."""
from __future__ import annotations

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common import tifxyz
from ..common.config import data_root, load
from ..common.log import get_logger
from . import axis as AX
from . import densify as DN
from . import neighbours as NB
from . import spacing as SP
from . import turns as TU
from .objects import build_object, level_coords
from .store import SPLIT_HELDOUT, SPLIT_TRAIN, BandSpec, BandStore
from .s3 import S3

log = get_logger("pipeline")


def _cfg(a):
    cfg = load(a.config)
    band = load(cfg.get("band", "bands.yaml"))
    return cfg, band


def _want(oid, only):
    return not only or any(fnmatch.fnmatch(oid, p) for p in only)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cell_vox(cell_um: float, c: dict) -> float:
    """The common cell (um) in processing-level voxels of the object's volume."""
    return cell_um / (float(c["pitch_um"]) * 2 ** int(c["level"]))


def _level_for_pitch(p: float, fine_level: int) -> int:
    """The pyramid level closest to the ~9.4 um processing pitch (2.4 um -> 2, 1.13 um -> 3, 9.36 um -> 0)."""
    import math
    return max(0, int(round(math.log2(9.4 / p))))


# ------------------------------------------------------------------------------------------------ sources
def cmd_sources(a):
    cfg, _ = _cfg(a)
    s3 = S3(cfg["bucket"])
    cat = []
    for gname, g in cfg["groups"].items():
        k = g["kind"]
        if k == "windings":
            seg_re = re.compile(rf"^{g['sample']}/segments/\d+-w(\d{{3}})_[^/]*/$")
            lo, hi = g["windings"]
            seen = {}
            for d in s3.dirs(f"{g['sample']}/segments/"):
                m = seg_re.match(d)
                if not m:
                    continue
                w = int(m.group(1))
                if not lo <= w <= hi or w in g.get("exclude_windings", []):
                    continue
                meshes = s3.dirs(d + "mesh/")
                coarse = [x for x in meshes if f"-on-{g['volume_id']}-" in x]
                if not coarse:
                    continue
                dense = [x for x in meshes if f"-on-{g.get('dense_volume_id', '-')}-" in x]
                if w in seen:                          # two folders for one winding: keep the newer
                    log.info(f"{gname}: w{w:03d} twice ({seen[w]['mesh']} / {coarse[0]}); keeping the newer")
                seen[w] = {"oid": f"{g['sample']}_w{w:03d}", "group": gname, "kind": k, "scroll": g["sample"],
                           "winding": w, "mesh": coarse[0].rstrip("/"), "dense": dense[0].rstrip("/") if dense else None,
                           "volume": g["volume"], "level": g["level"], "pitch_um": g["pitch_um"], "tier": g["tier"],
                           "split": SPLIT_HELDOUT if w in g.get("heldout_windings", []) else SPLIT_TRAIN}
            cat += [seen[w] for w in sorted(seen)]
        elif k == "merged":
            cat.append({"oid": f"{g['sample']}_merged", "group": gname, "kind": k, "scroll": g["sample"],
                        "mesh": g["mesh"], "volume": g["volume"], "level": g["level"], "pitch_um": g["pitch_um"],
                        "tier": g["tier"]})
        elif k == "segments":
            for smp in g["samples"]:
                vols = {d.rstrip("/").rsplit("/", 1)[1].split("-")[0]: d.rstrip("/")
                        for d in s3.dirs(f"{smp}/volumes/")}
                for d in s3.dirs(f"{smp}/segments/"):
                    name = d.rstrip("/").rsplit("/", 1)[1]
                    if name in g.get("exclude", []):
                        continue
                    lab_root = [x for x in s3.dirs(d) if x.endswith("ink-labels/")]
                    if g.get("require_labels") and not lab_root:
                        continue
                    for lv in s3.dirs(lab_root[0]) if lab_root else []:
                        m = re.match(r".*/ink-labels/([0-9.]+)um-volume-(\d+)/$", lv)
                        if not m:
                            continue
                        pitch, volid = float(m.group(1)), m.group(2)
                        versions = sorted(s3.dirs(lv))
                        meshes = [x for x in s3.dirs(d + "mesh/") if f"-on-{volid}-" in x]
                        if not versions or not meshes or volid not in vols:
                            continue
                        cat.append({"oid": f"{smp}_{name}_{volid}", "group": gname, "kind": k, "scroll": smp,
                                    "mesh": meshes[0].rstrip("/"), "labels": versions[-1].rstrip("/"),
                                    "volume": vols[volid], "pitch_um": pitch,
                                    "level": _level_for_pitch(pitch, g.get("level_fine_scan", 2)),
                                    "tier": g["tier"], "label_tier": g["label_tier"]})
    out = data_root() / "catalog.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cat, indent=1))
    by = defaultdict(int)
    for c in cat:
        by[c["group"]] += 1
    log.info(f"SOURCES_DONE {len(cat)} objects {dict(by)} -> {out}")


def _catalog(a) -> list[dict]:
    cat = json.loads((data_root() / "catalog.json").read_text())
    return [c for c in cat if _want(c["oid"], a.only)]


# ------------------------------------------------------------------------------------------------ fetch
def cmd_fetch(a):
    cfg, _ = _cfg(a)
    s3 = S3(cfg["bucket"])
    tot = 0
    for c in _catalog(a):
        base = data_root() / "meshes" / c["oid"]
        tot += s3.get_prefix(c["mesh"], base / "mesh")
        if c.get("dense"):
            tot += s3.get_prefix(c["dense"], base / "dense")
        if c.get("labels"):
            tot += s3.get_prefix(c["labels"], base / "labels")
        log.info(f"fetched {c['oid']}")
    log.info(f"FETCH_DONE {tot / 1e9:.2f} GB")


# ------------------------------------------------------------------------------------------------ prep
def _supervision_cells(lab_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Cells holding supervised label pixels (supervision.zarr, else inklabels.zarr non-zero), block-max to the grid."""
    import zarr
    for name in ("supervision.zarr", "inklabels.zarr"):
        p = lab_dir / name
        if not p.exists():
            continue
        z = zarr.open(str(p), mode="r")
        arr = z["0"] if hasattr(z, "keys") and "0" in z else z
        a = np.asarray(arr[...])
        a = a.reshape(a.shape[-2:]) if a.ndim > 2 else a
        H, W = shape
        fy, fx = a.shape[0] / H, a.shape[1] / W
        if abs(fy - fx) / max(fx, 1e-6) > 0.05:
            log.info(f"labels {p}: shape {a.shape} does not match the grid {shape} (factors {fy:.2f}, {fx:.2f})")
            return None
        ys = np.minimum((np.arange(H) * fy).astype(int), a.shape[0] - 1)
        xs = np.minimum((np.arange(W) * fx).astype(int), a.shape[1] - 1)
        f = max(1, int(round(fy)))
        from scipy.ndimage import maximum_filter
        m = maximum_filter((a > 0).astype(np.uint8), size=f) > 0
        return m[ys[:, None], xs[None, :]]
    return None


def cmd_prep(a):
    cfg, band = _cfg(a)
    spec = BandSpec.from_dict(band["spec"])
    root = data_root()
    dev = _device()
    cat = _catalog(a)
    cell_um = float(cfg["cell_um"])
    K = int(cfg.get("neighbours_K", 3))
    for gname, g in cfg["groups"].items():
        items = [c for c in cat if c["group"] == gname]
        if not items:
            continue
        if g["kind"] == "windings":
            loaded = {}
            for c in items:
                d = root / "meshes" / c["oid"]
                P, V, _ = tifxyz.read(d / "mesh")
                info = {"densified": False}
                if c.get("dense") and (d / "dense").exists():
                    Pd, Vd, _ = tifxyz.read(d / "dense")
                    r = DN.densify(Pd, Vd, P, V)
                    if r is not None:
                        P, V, info = r[0], r[1], {"densified": True, **r[2]}
                loaded[c["winding"]] = (c, level_coords(P, c["level"]), V, info)
            axis = AX.from_meshes([(P, V) for _, P, V, _ in loaded.values()])
            stores = {}
            for w, (c, P, V, info) in loaded.items():
                meta = {"scroll": c["scroll"], "volume": c["volume"], "level": c["level"], "pitch_um": c["pitch_um"],
                        "cell_um": cell_um, "mesh": c["mesh"], "winding": w, "axis_trusted": True, **info}
                tier = c["tier"] if info.get("densified") else max(c["tier"], 3)   # 187-um source cells: sigma >= 1 vox
                stores[w] = build_object(root, c["oid"], P, V, axis=axis, cell_vox=_cell_vox(cell_um, c),
                                         spec=spec, meta=meta, tier=tier, split=c["split"], winding=w, device=dev,
                                         overwrite=a.force)
            objs = {w: {"P": s.geom["P"].astype(np.float64), "N": s.geom["N"].astype(np.float64), "V": s.geom["V"]}
                    for w, s in stores.items()}
            nbr = NB.windings(objs, axis, K=K, device=dev)
            ks = list(range(-K, 0)) + list(range(1, K + 1))
            held = {w for w, (c, _, _, _) in loaded.items() if c["split"] == SPLIT_HELDOUT}
            for w, s in stores.items():
                if w not in held:                     # training windings never put their input on a held-out winding
                    for i, k in enumerate(ks):
                        if w + k in held:
                            nbr[w][i] = np.nan
                s.update_geom(nbr=nbr[w])
                s.update_meta(winding_index_offsets=ks)
                s.update_meta(spacing_vox=SP.from_neighbours(s) or 13.0)
                log.info(f"prepared {s.meta['id']}: {s.H} x {s.W} cells, spacing {s.meta['spacing_vox']:.1f} vox, "
                         f"densified {s.meta.get('densified')}")
        elif g["kind"] == "merged":
            c = items[0]
            d = root / "meshes" / c["oid"]
            P, V, _ = tifxyz.read(d / "mesh")
            P = level_coords(P, c["level"])
            axis = AX.from_meshes([(P, V)])
            meta = {"scroll": c["scroll"], "volume": c["volume"], "level": c["level"], "pitch_um": c["pitch_um"],
                    "cell_um": cell_um, "mesh": c["mesh"], "axis_trusted": True}
            s = build_object(root, c["oid"], P, V, axis=axis, cell_vox=_cell_vox(cell_um, c),
                             spec=spec, meta=meta, tier=c["tier"], split=SPLIT_TRAIN, device=dev, overwrite=a.force)
            Pg, Vg, Ng = s.geom["P"].astype(np.float64), s.geom["V"], s.geom["N"].astype(np.float64)
            Theta = TU.unwrap(Pg, Vg, axis)
            out = TU.outward_sign(Pg, Vg, Theta, axis)
            lo, hi = np.nanmin(Theta), np.nanmax(Theta)
            f0, f1 = g.get("heldout_turn_fraction", [0.8, 0.86])
            held = np.isfinite(Theta) & (Theta >= lo + f0 * (hi - lo)) & (Theta < lo + f1 * (hi - lo))
            split = np.where(held, SPLIT_HELDOUT, SPLIT_TRAIN).astype(np.uint8)
            winding = np.where(np.isfinite(Theta), np.floor((Theta - lo) / (2 * np.pi)), -1).astype(np.int16)
            nbr = NB.merged(Pg, Ng, Vg, Theta, out, axis, K=K, device=dev)
            ks = list(range(-K, 0)) + list(range(1, K + 1))
            h0, h1 = lo + f0 * (hi - lo), lo + f1 * (hi - lo)
            for i, k in enumerate(ks):                 # training cells never put their input on the held-out turns
                tgt = Theta + 2 * np.pi * k * out
                into = np.isfinite(Theta) & ~held & (tgt >= h0) & (tgt < h1)
                nbr[i][into] = np.nan
            s.update_geom(nbr=nbr, split=split, winding=winding)
            s.update_meta(winding_index_offsets=ks, outward=int(out))
            s.update_meta(spacing_vox=SP.from_neighbours(s) or 13.0)
            log.info(f"prepared {s.meta['id']}: {s.H} x {s.W} cells, held-out {held.mean():.3f}, spacing "
                     f"{s.meta['spacing_vox']:.1f} vox")
        elif g["kind"] == "segments":
            from ..volume import S3Source
            for c in items:
                d = root / "meshes" / c["oid"]
                P, V, _ = tifxyz.read(d / "mesh")
                sup = _supervision_cells(d / "labels", V.shape) if (d / "labels").exists() else None
                tier = np.full(V.shape, c["tier"], np.uint8)
                if sup is not None:
                    tier[sup] = c["label_tier"]
                src = S3Source(f"{cfg['bucket']}/{c['volume']}", root / "cache" / Path(c["volume"]).name)
                try:
                    axis = AX.from_volume(src, c["level"])
                except Exception as e:
                    log.info(f"{c['oid']}: no axis from the volume ({e}); normals keep the mesh's sign")
                    axis = None
                P = level_coords(P, c["level"])
                meta = {"scroll": c["scroll"], "volume": c["volume"], "level": c["level"], "pitch_um": c["pitch_um"],
                        "cell_um": cell_um, "mesh": c["mesh"], "labels": c.get("labels"),
                        "labelled_cells": int(sup.sum()) if sup is not None else 0}
                s = build_object(root, c["oid"], P, V, axis=axis, cell_vox=_cell_vox(cell_um, c),
                                 spec=spec, meta=meta, tier=tier, split=SPLIT_TRAIN, device=dev, overwrite=a.force)
                log.info(f"prepared {s.meta['id']}: {s.H} x {s.W} cells, labelled {meta['labelled_cells']}")
    log.info("PREP_DONE")


# ------------------------------------------------------------------------------------------------ bands
def cmd_bands(a):
    from ..volume import LocalZarrSource, S3Source
    from .bands import BandBuilder
    cfg, band = _cfg(a)
    bc = band.get("builder", {})
    root = data_root()
    stores = [BandStore(p, "r+") for p in sorted((root / "objects").iterdir()) if BandStore.exists(p)
              and _want(p.name, a.only)]
    groups = defaultdict(list)
    for s in stores:
        groups[(s.meta["volume"], int(s.meta["level"]))].append(s)
    dev = _device()
    # parallel builds of different volumes: smaller caches / budget per process via the environment;
    # a lock per object so that a second process waits instead of building the same object
    import os
    import time as _t
    env = {k: int(os.environ[f"SURFNET_BAND_{k.upper()}"]) for k in ("fine_capacity", "coarse_capacity", "budget")
           if f"SURFNET_BAND_{k.upper()}" in os.environ}
    bc = dict(bc, **env)

    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    for (vol, level), ss in groups.items():
        for s in ss:
            lk = s.root / ".building"
            t0 = _t.time()
            while lk.exists():
                try:
                    pid = int(lk.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid == os.getpid() or not _alive(pid):
                    break
                if int(_t.time() - t0) % 300 < 30:
                    log.info(f"bands: {s.meta['id']} is being built by pid {pid}: waiting")
                _t.sleep(30)
        locks = [s.root / ".building" for s in ss]
        for lk in locks:
            lk.write_text(str(os.getpid()))
        try:
            url = ss[0].meta.get("volume_url") or f"{cfg['bucket']}/{vol}"      # own sheets: any S3 URL or local zarr
            if str(url).startswith("s3://"):
                src = S3Source(url, root / "cache" / Path(vol.rstrip("/")).name,
                               workers=int(bc.get("s5cmd_workers", 64)), transient=a.transient)
            else:
                src = LocalZarrSource(url)
            ss = [BandStore(s.root, "r+") for s in ss]                   # reopen: another process may have built tiles
            log.info(f"bands: {vol} level {level}: {len(ss)} objects")
            st = BandBuilder(src, level, device=dev, fine_capacity=int(bc.get("fine_capacity", 2048)),
                             coarse_capacity=int(bc.get("coarse_capacity", 768)), group=int(bc.get("group", 16)),
                             fetch_batch=int(bc.get("fetch_batch", 2048)), budget=int(bc.get("budget", 12000))).build(ss)
            log.info(f"bands: {vol}: {st}")
        finally:
            for lk in locks:
                lk.unlink(missing_ok=True)
    gt = SP.gt_median([BandStore(p, "r") for p in sorted((root / "objects").iterdir()) if BandStore.exists(p)])
    for s in stores:
        if "nbr" not in s.geom and not s.meta.get("spacing_fixed") and gt:
            s.update_meta(spacing_vox=gt, spacing_from="median of the GT objects' neighbour spacing")
    log.info("BANDS_DONE")
