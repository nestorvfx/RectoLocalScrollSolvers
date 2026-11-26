# SurfNet and the recto local solver: guide

- [Quick start](#quick-start)

1. [The tools: surfnet correct and surfnet grow](#1-the-tools-surfnet-correct-and-surfnet-grow)
2. [The recto local solver](#2-the-recto-local-solver)
3. [Fine-tuning SurfNet on your own sheet](#3-fine-tuning-surfnet-on-your-own-sheet)
4. [Training SurfNet's base model](#4-training-surfnets-base-model)

Which to use, and the benchmark behind it: the [README](../README.md) and the [supplement](SUPPLEMENTARY.md).

## Quick start

From the repository's root.

```bash
bash setup.sh                       # Linux + NVIDIA GPU: a .venv with PyTorch, SurfNet and the recto local solver
source .venv/bin/activate           # (bash setup.sh --test also runs the CPU tests)
bash benchmark/vc3d_setup.sh vc3d   # VC3D, for GrowPatch
```

**Weights:** SurfNet's are on Hugging Face, [nestorvfx/surfnet](https://huggingface.co/nestorvfx/surfnet/tree/main),
91 MB each: `surfnet_base.pt`, the base model, and `surfnet_pherc0211.pt`, fine-tuned on PHerc0211
([the README](../README.md#fine-tuned-on-pherc0211)).

```bash
wget https://huggingface.co/nestorvfx/surfnet/resolve/main/surfnet_base.pt
```

The recto local solver has no weights. It runs its CUDA kernels on an NVIDIA GPU, and the same steps in PyTorch, more
slowly, elsewhere ([below](#2-the-recto-local-solver)).

**Try it on two benchmark cases.** `examples/` holds two benchmark cases, a 1.9 mm hole (in the README's Figure 1) and a 1.9 mm extension: a user's inputs and
the ground truth to score against. Each method takes under a minute per case on a T4.

```bash
CT=s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr
E=examples/hole_1.9mm                                                    # or examples/extension_1.9mm
recto-local-solver solve --sheet $E/sheet.tifxyz --known $E/known.png --solve $E/area.png --volume $CT/0 \
  --axis recto_local_solver/data/axis_PHerc0139.json --frame-only --out out/recto
surfnet correct --ckpt surfnet_base.pt --sheet $E/sheet.tifxyz --known $E/known.png --solve $E/area.png \
  --volume $CT --pitch-um 9.362 --start bridge --out out/surfnet
python benchmark/score_surface.py --case $E out/recto/frame.tifxyz out/surfnet/surface
```

The benchmark's scores, for comparison (`score_surface.py` prints the same score; on another GPU they can differ
slightly):

| example | recto local solver | SurfNet alone | GrowPatch | GrowPatch → SurfNet |
|---|---|---|---|---|
| `hole_1.9mm` | 91.1 | 92.2 | 68.2 | 88.4 |
| `extension_1.9mm` | 52.2 | 90.8 | 79.6 | 83.4 |

**Your own sheet.** All three methods take the same inputs, on the grid of one sheet:

| input | what |
|---|---|
| `sheet.tifxyz` | the sheet (VC3D tifxyz, level-0 voxels); the approved cells need positions, the area's cells may be empty |
| `known.png` | white = the approved cells: they stay in place and are the context the methods read |
| `area.png` | white = the area to place |

Mark the approved cells around the area; SurfNet reads those within 3 mm of it. In the benchmark, a 0.95 mm ring, a
strip 0.24-1.9 mm wide or scattered patches of approved cells were enough.

```bash
S3=s3://vesuvius-challenge-open-data/PHerc0139
M7=$S3/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2
CT=$S3/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr

# GrowPatch → SurfNet: GrowPatch grows the area along the m7; SurfNet corrects it, keeping GrowPatch's surface
# where its own runs disagree (--keep-draft)
surfnet grow --sheet sheet.tifxyz --known known.png --solve area.png --m7 $M7.zarr --normal-grids $M7.normal-grids \
  --pitch-um 9.362 --vc3d vc3d/squashfs-root/AppRun --out grown/
surfnet correct --ckpt surfnet_base.pt --sheet grown/surface --known known.png --solve area.png \
  --volume $CT --pitch-um 9.362 --mode fast --keep-draft --out corrected/

# the recto local solver
recto-local-solver solve --sheet sheet.tifxyz --known known.png --solve area.png --volume $CT/0 \
  --axis recto_local_solver/data/axis_PHerc0139.json --frame-only --out recto/

# SurfNet alone, for holes up to 3.8 mm across
surfnet correct --ckpt surfnet_base.pt --sheet sheet.tifxyz --known known.png --solve area.png \
  --volume $CT --pitch-um 9.362 --start bridge --out corrected/
```

- **SurfNet** writes `corrected/surface` (tifxyz), `corrected/maps.npz` and `corrected/report.json`. In `maps.npz`, the
  cells where SurfNet's runs disagree by more than half a winding are the ones to review first.
- **The recto local solver** writes `recto/frame.tifxyz` and `recto/solve.json`. Its optional fibre step (run it by
  leaving out `--frame-only`) did not help on this scroll. Another scroll needs its own axis and fibre depths
  ([below](#2-the-recto-local-solver)).
- **GrowPatch** uses every CPU core by default, and its surface then varies slightly from run to run. The benchmark ran
  it with `--threads 1`, which is reproducible.

Every option is in the sections below.

**Not tested: the recto local solver → SurfNet.** The solver's surface can be SurfNet's start in place of GrowPatch's:
`surfnet correct --sheet recto/frame.tifxyz --known known.png --solve area.png --volume $CT --pitch-um 9.362 --mode fast
--out corrected/`. It was not benchmarked.

## 1. The tools: surfnet correct and surfnet grow

| input | form |
|---|---|
| `--sheet` | a tifxyz directory (`x.tif`, `y.tif`, `z.tif`, `meta.json`) in the volume's level-0 voxels; invalid cells -1; any grid step |
| `--known` | a mask (PNG / TIF / NPY) on the sheet's grid, non-zero = known correct: pinned, and shown to the model as context |
| `--solve` | a mask of the area to recover. `correct`: default every valid cell not known; with `--start bridge` its cells may be empty (a hole). `grow`: required |
| `--volume` | the CT: an `s3://` OME-Zarr URL (fetched with s5cmd, cached in `--cache`) or a local OME-Zarr v2 directory, uint8 |
| `--pitch-um` | the voxel size of the volume's level 0 |

**`surfnet correct`** places the area's surface from the known cells.
- **Processing level and grid.** It reads the pyramid level closest to the model's 9.4 µm (`--level` overrides) and
  resamples the sheet to the model's cell of 47.46 µm (masks by nearest neighbour) when its step, measured on the known
  cells, differs from that by more than 5 %. The benchmark's sheets (up to 3.7 % off) were solved on their own grid.
- **Normals** follow the grid's orientation (d/drow × d/dcol), toward the recto for VC3D meshes. `--flip` reverses
  them; `--axis` (a JSON `{z, y, x}` polyline in level-0 voxels) points them at the scroll axis.
- **Precision:** fp16, as in every benchmark run (environment variable `SURFNET_AMP`: `bf16` or `fp32` instead).

| option | default | meaning |
|---|---|---|
| `--start` | `draft` | `draft`: start from the sheet's own (smoothed) area cells, e.g. GrowPatch's surface. `bridge`: bridge the area from the known cells only (thin-plate) |
| `--mode` | `default` | `default`: 5 runs, 2 flips, up to 3 refinement passes, confidence maps. `fast`: 4 runs, 1 flip, 1 pass, the solver compiled (on a GPU), no confidence maps |
| `--resolve` | off | then solve again, with 8 runs, the cells whose runs differ by more than half a winding (6.25 vox at 9.4 µm), the others pinned |
| `--keep-draft` | off | then keep the draft (e.g. GrowPatch's surface from `surfnet grow`) on the cells whose runs still differ by more than half a winding: with `--mode fast` on GrowPatch's surface, the recommended way |
| `--runs`, `--tta`, `--passes` | the mode's | runs from differently perturbed starts (the per-cell median is kept); flips / rotations of each window averaged (1, 2, 4, 8); refinement passes at most (a run stops once its frame moves < 0.25 vox) |
| `--dmin`, `--dmax`, `--clip`, `--noise` | 4, 20, 20, 0.6 | each run's start perturbation, vox: \|A\| in [dmin, dmax], 0 at the known cells, clipped |
| `--margin` | 64 | known cells within this many cells (3 mm) of the area join the solve |
| `--ahead` | 40 | other valid cells within this distance are solved as context but not kept (`draft` only; the benchmark's sheets had none: they held the known cells and the area only) |

| output | content |
|---|---|
| `surface/` | tifxyz on the model grid, in level-0 voxels; the known cells unchanged |
| `maps.npz` | `move` (vox along the input's normals), `spread` (max - min over the runs: large where they disagree), `known`, `solve`; `confidence` and `confidence_input` (the fine head's mass within 1 vox of the result and of the input) in the default mode |
| `report.json` | the settings, cell counts, times and summary numbers |

**Reach.** From the known cells alone (`--start bridge`), SurfNet scored 96.0 on the benchmark's holes up to 3.8 mm
across, with none of them failed (README, Results). Farther from known cells, and above all beyond a known edge, its start (the
bridge) can be several windings off and the result can follow a neighbouring winding. `correct` warns when an area's
cells lie more than 1.9 mm (40 cells) from every known cell.

**Memory.** Belief propagation works on the bounding box of the solved cells, so memory grows with that box.

**`surfnet grow`** runs GrowPatch (VC3D `vc_grow_seg_from_seed --resume`, `benchmark/vc3d_setup.sh`) along the m7
surface prediction:
- the seed: the known cells on GrowPatch's 20-vox grid, which is the sheet's grid resampled to that step as measured on
  the known cells, with a node on the first known row and column (a known strip two cells wide is always in the seed)
  and the area bridged from the known cells (so it reaches the area's edge); as many generations as
  the farthest area cell needs, plus 3; growth kept to that grid (no grid expansion), its positions not bounded (a box
  around the bridge would cut it off where the sheet curves away from the bridge);
- GrowPatch keeps its seed's grid and grows it at the same step, so its surface goes back onto the sheet's grid cell
  for cell (bilinear); area cells it did not reach are bridged, and where it gives no surface at all (it failed, timed
  out, or no known cell lies on its seed grid) the whole area is, with a warning;
- outputs: `surface/` (the sheet's grid: the same masks apply), `growpatch/` (GrowPatch's own), `report.json`
  (seconds, the share of the area GrowPatch covered);
- `--threads` (default: all cores): with several threads GrowPatch's surface differs from run to run; with
  `--threads 1` (the benchmark's setting) it is the same on every run, and slower.

`--m7` is the m7 OME-Zarr (s3 or local), `--normal-grids` its normal grids (a local directory or an s3 prefix,
downloaded once), `--vc3d` the AppRun.

## 2. The recto local solver

```bash
recto-local-solver solve --sheet sheet.tifxyz --known known.png --solve area.png --volume $CT/0 --axis axis.json \
  --out recto/                        # or: python -m recto_local_solver solve ...
```

| option | default | meaning |
|---|---|---|
| `--sheet` | | a tifxyz directory in the volume's level-0 voxels; invalid cells -1. Only the known cells' positions are read. Its rows must run along the scroll's height (checked); a grid step more than 10 % from 5.07 voxels is resampled (below) |
| `--known` | | a mask (PNG / TIF / NPY) on the sheet's grid, non-zero = known correct: pinned, calibrated on, and the fibres' source |
| `--solve` | | a mask of the area to place; its cells may be empty |
| `--volume` | | the CT's level 0: an OME-Zarr array of uint8, raw 128³ chunks, `/` separator, as the open-data volumes are: an `s3://` URL (fetched with s5cmd) or a local copy (read in place) |
| `--axis` | | the scroll axis, JSON `{z, y, x}` in level-0 voxels; the normals point toward it |
| `--focus` | `1.0,2.0` | the depths (voxels along the normals) at which a known surface's horizontal and vertical fibres show best, per scroll |
| `--context` | 80 | known columns beside the area that the fibre step reads (3.8 mm) |
| `--frame-only` | off | the frame alone, no fibre step |
| `--cache` | `OUT/cache` | the CT chunk cache on disk (an `s3://` volume) |
| `--gpu-cache-gb` | 8, at most 40 % of the GPU | the CT chunks held on the GPU (a batch beyond it empties the cache first) |

**Hardware.** On an NVIDIA GPU with PyTorch's CUDA build and CuPy (setup.sh) the solver runs its CUDA kernels (compiled
at run time for the GPU in use) and its sparse solve with NVIDIA cuDSS (nvmath-python). Without them it runs the same
steps in PyTorch, and the sparse solve with SciPy (`RECTO_KERNELS=0` forces this). The kernels match their PyTorch
code to a checked tolerance, not bit for bit. The fallback was measured on a T4 GPU, on 5 of the benchmark's cases
drawn at random (up to 160 × 160 cells; `docs/results/recto_fallback.json`): 2.9 times as long for the frame and 3.4
times for the frame + fibre step, its scores within 2.1 points of the kernels' on the same cases, its surfaces a
median 0-0.45 voxels from theirs (at most 2 % of the cells more than 6 voxels apart). It was also checked on 3 cases
while it was built, and the tests run it on the CPU on a synthetic scroll. It was written to run on other GPUs too
(AMD's with PyTorch's ROCm build), not tried there.

Environment variables: `RECTO_DEVICE` (`cuda` or `cpu`; default the GPU when PyTorch sees one), `RECTO_KERNELS=0` (the
PyTorch code instead of the CUDA kernels), `RECTO_KERNELS_CHECK=1` (each kernel's first call also checked against its
PyTorch code, a line per kernel in the log; or a list of kernel names), `RECTO_GPU_CACHE_GB` (as `--gpu-cache-gb`),
`RECTO_S5CMD` (the s5cmd binary).

| output | content |
|---|---|
| `frame.tifxyz` | the frame, on the sheet's grid (the solver's, if it was resampled); the known cells unchanged |
| `frame_fibre_step.tifxyz` | the frame + the fibre step |
| `solve.json` | `calib` (the matched filter's positive and negative samples, its separation d′), `cells`, `solved`, `bridged` (cells the growth did not reach, filled smoothly), `patches` (the growth's patches with and without the gate), seconds per step, the fibre step's `window`, `fibres` found and used, `moved_p50` (its median move, voxels), `grid` and `resampled` (the grid solved on, and its step in the sheet's cells) |

A rerun into the same `--out` replaces these.

**The grid.** One cell per 5.07 level-0 voxels (47.46 µm on a 9.362 µm scan), the horizontal fibres along the grid's
columns and the scroll's height along its rows, as VC3D flattens sheets. The solver works in voxels on that grid. A
sheet whose step, measured on the known cells, lies more than 10 % from it (GrowPatch's own segments: 20 voxels) is
first resampled to it, as `surfnet correct` resamples (positions bicubic, masks nearest), and the results are written on
that grid; the benchmark's sheets (4.9-5.2 voxels) were solved on their own. A sheet whose rows run around the scroll is
refused: transpose it and its masks.

**Another scroll.**
- The axis: `surfnet.data.axis.from_volume` estimates one from the CT (the centroid per height of a coarse level):

  ```python
  from surfnet.volume import open_volume
  from surfnet.data.axis import from_volume
  from_volume(open_volume("s3://.../volume.zarr", "cache"), level=0).to_json("axis.json")
  ```
- The fibre-focus depths: PHerc0139 1.0,2.0 (from the solver's detector on known cells of the benchmark), PHerc0211
  1.0,4.3 (from approved surfaces).

**How it works** (`recto_local_solver/`):
1. **Calibration** (`solve.calibrate`): a Fisher matched filter of the known cells' depth profile (the vertical-fibre
   ply, the horizontal-fibre ply, the intensity) against the other depths, from the cells at least 3 cells inside the
   known edge.
2. **The frame** (`hsolve.frame_strip` → `sheetflat.flatgrow` → `sheetsolve.solve_patch`): the area is grown from the
   known edge in patches, each reaching 16 cells beyond the grown edge and keeping the 8 nearest it, each predicted from
   the grid around it and solved as a whole along its normals (±24 voxels, several windings): a coarse level by belief
   propagation on 5 × 5-cell bins, a fine level by semi-global matching, then a continuous refinement and up to 5 outer
   passes on the moved surface. The data term is the matched filter and the fibre-focus term (the across-fibre pattern
   of neighbouring bins along the fibres, read at the focus depths); the pairwise term follows the CT's own lamination
   slopes (a sheared structure tensor). The known cells are pinned, every patch re-solves all the cells grown so far,
   and cells that ramp across layers (a slant gate, 6°) are held back. Cells not reached within 14 patches (112 cells)
   are grown again without the gate (up to 32 patches more), the rest bridged.
3. **The fibre step** (`hsolve.fibre_field_id`): the known cells and the area are read ±22 voxels around the frame
   (`hfibwin`, `hfib`: the stack straightened along the fibres, cleaned along them in 3-D). The thick fibres visible
   on the known cells' own surface are followed onto every surface near the frame (moved by up to ±18 voxels), and one
   semi-global labelling picks the surface on which they stay visible. The fibres tracked in 3-D within -3 to +4
   voxels of it then move the surface through their centres, never onto the dark gap between windings; the move is
   faded in over 8 cells from the known cells.

**Time** (T4 GPU), roughly linear in the area: 25-42 s for 160 × 160 cells (7.6 × 7.6 mm), 81 s for 250 × 250 cells,
the CT already on disk. Over the benchmark's 80 cases, the CT download included: 39 s per case for the frame and 48 s
for the frame + fibre step (median 24 and 32 s; at most 6-7 min, a 15.2 mm extension).

## 3. Fine-tuning SurfNet on your own sheet

Fine-tuning on cells you verified on one sheet adapts the model to that scroll's CT and to your surface convention.

```bash
export SURFNET_DATA=/data/surfnet
surfnet add-sheet --id mysheet --sheet my_sheet.tifxyz --verified verified.png \
  --volume s3://.../my_volume.zarr --pitch-um 7.91 [--rejected wrong.png] [--heldout check.png]
surfnet bands --only mysheet
surfnet train --config finetune.yaml --init surfnet_base.pt --objects mysheet --run runs/ft_mysheet
surfnet export --ckpt runs/ft_mysheet/last.pt --out surfnet_mysheet.pt
```

Verified cells are the targets; other valid cells are context only; rejected cells are never targets; held-out cells
are never trained on and the validation reports them.

A test of one fine-tune, on five areas of a PHerc0211 sheet it never trained on: the README's
[Fine-tuned on PHerc0211](../README.md#fine-tuned-on-pherc0211) and [the supplement](SUPPLEMENTARY.md#5-fine-tuning-on-pherc0211).
In practice, on PHerc0211 (site 1, sheet 2):
- the base model was fine-tuned on the sheet's approved areas;
- it then corrected the rest area by area, every area approved by hand;
- as more of the sheet was approved it was fine-tuned again (the last: 6,000 steps on 60 approved tiles and 13
  quarter tiles).

Hold out a few approved areas (`--heldout`) to see whether the fit carries over to cells it did not train on.

`configs/finetune.yaml`: from the base weights with a fresh optimiser, learning rate 1.5e-4, 200 warm-up steps, 6,000
steps with a cooldown over the last third, EMA 0.998. About 40 minutes on a 20 GB GPU; the bands take about 37 MB per
cm².

## 4. Training SurfNet's base model

The dataset (38 training objects and the held-out ones, about 141 GB of bands, about 1.5 h):

```bash
export SURFNET_DATA=/data/surfnet
surfnet sources --catalog configs/dataset/catalog.json     # the recorded build's exact bucket keys
surfnet fetch                                              # meshes and label zarrs (2.3 GB)
surfnet prep                                               # geometry on 47.46 um cells, normals, tiers, splits, neighbours
surfnet bands --transient                                  # CT bands; chunks deleted after use
surfnet ctcheck                                            # excludes cells without CT
surfnet onsheet                                            # targets only where the mesh lies on the papyrus
surfnet faces --recorded configs/dataset/objects_meta.json # the recorded face offsets and winding spacings
surfnet train --config train.yaml --run runs/base
```

- Two calibrations come from hand-verified PHerc0211 cells, which are not public, so the build applies the recorded
  values: the on-sheet threshold (0.771) and the face offsets of the official meshes.
- 42,000 steps of batch 8: AdamW at 1e-3, warm-up 1,000, a linear cooldown to 1e-5 from step 35,000, EMA 0.999, bf16.
  About 9 h on an RTX A4500 (20 GB). `--resume auto` continues a run exactly.
- Validation every 5,000 steps on PHerc0139 w056-w057 and PHerc1667's held-out turns; it chose the checkpoint.
- Never trained on: PHerc0139 w056-w059, PHerc0841, PHerc1447, PHerc0343.

The training data, synthetic errors of official ground truth:

| data | objects | area |
|---|---|---|
| PHerc0139 ground-truth windings (w023-w040, w047-w055) | 27 | 1,081 cm² |
| PHerc1667 merged unwrap (31 wraps) | 1 | 1,212 cm² |
| ink-labelled segments (PHercParis4 × 7, PHerc0814, PHerc0009B, PHerc0500P2) | 10 | 991 cm² |
| **total** | **38** | **3,284 cm²** |
