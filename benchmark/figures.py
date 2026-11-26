"""The README's and the supplement's figures, from the benchmark's saved outputs (no SurfNet run).

  python figures.py example --cases DIR --results DIR ... --errors DIR --case NAME --methods G,GF4CmedK,RF --title T
                            --out F.jpg
      one case, one row per method, labelled with its score: a CT cross-section with the method's surface (thick: its
      colour where on the right sheet, darker where on the wrong one), the ground truth (white) and the known cells
      (green, in place of the method's line), and beside it the CT on the method's surface over the case (red: on the
      wrong sheet; green: the known cells; yellow outline: the area; cyan: the cross-section, left to right; a view cut
      down a column is turned a quarter so that the cut runs left to right); a bar above shows the known cells (green)
      and the area (yellow) along the cross-section
  python figures.py stack  A.jpg B.jpg ... --out F.jpg      examples one under another (--cols 2: two per row)
  python figures.py chart  --analysis W.json --out F.png     score by kind of area; failed cases
  python figures.py reach  --scores S.jsonl ... --out F.png  score by distance from the nearest known cell
  python figures.py pairs  --analysis W.json --out F.png     per case: GrowPatch against GrowPatch → SurfNet and
                                                               against the recto local solver's frame

W.json is `ways.py analyse`'s output. Each method's surface is its saved result, read as bench.py's scorer reads it
(--results: the stages' results trees); its score, and the right or wrong sheet per cell, from its saved error map
(bench.py score; surfnet.eval.score: the wrong sheet is half a winding away or more, or not covered). The CT is read with
SurfNet's own reader (s3:// with s5cmd, cached)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surfnet.eval.score import credit  # noqa: E402
from surfnet.volume import GPUChunkCache, open_volume  # noqa: E402

VOLUME = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
# the ways the README compares, in one colour each everywhere (validated categorical slots 1-5, in order)
NAME = {"GF4CmedK": "GrowPatch → SurfNet", "G": "GrowPatch", "S5": "SurfNet alone", "RF": "recto local solver",
        "RFS": "recto local solver + fibre step",
        "S5base": "SurfNet alone, base model", "S5ft": "SurfNet alone, fine-tuned"}   # the fine-tune test's two models
PAL = {"GF4CmedK": "#2a78d6", "G": "#eb6834", "S5": "#1baf7a", "RF": "#eda100", "RFS": "#e87ba4",
       "S5base": "#1baf7a", "S5ft": "#6250d6"}                                       # (slot 7: the fine-tuned model)
COL = {m: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for m, c in PAL.items()}
GREEN, RED, YELLOW, CYAN = (40, 200, 70), (230, 30, 60), (245, 200, 0), (0, 200, 230)
KNOWN = tuple(round(0.55 * g + 0.45 * 255) for g in GREEN)    # the known cells' line in the cross-section: light green
BG, INK, SUB = (252, 252, 251), (20, 20, 20), (90, 90, 90)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT.replace("Sans", "Sans-Bold") if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


class CT:
    """Trilinear CT at level-0 voxel coordinates (zyx), through SurfNet's chunk cache on the CPU."""

    def __init__(self, url, cache, capacity=700):
        self.src = open_volume(url, cache)
        self.c = GPUChunkCache(self.src, 0, "cpu", capacity)

    def __call__(self, zyx: np.ndarray, batch=150_000) -> np.ndarray:
        q = torch.as_tensor(np.ascontiguousarray(zyx.reshape(-1, 3)), dtype=torch.float32)
        out = torch.zeros(len(q))
        for i in range(0, len(q), batch):
            b = q[i:i + batch]
            keys = self.c.keys_for(b, margin=2.0)
            self.src.fetch(0, keys)
            self.c.ensure(keys)
            out[i:i + batch] = self.c.sample(b)
        return out.reshape(zyx.shape[:-1]).numpy()


def grid_normals(P, ref):
    n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[np.sum(n * ref, -1) < 0] *= -1
    return n


def upsample(a, up):
    return ndimage.zoom(a, (up, up) + (1,) * (a.ndim - 2), order=1)


def grey(img, lo, hi):
    g = (np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, -1)
    rgb[~np.isfinite(img)] = 0
    return rgb


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask)


def label(width, parts, size, pad=6):
    """One line of text segments (string, colour, bold), as wide as `width`."""
    im = Image.new("RGB", (width, size + 2 * pad + 4), BG)
    d = ImageDraw.Draw(im)
    x = 6
    for s, c, b in parts:
        d.text((x, pad), s, fill=c, font=font(size, b))
        x += int(font(size, b).getlength(s))
    return im


def hstack(parts, gap=12):
    H = max(p.height for p in parts)
    out = Image.new("RGB", (sum(p.width for p in parts) + gap * (len(parts) - 1), H), BG)
    x = 0
    for p in parts:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def vstack(parts, gap=6):
    W = max(p.width for p in parts)
    out = Image.new("RGB", (W, sum(p.height for p in parts) + gap * (len(parts) - 1)), BG)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + gap
    return out


def save(im: Image.Image, out: str):
    im.save(out, quality=90) if out.lower().endswith((".jpg", ".jpeg")) else im.save(out)


def example(a):
    cdir = Path(a.cases) / a.case
    info = json.loads((cdir / "case.json").read_text())
    z = {k: v.astype(np.float64) if v.dtype.kind == "f" else v for k, v in np.load(cdir / "arrays.npz").items()}
    P, N, K, S, V = z["P"], z["N"], z["K"], z["S"], z["V"]
    sp = info["spacing_vox"]
    methods = a.methods.split(",")
    err = {m: np.load(Path(a.errors) / a.case / f"err_{m}.npy").astype(np.float64) for m in methods}
    ok = {m: np.isfinite(e) & (np.abs(np.nan_to_num(e, nan=1e9)) < sp / 2) for m, e in err.items()}   # right sheet
    score = {m: 100 * credit(e[S], np.isfinite(e[S]), sp).mean() for m, e in err.items()}
    off = {m: 100 * (1 - ok[m][S].mean()) for m in methods}      # % of the area on the wrong sheet
    KS = K | S
    from bench import find_result
    surf = {}
    for m in methods:                              # the method's own surface, as the scorer read it
        q = find_result([Path(r) for r in a.results], a.case, m)
        if q is None:
            raise SystemExit(f"{m}: no result for {a.case} in {a.results}")
        surf[m] = (q[0], q[1] & KS)
    surf["truth"] = (P, V & KS)
    ct = CT(a.volume, a.cache)
    # the cross-section: a.span cells along a grid line through the densest area where the focus method (default: the
    # first) is on the wrong sheet and the contrast method (default: the last other one) on the right one; across the
    # known strip for an area known on one side, else along the grid direction that crosses more of the focus's wrong
    # cells
    foc = a.focus or methods[0]
    bad0 = S & ~ok[foc]
    pick = bad0 & ok[a.contrast or [m for m in methods if m != foc][-1]]
    d2 = ndimage.uniform_filter((pick if pick.any() else bad0).astype(float), 21)
    r0, c0 = np.unravel_index(np.argmax(np.where(S, d2, -1)), d2.shape)
    if pick.any() and not pick[r0, c0]:            # through such a cell: the nearest one
        _, (ri, ci) = ndimage.distance_transform_edt(~pick, return_indices=True)
        r0, c0 = ri[r0, c0], ci[r0, c0]
    side = info["case"].get("layout") == "side"
    if a.along != "auto":
        along_rows = a.along == "rows"
    elif side:
        kr, kc = np.nonzero(K)
        sr, sc = np.nonzero(S)
        along_rows = abs(kc.mean() - sc.mean()) >= abs(kr.mean() - sr.mean())
    else:
        along_rows = bad0[r0].sum() >= bad0[:, c0].sum()
    line = [(r0, c) for c in range(P.shape[1])] if along_rows else [(r, c0) for r in range(P.shape[0])]
    idx = [i for i, rc in enumerate(line) if (KS if side else V)[rc]]    # one side known: the strip and the area only
    wrong = [i for i in idx if bad0[line[i]]] or [i for i in idx if S[line[i]]]
    mid = (min(wrong) + max(wrong)) // 2
    if side:                                                  # from the strip's inner part across the area
        si = [i for i in idx if S[line[i]]]
        ki = [i for i in idx if K[line[i]]]
        start = max(ki) - 16 if max(ki) < min(si) else min(ki) + 16 - a.span
        cells = [line[i] for i in idx if start <= i < start + a.span]
    else:
        cells = [line[i] for i in idx if abs(i - mid) <= a.span // 2]
    rs, cs = np.nonzero(S)                         # the view: the area and a.pad cells of its surroundings
    box = (slice(max(0, rs.min() - a.pad), rs.max() + a.pad + 1), slice(max(0, cs.min() - a.pad), cs.max() + a.pad + 1))
    u = P[cells[-1]] - P[cells[0]]
    u /= np.linalg.norm(u)
    o = P[r0, c0]
    w = N[r0, c0] - np.dot(N[r0, c0], u) * u
    w /= np.linalg.norm(w)
    tp = np.array([[np.dot(P[c] - o, u), np.dot(P[c] - o, w)] for c in cells])
    umin, umax = tp[:, 0].min() - 10, tp[:, 0].max() + 10
    wmin, wmax = tp[:, 1].min() - a.windings * sp, tp[:, 1].max() + a.windings * sp
    us = np.arange(umin, umax, 1.0)
    s = a.width / len(us)                          # px per voxel: every example's cross-section is a.width wide
    grow = max(0.0, a.min_height / s - (wmax - wmin)) / 2   # and at least a.min_height tall: more windings shown
    wmin, wmax = wmin - grow, wmax + grow
    ws = np.arange(wmax, wmin, -1.0)
    plane = ct((o + us[None, :, None] * u + ws[:, None, None] * w)[..., ::-1])
    lo, hi = np.percentile(plane[plane > 0], [0.5, 99.5]) if (plane > 0).any() else (0, 255)
    Ws, Hs = round(len(us) * s), round(len(ws) * s)
    base = Image.fromarray(grey(plane, lo, hi)).resize((Ws, Hs), Image.BICUBIC)

    def line_of(m, only=None):
        """the method's own cells along the cut (only: those of this mask too), densified (linear, 5 points per cell)
        and projected on the plane: a list of runs (consecutive valid cells) of (points [n, 2], right [n])."""
        Pm, Vm = surf[m]
        Vm = Vm if only is None else Vm & only
        okm = ok.get(m, np.ones_like(S))
        runs, cur = [], []
        for i, rc in enumerate(cells):
            if Vm[rc]:
                cur.append(i)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        out = []
        for run in runs:
            pts, rt = [], []
            for j, i in enumerate(run):
                p0 = Pm[cells[i]]
                p1 = Pm[cells[run[j + 1]]] if j + 1 < len(run) else p0
                for t in (np.linspace(0, 1, 5, endpoint=False) if j + 1 < len(run) else [0.0]):
                    q = p0 + t * (p1 - p0) - o
                    pts.append((q @ u, q @ w))
                    rt.append(bool(okm[cells[i if t < 0.5 else run[j + 1]]]) or not S[cells[i if t < 0.5 else run[j + 1]]])
            out.append((np.array(pts), np.array(rt)))
        return out

    def polyline(d, runs, col, dark, width):
        for pts, rt in runs:
            xy = [((a0 - umin) * s, (wmax - b0) * s) for a0, b0 in pts]
            for k in range(len(xy) - 1):
                d.line([xy[k], xy[k + 1]], fill=col if rt[k] and rt[k + 1] else dark, width=width)
            if len(xy) == 1:
                d.ellipse([xy[0][0] - width / 2, xy[0][1] - width / 2, xy[0][0] + width / 2, xy[0][1] + width / 2],
                          fill=col if rt[0] else dark)

    edge = np.zeros_like(K)                        # the known cells next to the others along the cut
    for c0_, c1_ in zip(cells, cells[1:]):
        if K[c0_] != K[c1_]:
            edge[c0_ if K[c0_] else c1_] = True
    rows_, up, ref_img = [], 3, None
    for m in methods:
        Pm, Vm = surf[m]
        Pb, Vb = Pm[box], Vm[box]
        Nb = grid_normals(Pb, N[box])
        Pu, Nu = upsample(Pb, up), upsample(Nb, up)
        Nu /= np.maximum(np.linalg.norm(Nu, axis=-1, keepdims=True), 1e-9)
        Vu = np.kron(Vb, np.ones((up, up), bool))[:Pu.shape[0], :Pu.shape[1]]
        img = np.mean([ct((Pu + d0 * Nu)[..., ::-1]) for d0 in (-1.0, 0.0, 1.0)], 0)
        img = np.where(Vu, img, np.nan)
        if ref_img is None:
            ref_img = np.nanpercentile(img, [0.5, 99.5])
        rgb = grey(img, *ref_img).astype(np.float32)
        k_up = np.kron(K[box], np.ones((up, up), bool))
        bad = np.kron((S & ~ok[m])[box], np.ones((up, up), bool))
        rgb[k_up] = 0.72 * rgb[k_up] + 0.28 * np.array(GREEN)
        rgb[bad] = 0.5 * rgb[bad] + 0.5 * np.array(RED)
        rgb = rgb.astype(np.uint8)
        rgb[np.kron(outline(S[box]), np.ones((up, up), bool))] = YELLOW
        fl = Image.fromarray(rgb)
        dfl = ImageDraw.Draw(fl)                  # the cross-section's cells, from its left end to its right end
        ends = [((r - box[0].start + 0.5) * up, (c - box[1].start + 0.5) * up) for r, c in (cells[0], cells[-1])]
        dfl.line([(x, y) for y, x in ends], fill=CYAN, width=3)
        if not along_rows:                        # a cut down a column: turned a quarter (not mirrored), so that
            fl = fl.rotate(90, expand=True)       # the cut runs left to right, as in the cross-section
        fl = fl.resize((round(fl.width * Hs / fl.height), Hs), Image.BICUBIC)
        lw = max(4, round(4.4 * s))               # the method's line and the known cells'; the truth thinner
        im = base.copy()
        d = ImageDraw.Draw(im)
        polyline(d, line_of("truth"), (255, 255, 255), (255, 255, 255), max(3, round(0.6 * lw)))
        colm = COL.get(m, INK)                    # the method's line from where it meets the known cells; theirs over it
        polyline(d, line_of(m, only=~K | edge), colm, tuple(int(0.5 * c_) for c_ in colm), lw)
        polyline(d, line_of("truth", only=K), KNOWN, KNOWN, lw)
        here = 100 * np.mean([ok[m][c] for c in cells if S[c]])
        rows_ += [label(Ws + 12 + fl.width, [(NAME.get(m, m), COL.get(m, INK), True),
                                             (f"   score {score[m]:.1f}", INK, False),
                                             (f"   {off[m]:.1f} % on the wrong sheet", SUB, False)], 26),
                  hstack([im, fl])]
        print(f"{m}: score {score[m]:.1f}, wrong sheet {off[m]:.1f} % (this line: right {here:.0f} %)", flush=True)
    bar = Image.new("RGB", (Ws, 10), BG)          # the known cells (green) and the area (yellow) along the line
    db = ImageDraw.Draw(bar)
    for rc, (a0, _) in zip(cells, tp):
        col = GREEN if K[rc] else YELLOW if S[rc] else None
        if col:
            x = (a0 - umin) * s
            db.rectangle([x - 2.4 * s, 1, x + 2.4 * s, 9], fill=col)
    head = label(rows_[0].width, [(a.title, INK, True)] if a.title else [], 30)
    k = key(30)
    head.paste(k, (head.width - k.width - 8, (head.height - k.height) // 2))
    save(vstack([head, bar] + rows_, gap=4), a.out)
    print("example", a.out, flush=True)


def key(h, up=4):
    """the cross-section's key: a white dot for the ground truth, a green one for the known (approved) cells."""
    f = font(round(0.6 * h))
    parts = []
    for lab, col in (("ground truth", (255, 255, 255)), ("approved cells", KNOWN)):
        r = round(0.26 * h) * up                  # drawn large and shrunk: a smooth edge
        dot = Image.new("RGB", (2 * r + 2 * up, h * up), BG)
        y = h * up // 2
        ImageDraw.Draw(dot).ellipse([up, y - r, up + 2 * r, y + r], fill=col,
                                    outline=tuple(int(0.55 * c_) for c_ in col), width=up)
        dot = dot.resize((dot.width // up, h), Image.LANCZOS)
        txt = Image.new("RGB", (int(f.getlength(lab)) + 22, h), BG)
        ImageDraw.Draw(txt).text((7, (h - f.size) // 2 - 1), lab, fill=INK, font=f)
        parts += [dot, txt]
    return hstack(parts, gap=0)


def hstack(parts, gap=6):
    out = Image.new("RGB", (sum(p.width for p in parts) + gap * (len(parts) - 1), max(p.height for p in parts)), BG)
    x = 0
    for p in parts:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def stack(a):
    ims = [Image.open(p).convert("RGB") for p in a.images]
    rows = [hstack(ims[i:i + a.cols], gap=a.gap) for i in range(0, len(ims), a.cols)]
    save(vstack(rows, gap=a.gap), a.out)
    print("stack", a.out)


# ------------------------------------------------------------------------------------------------ charts
INKC = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
FAMS = [("holes up to 3.8 mm", "holes\n1.9-3.8 mm"), ("holes 7.6-11.9 mm", "holes\n7.6-11.9 mm"),
        ("7.6 mm area, known on two sides / patches", "7.6 mm area,\ntwo sides / patches"),
        ("extensions 1.9-15.2 mm", "extensions\n1.9-15.2 mm"), ("7.6 mm area, known on one side", "7.6 mm area,\none side"),
        ("all", "all")]
WAYS = ["G", "GF4CmedK", "S5", "RF"]                       # the README's (RFS: the supplement's tables)


def _style(ax):
    ax.set_facecolor(INKC["surface"])
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(INKC["axis"])
    ax.tick_params(which="both", colors=INKC["secondary"], labelsize=9)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def chart(a):
    """The score by kind of area (the mean over the cases; higher is better), and each way's failed cases: those with
    more than 10 % / 20 % of the area on the wrong sheet (fewer is better)."""
    from ways import family
    plt = _plt()
    d = json.load(open(a.analysis))
    ways = a.ways.split(",")
    by = {f: [n for n, r in d.items() if f == "all" or family(r["case"]["case"]) == f] for f, _ in FAMS}
    mean = lambda f, m: float(np.mean([d[n]["methods"][m]["score"] for n in by[f]]))         # noqa: E731
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [3.1, 1.35]},
                                 facecolor=INKC["surface"])
    _style(ax)
    n, w = len(ways), 0.84 / len(ways)
    for j, m in enumerate(ways):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(FAMS))]
        ys = [mean(f, m) for f, _ in FAMS]
        ax.bar(xs, ys, w * 0.9, color=PAL[m], label=NAME[m])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.2, f"{y:.0f}", ha="center", va="bottom", fontsize=7.5, color=INKC["secondary"])
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([f"{lab}\n({len(by[f])} case{'s' if len(by[f]) != 1 else ''})" for f, lab in FAMS], fontsize=9,
                       color=INKC["primary"])
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color=INKC["grid"], lw=0.6)
    ax.set_ylabel("score (0-100)", fontsize=9, color=INKC["secondary"])
    ax.set_title("Accuracy: the score, mean by kind of area (higher is better)",
                 fontsize=10.5, loc="left", color=INKC["primary"])
    ax.legend(frameon=False, fontsize=9, ncol=min(n, 5), loc="upper center", bbox_to_anchor=(0.62, -0.2),
              labelcolor=INKC["primary"])
    _style(bx)
    ths, k = (10, 20), len(ways)
    vmax = 0
    for j, m in enumerate(ways):
        ys = [i * (k + 1) + j for i in range(len(ths))]
        vs = [100.0 * sum(r["methods"][m]["wrong"] > t for r in d.values()) / len(d) for t in ths]
        bx.barh(ys, vs, 0.85, color=PAL[m]); vmax = max(vmax, *vs)
        for y, v in zip(ys, vs):
            bx.text(v + 1.0, y, f"{v:.0f} %", va="center", fontsize=8.5, color=INKC["primary"])
    bx.set_yticks([i * (k + 1) + (k - 1) / 2 for i in range(len(ths))])
    bx.set_yticklabels([f"more than\n{t} % of the\narea wrong" for t in ths], fontsize=9, color=INKC["primary"])
    bx.invert_yaxis()
    bx.set_xlim(0, max(85.0, vmax * 1.15 + 2))                          # (room for the largest bar and its label)
    bx.xaxis.grid(True, color=INKC["grid"], lw=0.6)
    bx.set_xlabel(f"share of the {len(d)} cases (%)", fontsize=9, color=INKC["secondary"])
    bx.set_title("Failure rate (lower is better):\nthe cases with more than 10 % or 20 %\nof the area on the wrong sheet",
                 fontsize=10.5, loc="left", color=INKC["primary"])
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=INKC["surface"])
    print("chart", a.out)


def reach(a):
    """The score by distance from the nearest known cell, with the known cells around the area (holes, two sides,
    patches) and on one side only (extensions, one-sided areas): the mean over the cases with cells at that
    distance."""
    from ways import read_scores
    plt = _plt()
    bins = [("0-10", "0-0.5"), ("10-20", "0.5-0.9"), ("20-40", "0.9-1.9"), ("40-80", "1.9-3.8"), ("80-160", "3.8-7.6"),
            ("160-inf", "7.6-15")]
    ways = a.ways.split(",")
    rows = [r for r in read_scores(a.scores).values() if all(m in r["methods"] for m in ways)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=INKC["surface"], sharey=True)
    for ax, side, title in ((axes[0], False, "approved cells around the area"), (axes[1], True, "approved cells on one side")):
        _style(ax)
        sel = [r for r in rows if (r["case"]["case"]["layout"] == "side") == side]
        used = [b for b in bins if any(b[0] in r["methods"]["G"].get("by_distance", {}) for r in sel)]
        for m in ways:
            ys = [100 * np.mean([r["methods"][m]["by_distance"][b][0] for r in sel
                                 if b in r["methods"][m].get("by_distance", {})]) for b, _ in used]
            ax.plot(range(len(used)), ys, marker="o", lw=2, ms=6, color=PAL[m], label=NAME[m])
        ax.set_xticks(range(len(used)))
        ax.set_xticklabels([lab_ for _, lab_ in used], fontsize=9)
        ax.set_xlabel("distance from the nearest approved cell (mm)", fontsize=9, color=INKC["secondary"])
        ax.set_ylim(0, 102)
        ax.yaxis.grid(True, color=INKC["grid"], lw=0.6)
        ax.set_title(f"{title} ({len(sel)} cases)", fontsize=10, loc="left", color=INKC["primary"])
    axes[0].set_ylabel("score (0-100)", fontsize=9, color=INKC["secondary"])
    axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INKC["primary"])
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=INKC["surface"])
    print("reach", a.out)


def pairs(a):
    """Every case: GrowPatch's score (x) against another way's (y), in one row: two panels per way (known cells
    around the area / on one side); above the diagonal the way did better than GrowPatch."""
    plt = _plt()
    d = json.load(open(a.analysis))
    ways = a.ways.split(",")
    fig, axes = plt.subplots(1, 2 * len(ways), figsize=(3.6 * 2 * len(ways), 4.0), facecolor=INKC["surface"],
                             squeeze=False)
    lo = 5 * np.floor(min(min(r["methods"][m]["score"] for m in ["G"] + ways) for r in d.values()) / 5) - 2
    for i, m in enumerate(ways):
        row = axes[0][2 * i:2 * i + 2]
        for ax, side, title in ((row[0], False, "approved cells around the area"), (row[1], True, "approved cells on one side")):
            _style(ax)
            sel = [r for r in d.values() if (r["case"]["case"]["layout"] == "side") == side]
            x = np.array([r["methods"]["G"]["score"] for r in sel])
            y = np.array([r["methods"][m]["score"] for r in sel])
            ax.plot([lo, 100], [lo, 100], color=INKC["axis"], lw=1, zorder=1)
            ax.scatter(x, y, s=34, color=PAL[m], edgecolor=INKC["surface"], linewidth=1, zorder=2)
            ax.set_xlim(lo, 101)
            ax.set_ylim(lo, 101)
            ax.set_aspect("equal")
            ax.grid(True, color=INKC["grid"], lw=0.6)
            ax.set_xlabel(f"{NAME['G']}: score", fontsize=9, color=INKC["secondary"])
            ax.set_ylabel(f"{NAME[m]}: score", fontsize=9, color=INKC["secondary"])
            ax.set_title(f"{title} ({len(sel)} cases)", fontsize=10, loc="left", color=INKC["primary"])
    fig.tight_layout()
    fig.savefig(a.out, dpi=170, facecolor=INKC["surface"])
    print("pairs", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("example")
    p.add_argument("--cases", required=True)
    p.add_argument("--results", nargs="+", required=True, help="results trees with the methods' surfaces (the first "
                   "that has a method's)")
    p.add_argument("--errors", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--focus", default=None, help="centre the cross-section on this method's wrong cells (default: the "
                   "first method's)")
    p.add_argument("--contrast", default=None, help="and where this method is right (default: the last method other than "
                   "the focus)")
    p.add_argument("--span", type=int, default=160, help="the cross-section's length in cells at most")
    p.add_argument("--along", default="auto", choices=("auto", "rows", "cols"))
    p.add_argument("--width", type=int, default=1300, help="the cross-section's width in pixels")
    p.add_argument("--min-height", type=int, default=300, help="the cross-section's height in pixels at least")
    p.add_argument("--pad", type=int, default=16, help="cells around the area in the view of the surface")
    p.add_argument("--windings", type=float, default=3.0, help="windings shown above and below the truth")
    p.add_argument("--volume", default=VOLUME)
    p.add_argument("--cache", default="ct_cache")
    p.add_argument("--out", required=True)
    p = sub.add_parser("stack")
    p.add_argument("images", nargs="+")
    p.add_argument("--cols", type=int, default=1, help="images per row")
    p.add_argument("--gap", type=int, default=36)
    p.add_argument("--out", required=True)
    for name, ways in (("chart", ",".join(WAYS)), ("pairs", "GF4CmedK,RF")):
        p = sub.add_parser(name)
        p.add_argument("--analysis", required=True, help="ways.py analyse's json")
        p.add_argument("--ways", default=ways, help="the ways shown, in this order")
        p.add_argument("--out", required=True)
    p = sub.add_parser("reach")
    p.add_argument("--scores", nargs="+", required=True, help="score files (later files add to earlier ones)")
    p.add_argument("--ways", default=",".join(WAYS), help="the ways shown, in this order")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"example": example, "stack": stack, "chart": chart, "reach": reach, "pairs": pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
