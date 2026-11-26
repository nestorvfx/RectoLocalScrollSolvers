"""SurfNet inference on a whole sheet (any scroll, from its CT volume).

Inputs: the sheet grid (tifxyz) as the HINT surface (optional), the verified (green) cells with their exact
geometry, optional REJECTED cells ("the hint is wrong here").
1. start estimate E0: green exact; elsewhere the hint (hint on) or the membrane extension of green (hint off);
2. frame F = low-pass of E in xyz (sigma 6 cells, normalised over the valid cells), its normals N_F;
3. LOCATE: context stacks along F (+-96 vox at 2 vox, level-1 volume, 1 px per cell) in 160-cell windows (stride
   112, Gaussian blending), dihedral TTA over `tta` of the 8 flips / rotations (the axis channels transformed with the
   grid; the refinement passes use the same TTA); the target-winding
   distribution over +-80 vox; a global BP over the sheet (1-vox labels; smoothness on the true surface; green pinned;
   a penalty where a neighbour winding is predicted: non-crossing) -> E1 = F + t N_F;
4. REFINE (up to `passes` times): F = low-pass(E); both streams in windows (fine core 80 cells inside the 160-cell
   context; cores tile the sheet), fine distribution over +-10 vox (0.5 bins) + outside; BP with green pinned; cells
   the fine head calls outside keep the locate result; stop when the frame moves < 0.25 vox at p99;
5. maps: the final offset vs the hint, the locate / refine entropy, the BP margin, the outside probability.
The prompts follow the training exactly (sampler._prompts): a Gaussian in depth, a clipped ramp and a mask per
surface (green / hint / rejected) at its offset along the frame normal."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from ..common.geometry import bilinear, valid_at
from . import bp as BP
from .surface import height_deltas, normals
from .surface import gauss_weight as _gauss_weight
from ..common.geometry import axis_direction
from ..volume import GPUChunkCache, to_coarser
from ..common import frames as GM
from ..model import NetSpec, TwoStream, bins_ctx, readout


TIMES: dict = defaultdict(float)          # seconds per part of the work (ct_fetch, ct_load, net, bp), summed; reset by
                                          # the caller (correct.joint_solve reports them per call)


@contextmanager
def timed(name: str, dev=None):
    """Adds the block's wall time to TIMES[name] (the GPU synchronised before and after, so the parts add up)."""
    sync = dev is not None and str(dev).startswith("cuda") and torch.cuda.is_available()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            torch.cuda.synchronize()
        TIMES[name] += time.perf_counter() - t0


def amp_dtype(name: str | None = None):
    """The autocast precision of inference: fp16 (default; env SURFNET_AMP), the precision of every benchmark run; bf16
    (as trained; Ampere or newer); or fp32 (no autocast)."""
    name = name or os.environ.get("SURFNET_AMP", "fp16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]


def load_model(path, device="cuda", use_ema=True):
    st = torch.load(path, map_location=device, weights_only=False)
    spec = NetSpec.from_dict(st["spec"])
    m = TwoStream(spec).to(device)
    m.load_state_dict(st["ema"] if use_ema and "ema" in st else st["model"])
    m.eval()
    return m, spec


class Volume:
    """CT access for both streams: level 0 (fine) and level 1 (context) chunk caches of one volume source."""

    def __init__(self, source, device="cuda", cap_fine=2500, cap_ctx=800):
        self.src = source
        self.fine = GPUChunkCache(source, 0, device, cap_fine)
        self.ctx = GPUChunkCache(source, 1, device, cap_ctx)

    def sample(self, cache, zyx, ok):
        q = zyx[:, ok]
        if q.numel():
            keys = cache.keys_for(q[::4] if q.shape[0] > 8 else q, margin=9.0)
            with timed("ct_fetch"):
                self.src.fetch(cache.level, keys)
            with timed("ct_load"):
                cache.ensure(keys)
        v = cache.sample(zyx)
        return torch.where(ok[None], v, torch.zeros_like(v))

    def prefetch(self, P: torch.Tensor, N: torch.Tensor, valid: torch.Tensor, reach0: float, reach1: float,
                 every: int = 4, step: float = 16.0) -> int:
        """Download, in one batch per level, the chunks within `reach0` (level 0) / `reach1` (level 1; both in level-0
        voxels) of a surface along its normals: P, N [H, W, 3] xyz, valid [H, W]. The windows then find their chunks
        on disk instead of fetching a few at a time. -> the number of chunk keys."""
        m = valid[::every, ::every]
        if not bool(m.any()):
            return 0
        p, n = P[::every, ::every][m], N[::every, ::every][m]                       # [M, 3] xyz
        n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp(min=1e-9)
        total = 0
        for cache, reach, coarse in ((self.fine, reach0, False), (self.ctx, reach1, True)):
            t = torch.arange(-reach, reach + step, step, device=p.device, dtype=p.dtype)
            zyx = (p[None] + t.view(-1, 1, 1) * n[None]).flip(-1)
            if coarse:
                zyx = to_coarser(zyx)
            keys = cache.keys_for(zyx, margin=every * 5.0 + step)
            with timed("ct_fetch"):
                self.src.fetch(cache.level, keys)
            total += len(keys)
        return total


def _prompt(off_cells, m_cells, kk, step, psig, khalf):
    """off [h, w] (vox along the frame normal, NaN = absent), m [h, w] -> 3 channels [3, D, h, w]."""
    k = off_cells / step
    kd = kk.view(-1, 1, 1)
    mf = (m_cells & torch.isfinite(off_cells)).float()[None]
    k = torch.nan_to_num(k, nan=0.0)[None]
    gauss = torch.exp(-((kd - k) ** 2) / (2 * psig ** 2)) * mf
    ramp = ((kd - k) / khalf).clamp(-1, 1) * mf
    return torch.cat([gauss[None], ramp[None], mf.expand_as(gauss)[None]], 0)


def _dihedral(x, t):
    """x [..., H, W]; t in 0..7: bit 0 flip H, bit 1 flip W, bit 2 transpose."""
    if t & 4:
        x = x.transpose(-1, -2)
    if t & 1:
        x = x.flip(-2)
    if t & 2:
        x = x.flip(-1)
    return x


def _dihedral_inv(x, t):
    if t & 2:
        x = x.flip(-1)
    if t & 1:
        x = x.flip(-2)
    if t & 4:
        x = x.transpose(-1, -2)
    return x


def _tta_input(x, t):
    """x [1, 13, D, H, W]: the grid transform of the axis channels (11 = AXIS_Y, 12 = AXIS_X)."""
    y = _dihedral(x, t).clone()
    if t & 4:
        y[:, [11, 12]] = y[:, [12, 11]]
    if t & 1:
        y[:, 11] = -y[:, 11]
    if t & 2:
        y[:, 12] = -y[:, 12]
    return y


class Inference:
    def __init__(self, model: TwoStream, spec: NetSpec, vol: Volume, axis=None, device="cuda", tta=8, amp=None):
        self.m, self.sp, self.vol, self.axis, self.dev, self.tta = model, spec, vol, axis, device, tta
        self.amp = amp_dtype(amp)
        self.C, self.core = 160, 80

    # ------------------------------------------------------------------ stacks along the frame
    def _ctx_stack(self, Fp, Fn, V, r0, c0, n):
        dev = self.dev
        a = torch.arange(n, device=dev, dtype=torch.float64)
        R, Cc = torch.meshgrid(r0 + a, c0 + a, indexing="ij")
        ok = valid_at(V, R, Cc)
        pos, nrm = bilinear(Fp, R, Cc), bilinear(Fn, R, Cc)
        nrm = nrm / torch.linalg.vector_norm(nrm, dim=-1, keepdim=True).clamp(min=1e-9)
        k = torch.arange(-self.sp.ck, self.sp.ck + 1, device=dev, dtype=torch.float64) * 2.0
        pts = pos[None] + k.view(-1, 1, 1, 1) * nrm[None]
        ct = self.vol.sample(self.vol.ctx, to_coarser(pts.flip(-1)), ok)
        return ct.float(), ok

    def _fine_stack(self, Fp, Fn, V, r0, c0, n, p=3):
        dev = self.dev
        a = torch.arange(p * n, device=dev, dtype=torch.float64)
        R, Cc = torch.meshgrid(r0 + (a - (p - 1) / 2) / p, c0 + (a - (p - 1) / 2) / p, indexing="ij")
        ok = valid_at(V, R, Cc)
        pos, nrm = bilinear(Fp, R, Cc), bilinear(Fn, R, Cc)
        nrm = nrm / torch.linalg.vector_norm(nrm, dim=-1, keepdim=True).clamp(min=1e-9)
        k = torch.arange(-self.sp.fk, self.sp.fk + 1, device=dev, dtype=torch.float64)
        pts = pos[None] + k.view(-1, 1, 1, 1) * nrm[None]
        ct = self.vol.sample(self.vol.fine, pts.flip(-1), ok)
        return ct.float(), ok

    def _inputs(self, ct, ok, offs, masks, ax, kk, step, psig, khalf, pix=1):
        from ..train import intensity as I
        D = kk.numel()
        ctn = I.normalise_eval(ct[None, None], ok[None, None, None])
        rep = (lambda t: t.repeat_interleave(pix, -2).repeat_interleave(pix, -1)) if pix > 1 else (lambda t: t)
        chans = [ctn]
        for o, m in zip(offs, masks):
            chans.append(_prompt(rep(o), rep(m), kk, step, psig, khalf)[None])
        chans = [chans[0]] + [c for c in chans[1:]]
        pr = torch.cat([c[0] for c in chans[1:]], 0)[None]                         # [1, 9, D, h, w]
        axp = rep(ax) * ok[None]
        return torch.cat([ctn, pr, ok[None, None, None].float().expand(1, 1, D, -1, -1),
                          axp[None, :, None].expand(1, 2, D, -1, -1)], 1).float()

    # ------------------------------------------------------------------ passes
    @torch.no_grad()
    def locate(self, Fp, Fn, V, surf_offs, surf_masks, ax):
        """-> per-cell probabilities over the context bins: target [nc, H, W], neighbours [2, nc, H, W]."""
        dev, sp = self.dev, self.sp
        H, W = V.shape
        nc = sp.nc
        C, stride = self.C, 112
        prob = torch.zeros((3, nc, H, W), device=dev)
        wsum = torch.zeros((H, W), device=dev)
        gw = _gauss_weight(C, C, dev)
        kk = torch.arange(-sp.ck, sp.ck + 1, device=dev, dtype=torch.float32)
        for r0 in range(-24, max(H - 24, 1), stride):
            for c0 in range(-24, max(W - 24, 1), stride):
                ra, rb, ca, cb = max(r0, 0), min(r0 + C, H), max(c0, 0), min(c0 + C, W)
                if ra >= rb or ca >= cb or not bool(V[ra:rb, ca:cb].any()):
                    continue
                ct, ok = self._ctx_stack(Fp, Fn, V, r0, c0, C)
                offs = [_crop(o, r0, c0, C, float("nan")) for o in surf_offs]
                msk = [_crop(m, r0, c0, C, False) for m in surf_masks]
                axw = _crop(ax, r0, c0, C, 0.0)
                x = self._inputs(ct, ok, offs, msk, axw, kk, 2.0, 0.75, sp.ck)
                pw = torch.zeros((3, nc, C, C), device=dev)
                for t in range(self.tta):
                    with timed("net", dev), torch.autocast("cuda", dtype=self.amp or torch.float32,
                                                           enabled=self.amp is not None):
                        lc = self.m(_tta_input(x, t))["ctx"][0]                         # [3, nc, C/2, C/2]
                    lc = F.interpolate(lc.float(), size=(C, C), mode="bilinear", align_corners=False)
                    pw += _dihedral_inv(torch.softmax(lc, 1), t) / self.tta
                g = gw[ra - r0:rb - r0, ca - c0:cb - c0]
                prob[:, :, ra:rb, ca:cb] += g * pw[:, :, ra - r0:rb - r0, ca - c0:cb - c0]
                wsum[ra:rb, ca:cb] += g
        prob = prob / wsum.clamp(min=1e-6)
        return prob[0], prob[1:], wsum > 0

    @torch.no_grad()
    def refine(self, Fp, Fn, V, surf_offs, surf_masks, ax):
        """-> fine probabilities [nf + 1, H, W] (last = outside) and the context target probabilities [nc, H, W]."""
        dev, sp = self.dev, self.sp
        H, W = V.shape
        C, core = self.C, self.core
        c0f = (C - core) // 2
        stride = 64
        nf, nc = sp.nf + 1, sp.nc
        pf = torch.zeros((nf, H, W), device=dev)
        pc = torch.zeros((nc, H, W), device=dev)
        wsum = torch.zeros((H, W), device=dev)
        gw = _gauss_weight(core, core, dev)
        kc = torch.arange(-sp.ck, sp.ck + 1, device=dev, dtype=torch.float32)
        kf = torch.arange(-sp.fk, sp.fk + 1, device=dev, dtype=torch.float32)
        for r in range(-8, max(H - 8, 1), stride):
            for c in range(-8, max(W - 8, 1), stride):
                ra, rb, ca, cb = max(r, 0), min(r + core, H), max(c, 0), min(c + core, W)
                if ra >= rb or ca >= cb or not bool(V[ra:rb, ca:cb].any()):
                    continue
                r0, cc0 = r - c0f, c - c0f
                ctc, okc = self._ctx_stack(Fp, Fn, V, r0, cc0, C)
                ctf, okf = self._fine_stack(Fp, Fn, V, r, c, core)
                xc = self._inputs(ctc, okc, [_crop(o, r0, cc0, C, float("nan")) for o in surf_offs],
                                  [_crop(m, r0, cc0, C, False) for m in surf_masks], _crop(ax, r0, cc0, C, 0.0),
                                  kc, 2.0, 0.75, sp.ck)
                xf = self._inputs(ctf, okf, [_crop(o, r, c, core, float("nan")) for o in surf_offs],
                                  [_crop(m, r, c, core, False) for m in surf_masks], _crop(ax, r, c, core, 0.0),
                                  kf, 1.0, 1.0, sp.fk, pix=3)
                acc_f = torch.zeros((nf, core, core), device=dev)
                acc_c = torch.zeros((nc, core, core), device=dev)
                for t in range(self.tta):
                    with timed("net", dev), torch.autocast("cuda", dtype=self.amp or torch.float32,
                                                           enabled=self.amp is not None):
                        o = self.m(_tta_input(xc, t), _tta_input(xf, t))
                    acc_f += _dihedral_inv(torch.softmax(o["fine"][0].float(), 0), t) / self.tta
                    lc = F.interpolate(o["ctx"][0, :1].float(), size=(C, C), mode="bilinear", align_corners=False)[0]
                    acc_c += _dihedral_inv(torch.softmax(lc, 0), t)[:, c0f:c0f + core, c0f:c0f + core] / self.tta
                g = gw[ra - r:rb - r, ca - c:cb - c]
                pf[:, ra:rb, ca:cb] += g * acc_f[:, ra - r:rb - r, ca - c:cb - c]
                pc[:, ra:rb, ca:cb] += g * acc_c[:, ra - r:rb - r, ca - c:cb - c]
                wsum[ra:rb, ca:cb] += g
        d = wsum.clamp(min=1e-6)
        return pf / d, pc / d, wsum > 0


def _crop(a, r0, c0, n, fill):
    """a [..., H, W] -> [..., n, n] window at (r0, c0), padded with fill."""
    H, W = a.shape[-2:]
    out = torch.full(a.shape[:-2] + (n, n), fill, dtype=a.dtype, device=a.device)
    ra, rb, ca, cb = max(r0, 0), min(r0 + n, H), max(c0, 0), min(c0 + n, W)
    if ra < rb and ca < cb:
        out[..., ra - r0:rb - r0, ca - c0:cb - c0] = a[..., ra:rb, ca:cb]
    return out


def run_sheet(inf: Inference, P_hint: torch.Tensor | None, V: torch.Tensor, green: torch.Tensor, P_green: torch.Tensor,
              rejected: torch.Tensor | None = None, passes: int = 3, frame_sigma: float = 6.0, nb_penalty: float = 2.0,
              log=None, start: torch.Tensor | None = None, early_stop: bool = True,
              bp_compiled: bool = False) -> dict:
    """The whole sheet. P_* [H, W, 3] xyz (float64), V / green / rejected [H, W] bool. `start` (xyz) replaces the
    hint as the starting estimate while the hint prompt stays P_hint; `early_stop` False runs every pass even once the
    frame has stopped moving."""
    dev = inf.dev
    sp = inf.sp
    H, W = V.shape
    Vn = V.cpu().numpy()
    P0 = P_green if P_hint is None else torch.where(green[..., None], P_green, P_hint)
    if start is not None:
        P0 = torch.where(green[..., None], P_green, start)
    if P_hint is None and start is None:                               # hint off: membrane extension of green
        X = GM.harmonic(P_green.permute(2, 0, 1).float(), green, V, iters=400, init_sigma=40.0)
        P0 = torch.where(green[..., None], P_green, X.permute(1, 2, 0).double())
    ax = torch.as_tensor(axis_direction(P0.cpu().numpy(), Vn), device=dev, dtype=torch.float32)
    E = P0.clone()
    maps = {}

    def frame(E_):
        Fp = GM.blur_masked(E_.permute(2, 0, 1).float(), V, frame_sigma).permute(1, 2, 0).double()
        Fp = torch.where(green[..., None], torch.where(V[..., None], Fp, E_), Fp)
        Fp = torch.where(V[..., None], Fp, E_)
        return Fp, normals(Fp, V, inf.axis)

    def offsets(Fp, Fn):
        """Prompt surfaces along the frame normal: green, hint, rejected (NaN where absent)."""
        g = ((P_green - Fp) * Fn).sum(-1).float()
        o = [torch.where(green, g, torch.full_like(g, float("nan")))]
        if P_hint is not None:
            h = ((P_hint - Fp) * Fn).sum(-1).float()
            o.append(torch.where(V, h, torch.full_like(h, float("nan"))))      # as in training: also on green
        else:
            o.append(torch.full_like(g, float("nan")))
        rj = rejected if rejected is not None and P_hint is not None else torch.zeros_like(V)
        o.append(torch.where(rj, o[1], torch.full_like(g, float("nan"))))
        return o, [green, torch.zeros_like(V), rj]                    # the trace is the frame only (no channel)

    # ---- locate
    Fp, Fn = frame(E)
    so, sm = offsets(Fp, Fn)
    pt, pnb, cov = inf.locate(Fp, Fn, V, so, sm, ax)
    Dc = (-torch.log(pt.clamp(min=1e-12))).clamp(max=10.0)
    if nb_penalty > 0:
        Dc = Dc + nb_penalty * pnb.max(0).values / pnb.max(0).values.amax(0, keepdim=True).clamp(min=1e-6)
    dx, dy = height_deltas(Fp, Fn, 1.0)
    pin = torch.where(green, (so[0] + sp.cout).clamp(0, sp.nc - 1), torch.full_like(so[0], float("nan")))
    with timed("bp", dev):
        r = BP.solve(Dc[None], (V & cov)[None], dx[None].float(), dy[None].float(), s=0.15, tau=8.0, iters=80,
                     pin_label=pin[None], compiled=bp_compiled)
    t = r["label"][0] - sp.cout
    E = torch.where((V & ~green)[..., None], Fp + t.double()[..., None] * Fn, P0)
    maps["locate_margin"] = r["margin"][0].float().cpu().numpy()
    maps["locate_entropy"] = (-(pt * torch.log(pt.clamp(min=1e-12))).sum(0)).cpu().numpy()
    if log:
        log.info(f"locate: |move| p50 {float(t[V & ~green].abs().median()):.2f} p90 "
                 f"{float(t[V & ~green].abs().quantile(0.9)):.2f} vox")
    # ---- refine
    for it in range(passes):
        Fp_new, Fn_new = frame(E)
        if it > 0:
            mv = ((Fp_new - Fp) * Fn).sum(-1).abs()[V]
            if log:
                log.info(f"refine {it}: frame moved p99 {float(mv.quantile(0.99)):.3f} vox")
            if early_stop and float(mv.quantile(0.99)) < 0.25:
                break
        Fp, Fn = Fp_new, Fn_new
        so, sm = offsets(Fp, Fn)
        pf, pc, covf = inf.refine(Fp, Fn, V, so, sm, ax)
        outside = pf[-1] > 0.5
        Df = (-torch.log(pf[:-1].clamp(min=1e-12))).clamp(max=10.0)
        dx, dy = height_deltas(Fp, Fn, 0.5)
        pin = torch.where(green, ((so[0] + sp.fout) / 0.5).clamp(0, sp.nf - 1), torch.full_like(so[0], float("nan")))
        with timed("bp", dev):
            r = BP.solve(Df[None], (V & covf & ~outside)[None], dx[None].float(), dy[None].float(), s=0.3, tau=6.0,
                         iters=60, pin_label=pin[None], compiled=bp_compiled)
        tf = r["label"][0] * 0.5 - sp.fout
        tc = readout(torch.log(pc.clamp(min=1e-12))[None], bins_ctx(sp, dev))["pred"][0]
        tt = torch.where(outside, tc, tf)
        E = torch.where((V & ~green)[..., None], Fp + tt.double()[..., None] * Fn, P0)
        maps["refine_margin"] = r["margin"][0].float().cpu().numpy()
        maps["outside"] = pf[-1].float().cpu().numpy()
        maps["refine_entropy"] = (-(pf * torch.log(pf.clamp(min=1e-12))).sum(0)).cpu().numpy()
        if log:
            log.info(f"refine {it}: |move| p50 {float(tt[V & ~green].abs().median()):.2f} p90 "
                     f"{float(tt[V & ~green].abs().quantile(0.9)):.2f} vox, outside {float(outside[V].float().mean()):.3f}")
    E = torch.where(green[..., None], P_green, E)
    return {"P": E, "maps": maps}
