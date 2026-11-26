"""SurfNet training samples (GPU): the CT is read along a smooth FRAME, never along the truth or the provisional surface.

Per crop (context grid C x C cells; the fine stream covers the central `core` x `core` cells):
1. lateral map (`spatial.lateral_map`): output cell -> region cell; the true geometry P, N at every cell;
2. green (verified) layout (`greens`), provisional surface g_prov (`generators`), where it is given (presence),
   rejected regions (weak negatives);
3. frame source = green (truth + label jitter) where verified, else the provisional (given or hidden) or the membrane
   extension of green; the FRAME = its low-pass in xyz (sigma 3-10 cells), as an offset field T along the true normal.
   The low-pass removes the truth's fine shape exactly as low-passing a trace does at inference;
4. `build(L, T)`: for any frame T (mode A: the frame above; mode B: a refined frame) both streams, the prompts and the
   targets:
   - context stream: coarse band, +-ck units of 2 vox (97 samples), 1 px per cell, C x C cells;
   - fine stream: fine band, +-fk units of 1 vox (25 samples), 3 px per cell, core x core cells; cells whose frame
     is > fine_max vox off (or in random fine-only holes) are invalid in the fine stream;
   - rays along the frame normal (`spatial.ray`: tilt, voxel factor, depth warp), target = the exact crossing with
     the true surface (vox along the ray);
   - channels (13 per stream): CT, then for green / provisional / rejected each a Gaussian in depth at the surface's
     position, a clipped ramp (still says "above / below" when the surface is outside the window) and a mask, then
     VALID, AXIS_Y, AXIS_X.
Targets: the context head over +-cout vox (1-vox bins) for the target winding and the windings k = -1 / +1 (where the
object has them); the fine head over +-fout vox (0.5-vox bins) + "outside"; the verify head = provisional within 1 vox.
Labels only where they are determinable (generators: no winding errors without green in view)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import fields as Rf
from . import intensity as I
from . import spatial as S
from . import generators as GN
from ..common import frames as GM
from . import greens as GR

IN_CH = 13
TIER_SIGMA_VOX = {1: 0.5, 2: 0.75, 3: 1.0}


@dataclass
class CropSpec:
    C: int = 160
    core: int = 80
    fk: int = 12
    fout: float = 10.0
    ck: int = 48
    cout: float = 80.0
    ppc: int = 3
    fine_max: float = 17.0
    ctx_max: float = 75.0
    prompt_sigma_f: float = 1.0
    prompt_sigma_c: float = 0.75


def bilinear(field: torch.Tensor, ry: torch.Tensor, rx: torch.Tensor) -> torch.Tensor:
    """field [C, R, R] sampled at region-cell coords ry, rx [h, w] -> [C, h, w] (border clamp)."""
    Rr = field.shape[-1]
    g = torch.stack([2 * rx / (Rr - 1) - 1, 2 * ry / (Rr - 1) - 1], -1)[None]
    return F.grid_sample(field[None].float(), g.float(), mode="bilinear", padding_mode="border", align_corners=True)[0]


def nearest(A: torch.Tensor, ry: torch.Tensor, rx: torch.Tensor, fill):
    Rr = A.shape[-1]
    r, c = ry.round().long(), rx.round().long()
    inside = (r >= 0) & (r < Rr) & (c >= 0) & (c < Rr)
    v = A[..., r.clamp(0, Rr - 1), c.clamp(0, Rr - 1)]
    return torch.where(inside, v, torch.full_like(v, fill))


def box_of(di: torch.Tensor, yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """[6] = mins and maxs of the index coordinates (still on the device)."""
    return torch.stack([di.min(), yi.min(), xi.min(), di.max(), yi.max(), xi.max()])


def sample_band(ct: torch.Tensor, di: torch.Tensor, yi: torch.Tensor, xi: torch.Tensor, bounds=None) -> torch.Tensor:
    """Trilinear sample of a uint8 band [Dv, Hv, Wv] at index coords [D, h, w] (zero outside), on the touched sub-box.
    `bounds` (6 floats: mins, maxs) may be read beforehand for several calls at once (one host read)."""
    Dv, Hv, Wv = ct.shape
    if bounds is None:
        bounds = box_of(di, yi, xi).tolist()
    lo = [int(max(0, math.floor(bounds[i]) - 1)) for i in range(3)]
    hi = [int(min(n - 1, math.ceil(bounds[3 + i]) + 1)) for i, n in enumerate((Dv, Hv, Wv))]
    if any(h_ < l_ for l_, h_ in zip(lo, hi)):
        return torch.zeros(di.shape, device=ct.device)
    sub = ct[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].float()
    sz = sub.shape

    def nrm(t, l_, n):
        return 2 * (t - l_) / max(n - 1, 1) - 1 if n > 1 else torch.zeros_like(t)

    g = torch.stack([nrm(xi, lo[2], sz[2]), nrm(yi, lo[1], sz[1]), nrm(di, lo[0], sz[0])], -1)
    return F.grid_sample(sub[None, None], g[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]


def k_of_u(u_t: torch.Tensor, u_grid: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
    """Invert the monotone ray parameter: u_grid [B, D, H, W] (band vox at input samples kk), u_t [B, H, W] -> the
    fractional input position k (linear extrapolation beyond the ends)."""
    B, D, H, W = u_grid.shape
    ug = u_grid.permute(0, 2, 3, 1).reshape(-1, D).contiguous()
    ut = u_t.reshape(-1, 1).contiguous().to(ug.dtype)
    i = torch.searchsorted(ug, ut).clamp(1, D - 1)
    u0 = torch.gather(ug, 1, i - 1)
    u1 = torch.gather(ug, 1, i)
    fr = (ut - u0) / (u1 - u0).clamp(min=1e-6)
    k0, k1 = kk[i - 1], kk[i]
    return (k0 + fr * (k1 - k0)).view(B, H, W)


def median3(x: torch.Tensor) -> torch.Tensor:
    """3 x 3 median of [h, w]."""
    p = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")
    u = F.unfold(p, 3)                                               # [1, 9, h*w]
    return u.median(1).values.view_as(x)


class Sampler:
    def __init__(self, pool, cfg: dict, device="cuda", seed: int = 0):
        self.pool = pool
        self.cfg = cfg
        self.sp = CropSpec(**cfg.get("crops", {}).get("spec", {}))
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.gen_cpu = torch.Generator()
        self.gen_cpu.manual_seed(seed + 1)
        sp = dict(cfg.get("spatial", {}))
        sp.setdefault("scale", [0.85, 1.15])
        sp["voxel"] = [max(0.9, sp.get("voxel", [0.9, 1.1])[0]), min(1.1, sp.get("voxel", [0.9, 1.1])[1])]
        self.sp_cfg = sp
        self.gcfg = dict(cfg.get("crops", {}).get("greens", {}))
        self.ncfg = dict(cfg.get("crops", {}).get("generators", {}))
        self.icfg = cfg.get("intensity", {})
        self.hole_p = float(cfg.get("augment", {}).get("p_holes", 0.3))
        self.fine_hole_p = float(cfg.get("crops", {}).get("fine_hole_p", 0.2))

    def seed(self, s: int):
        self.gen.manual_seed(s)
        self.gen_cpu.manual_seed(s + 1)

    # ------------------------------------------------------------------ the randomised snapping tracer
    def _snap(self, slot, ry, rx, face, seed_g, U):
        gen, dev = self.gen, self.device
        spec = slot["spec"]
        ctc = slot["ctc"]
        rad = int(math.ceil(Rf.su(gen, 2.0, 6.0)))
        js = torch.arange(-rad, rad + 1, device=dev, dtype=torch.float32)
        feat = Rf.si(gen, 0, 5)
        kern = torch.randn(5, generator=gen, device=dev) if feat == 4 else None
        gam = Rf.su(gen, 0.6, 1.6) if Rf.sc(gen, 0.5) else 1.0
        lat = Rf.su(gen, 0.0, 2.0)
        g = seed_g.clone()
        rounds = Rf.si(gen, 1, 5)
        b = torch.stack([seed_g.min(), seed_g.max(), ry.min(), ry.max(), rx.min(), rx.max()]).tolist()
        reach = rounds * rad + rad + 1                                    # g moves <= rad per round
        box = [(face + b[0] - reach + spec.coarse_half) / spec.coarse_step, b[2], b[4],
               (face + b[1] + reach + spec.coarse_half) / spec.coarse_step, b[3], b[5]]
        for _ in range(rounds):
            dep = face + g[None] + js.view(-1, 1, 1)
            di = (dep + spec.coarse_half) / spec.coarse_step
            prof = sample_band(ctc, di, ry[None].expand_as(di), rx[None].expand_as(di), box) / 255.0
            prof = prof.clamp(min=0) ** gam
            if feat == 0:
                f = prof
            elif feat in (1, 2):
                d = torch.zeros_like(prof)
                d[1:-1] = prof[2:] - prof[:-2]
                f = -d if feat == 1 else d
            elif feat == 3:
                k = torch.exp(-0.5 * (torch.arange(-4, 5, device=dev, dtype=torch.float32) / 2.0) ** 2)
                k = (k / k.sum()).view(1, 1, -1)
                sm = F.conv1d(prof.permute(1, 2, 0).reshape(-1, 1, prof.shape[0]), k, padding=4)
                f = prof - sm.reshape(prof.shape[1], prof.shape[2], -1).permute(2, 0, 1)
            else:
                sm = F.conv1d(prof.permute(1, 2, 0).reshape(-1, 1, prof.shape[0]), kern.view(1, 1, -1), padding=2)
                f = sm.reshape(prof.shape[1], prof.shape[2], -1).permute(2, 0, 1)
            f = f - 0.02 * js.abs().view(-1, 1, 1) * f.std().clamp(min=1e-6)      # nearest maximum preferred
            g = g + js[f.argmax(0)]
            g = median3(g)
            if lat > 0.25:
                g = GM.blur_masked(g[None], U, lat)[0]
        return g

    # ------------------------------------------------------------------ layout (frame-independent)
    @torch.no_grad()
    def layout(self, B: int, train: bool = True, slots=None) -> dict:
        dev, gen, sp = self.device, self.gen, self.sp
        C = sp.C
        slots = slots or self.pool.pick(B, self.gen_cpu)
        Rr = self.pool.R
        cell_vox_l = [s["cell_vox"] for s in slots]
        margin = sp.ck * 2.0 * 1.1 * 0.35 / min(cell_vox_l)
        lm = S.lateral_map(B, C, C, Rr, gen, dev, self.sp_cfg, train, margin=margin)
        yo, xo = Rf.grid(C, C, dev)
        ry, rx = lm.map(yo.expand(B, C, C), xo.expand(B, C, C))
        U = torch.stack([self.pool_usable(s["usable"], ry[b], rx[b]) for b, s in enumerate(slots)])
        if train:
            for b in range(B):
                if Rf.sc(gen, self.hole_p):
                    U[b] &= ~Rf.blobs(1, C, C, gen, dev, frac=[Rf.su(gen, 0.02, 0.12)], scale=[Rf.su(gen, 4.0, 24.0)])[0]
        tier = torch.stack([nearest(s["tier"], ry[b], rx[b], 0) for b, s in enumerate(slots)])
        onsheet = torch.stack([nearest(s["onsheet"], ry[b], rx[b], False) for b, s in enumerate(slots)])
        seg = torch.stack([torch.full((), bool(s.get("seg", False)), device=dev) for s in slots]).view(B, 1, 1)
        tier = torch.where(seg & (tier > 0), torch.full_like(tier, 3), tier)
        P = torch.stack([bilinear(s["P"], ry[b], rx[b]) for b, s in enumerate(slots)])          # [B, 3, C, C]
        N = torch.stack([bilinear(s["N"], ry[b], rx[b]) for b, s in enumerate(slots)])
        N = N / N.norm(dim=1, keepdim=True).clamp(min=1e-6)
        face = torch.stack([torch.full((), float(s["face"]), device=dev) for s in slots])
        face_l = [float(s["face"]) for s in slots]
        spacing = [s["spacing"] for s in slots]
        ks = next((s["ks"] for s in slots if "nbr" in s and s["ks"]), [])
        nbr = None
        if ks:
            nan = torch.full((len(ks), C, C), float("nan"), device=dev)
            nbr = torch.stack([nearest(s["nbr"], ry[b], rx[b], float("nan")) if "nbr" in s and s["ks"] == ks else nan
                               for b, s in enumerate(slots)])
        target_ok = U & (tier > 0) & onsheet
        gfn = self.gcfg.get("layout_fn") or GR.green_masks                   # tests: fixed layouts (adjacent.py)
        G, gcls, far = gfn(B, C, C, sp.core, gen, dev, self.gcfg)
        G = G & target_ok
        ident = G.flatten(1).any(1).tolist()                              # one host read per batch

        hp_fns, ext_fns, snap_fns = [], [], []
        for b in range(B):
            def hp(sig, b=b):                                         # the truth's high-pass along N: (P - LP(P)) . N
                lp = GM.lowpass_xyz(P[b:b + 1], U[b:b + 1], sig)
                return GM.offset_of(P[b:b + 1], lp, N[b:b + 1])[0]

            def ext(b=b, cache={}):                                   # once per sample and layout
                if b not in cache:
                    fixed = G[b] if ident[b] else Rf.disk(1, C, C, gen, dev, 4.0, 12.0)[0] & U[b]
                    cache[b] = GM.offset_of(GM.harmonic(P[b], fixed, U[b], iters=80, check=ident[b])[None],
                                            P[b:b + 1], N[b:b + 1])[0]
                return cache[b]

            def snap(seed_g, b=b):
                return self._snap(slots[b], ry[b], rx[b], face_l[b], seed_g, U[b])
            hp_fns.append(hp)
            ext_fns.append(ext)
            snap_fns.append(snap)
        ncfg = dict(self.ncfg)
        prov_fn = ncfg.get("provisional_fn") or GN.provisional                # tests: held-out start families
        g_prov, kind = prov_fn(B, C, C, gen, dev, spacing, G, nbr, ks, hp_fns, ext_fns, snap_fns, ncfg)
        amp_cap = ncfg.get("amp_cap")
        if amp_cap is not None:
            g_prov = torch.stack([g_prov[b].clamp(-amp_cap * spacing[b], amp_cap * spacing[b]) for b in range(B)])
        # the trace (provisional) is NOT shown as an input channel by default: it enters only as the reading frame
        # (crops.trace_channel = "none": a channel with the exact position is copied by the model instead of corrected)
        pmode = ncfg.get("presence", self.cfg.get("crops", {}).get("trace_channel", "none"))
        if pmode == "all":
            pres = U.clone()
        elif pmode == "none":
            pres = torch.zeros_like(U)
        else:
            pres = GN.presence(B, C, C, gen, dev) & U
        wrong = (g_prov.abs() > 1.0) & U
        prj = float(ncfg.get("p_rejected", self.cfg.get("crops", {}).get("p_rejected", 0.25)))
        rej = GR.rejected_masks(wrong, U & ~G & target_ok, gen, dev, prj)    # rejected areas: "not this surface"
        # green as the frame source: truth + label jitter (+ a small smooth bias)
        gG = 0.3 * torch.randn((B, C, C), generator=gen, device=dev)
        for b in range(B):
            if Rf.sc(gen, 0.3):
                gG[b] = gG[b] + Rf.su(gen, 0.2, 0.5) * Rf.smooth_noise(1, C, C, 24.0, gen, dev)[0] / 2
        src = g_prov.clone()                                         # the frame source: the trace ...
        p_ext = float(ncfg.get("p_ext_frame", self.cfg.get("crops", {}).get("p_ext_frame", 0.2)))
        for b in range(B):                                            # ... or the extension of green (untraced areas)
            if ident[b] and Rf.sc(gen, p_ext):
                e = ext_fns[b]()
                # only when the extrapolation stays within 1.5 windings (80 % of the cells): a membrane extended across
                # a curved crop can drift several windings, and the +-ctx_max clip would then make the frame run
                # exactly parallel to the truth
                if float(torch.quantile(e[U[b]].abs().float(), 0.8)) <= 1.5 * float(spacing[b]):
                    src[b] = torch.where(pres[b], src[b], e)
        src = torch.where(G, gG, src)
        sigf = [Rf.su(gen, 3.0, 10.0) for _ in range(B)]
        T_A = GM.offset_of(GM.lowpass_xyz(P + src[:, None] * N, U, sigf), P, N).clamp(-sp.ctx_max, sp.ctx_max)
        ax_cells = torch.stack([nearest(s["axdir"], ry[b], rx[b], 0.0) for b, s in enumerate(slots)])
        a_, b_, c_, d_ = lm.A[:, 0, 0], lm.A[:, 0, 1], lm.A[:, 1, 0], lm.A[:, 1, 1]
        Ainv = torch.stack([torch.stack([d_, -b_], -1), torch.stack([-c_, a_], -1)], -2) / (a_ * d_ - b_ * c_).view(B, 1, 1)
        ax = torch.einsum("bij,bjhw->bihw", Ainv, ax_cells)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-6) * (ax_cells.norm(dim=1, keepdim=True) > 0)
        sig_lab = torch.ones_like(T_A)
        for t_, sv in TIER_SIGMA_VOX.items():
            sig_lab = torch.where(tier == t_, torch.full_like(sig_lab, sv), sig_lab)
        return {"slots": slots, "lm": lm, "ry": ry, "rx": rx, "U": U, "tier": tier, "target_ok": target_ok, "P": P,
                "N": N, "face": face, "spacing": spacing, "ks": ks, "nbr": nbr, "G": G, "gG": gG, "gcls": gcls,
                "far": far, "g_prov": g_prov, "kind": kind, "pres": pres, "rej": rej, "T_A": T_A, "ax": ax,
                "sig_lab": sig_lab, "cell_vox": torch.stack([torch.full((), float(v), device=dev) for v in cell_vox_l]),
                "cell_vox_min": float(min(cell_vox_l)), "B": B}

    def pool_usable(self, Ug, ry, rx):
        Rr = Ug.shape[-1]
        inside = (ry >= 0) & (ry <= Rr - 1) & (rx >= 0) & (rx <= Rr - 1)
        r0 = ry.floor().long().clamp(0, Rr - 2)
        c0 = rx.floor().long().clamp(0, Rr - 2)
        return inside & Ug[r0, c0] & Ug[r0 + 1, c0] & Ug[r0, c0 + 1] & Ug[r0 + 1, c0 + 1]

    def refine_frame(self, L: dict, offset: torch.Tensor, sig=None, gen=None) -> torch.Tensor:
        """A frame from a surface given as offsets along N (vox): its low-pass in xyz (sigma 3-8 cells; green kept as the
        frame source), clipped to the fine stream's range."""
        B = L["B"]
        src = torch.where(L["G"], L["gG"], offset)
        sig = sig or [Rf.su(gen or self.gen, 3.0, 8.0) for _ in range(B)]
        return GM.offset_of(GM.lowpass_xyz(L["P"] + src[:, None] * L["N"], L["U"], sig), L["P"], L["N"])

    def synthetic_refine_offset(self, L: dict, gen=None) -> torch.Tensor:
        """A post-locate estimate: truth + a broadband residual (sigma log-U(0.3, 4) vox), p 0.15 a 6-15-vox patch."""
        gen, dev, C, B = gen or self.gen, self.device, self.sp.C, L["B"]
        out = torch.zeros((B, C, C), device=dev)
        for b in range(B):
            s = math.exp(Rf.su(gen, math.log(0.3), math.log(4.0)))
            f = GN.powerlaw(C, C, gen, dev, Rf.su(gen, 1.5, 3.0)) * s
            if Rf.sc(gen, 0.15):
                m = Rf.blur(Rf.blobs(1, C, C, gen, dev, frac=[Rf.su(gen, 0.03, 0.2)]).float(), 2.0)[0].clamp(0, 1)
                f = f + m * Rf.su(gen, 6.0, 15.0) * (1 if Rf.sc(gen, 0.5) else -1)
            out[b] = f
        return out

    # ------------------------------------------------------------------ inputs and targets for a frame
    @torch.no_grad()
    def build(self, L: dict, T: torch.Tensor, train: bool = True, fine: bool = True, gen=None) -> dict:
        dev, gen, sp = self.device, gen or self.gen, self.sp
        B, C = L["B"], sp.C
        slots, lm, ry, rx, U, face = L["slots"], L["lm"], L["ry"], L["rx"], L["U"], L["face"]
        T = torch.where(U, T, GM.blur_masked(torch.where(U, T, 0.0)[:, None], U, 4.0)[:, 0])   # invalid cells: smooth fill
        T = torch.nan_to_num(T, nan=0.0).clamp(-sp.ctx_max, sp.ctx_max)
        rays = S.ray(T, lm, L["cell_vox"], gen, dev, self.sp_cfg, train)
        sig = rays.sigma.view(B, 1, 1)
        out = {"L": L, "T": T, "sigc": sig * rays.c}
        # ---------------- context stream
        kc = torch.arange(-sp.ck, sp.ck + 1, device=dev, dtype=torch.float32)
        ksc = S.target(T, rays, 2.0)
        uc = S.ray_parameter(kc, ksc, rays, 2.0)                                       # [B, Dc, C, C] band vox
        bdep = face.view(B, 1, 1, 1) + T[:, None] + rays.c[:, None] * uc
        by = ry[:, None] + uc * rays.lat[:, 0:1]
        bx = rx[:, None] + uc * rays.lat[:, 1:2]
        ctc = torch.empty((B, 1, kc.numel(), C, C), device=dev)
        dis = [(bdep[b] + s["spec"].coarse_half) / s["spec"].coarse_step for b, s in enumerate(slots)]
        boxes = torch.stack([box_of(dis[b], by[b], bx[b]) for b in range(B)]).tolist()
        for b, s in enumerate(slots):
            ctc[b, 0] = sample_band(s["ctc"], dis[b], by[b], bx[b], boxes[b])
        vc = U[:, None, None]
        if train:
            ctc = I.augment(ctc, vc, gen, (2.0, L["cell_vox_min"]), self.icfg, depth=bdep - face.view(B, 1, 1, 1))
        else:
            ctc = I.normalise_eval(ctc, vc)
        prom_c = self._prompts(L, T, rays, uc, kc, 2.0, sp.prompt_sigma_c, sp.ck, None)
        xdt = torch.bfloat16 if self.cfg.get("crops", {}).get("input_bf16", True) else torch.float32
        out["x_ctx"] = torch.cat([ctc, prom_c, vc.float().expand(-1, 1, kc.numel(), -1, -1),
                                  L["ax"][:, :, None].expand(-1, -1, kc.numel(), -1, -1)], 1).to(xdt)
        t_c = -T / (sig * rays.c)                                                       # vox along the ray
        w_c = L["target_ok"] & (t_c.abs() <= sp.cout - 1)
        wt = torch.where(L["G"], torch.full_like(t_c, 0.1), torch.ones_like(t_c))
        wt = torch.where((L["g_prov"].abs() > 1.0) & ~L["G"], 2 * wt, wt)    # cells whose trace is wrong
        out.update(t_ctx=t_c, w_ctx=w_c.float() * wt, sig_ctx=(L["sig_lab"] / sig).clamp(min=0.5))
        nb_t, nb_w = [], []
        for k in (-1, 1):
            if L["nbr"] is not None and k in L["ks"]:
                f = L["nbr"][:, L["ks"].index(k)]
                tn = (f - T) / (sig * rays.c)
                ok = torch.isfinite(tn) & U & (tn.abs() <= sp.cout - 1)
                nb_t.append(torch.nan_to_num(tn, nan=0.0))
                nb_w.append(ok.float())
            else:
                nb_t.append(torch.zeros_like(T))
                nb_w.append(torch.zeros_like(T))
        out.update(t_nb=torch.stack(nb_t, 1), w_nb=torch.stack(nb_w, 1))
        if not fine:
            return out
        # ---------------- fine stream (core)
        p, core = sp.ppc, sp.core
        c0 = (C - core) // 2
        a = torch.arange(p * core, device=dev, dtype=torch.float32)
        ya, xa = torch.meshgrid(c0 + (a - (p - 1) / 2) / p, c0 + (a - (p - 1) / 2) / p, indexing="ij")
        ya, xa = ya.expand(B, -1, -1), xa.expand(B, -1, -1)
        ryp, rxp = lm.map(ya, xa)
        cells = torch.cat([T[:, None], rays.c[:, None], rays.lat], 1)
        cp = S.sample_cells(cells, ya, xa)
        Tp, cpx, laty, latx = cp[:, 0], cp[:, 1], cp[:, 2], cp[:, 3]
        kf = torch.arange(-sp.fk, sp.fk + 1, device=dev, dtype=torch.float32)
        ksf = -Tp / (rays.sigma.view(B, 1, 1) * cpx)
        uf = S.ray_parameter(kf, ksf, rays, 1.0)
        bdf = face.view(B, 1, 1, 1) + Tp[:, None] + cpx[:, None] * uf
        byf = ryp[:, None] + uf * laty[:, None]
        bxf = rxp[:, None] + uf * latx[:, None]
        ci = torch.div(torch.arange(p * core, device=dev), p, rounding_mode="floor") + c0
        fine_ok = U & (T.abs() <= sp.fine_max)
        if train:
            for b in range(B):
                if Rf.sc(gen, self.fine_hole_p):
                    fine_ok[b] &= ~Rf.blobs(1, C, C, gen, dev, frac=[Rf.su(gen, 0.03, 0.2)], scale=[Rf.su(gen, 6, 24)])[0]
        Upf = torch.stack([self.pool_usable(s["usable"], ryp[b], rxp[b]) for b, s in enumerate(slots)])
        Upf = Upf & fine_ok[:, ci][:, :, ci]
        ctf = torch.empty((B, 1, kf.numel(), p * core, p * core), device=dev)
        crd = [((bdf[b] + s["spec"].fine_half) / s["spec"].fine_step, s["spec"].ppc * byf[b] + (s["spec"].ppc - 1) / 2,
                s["spec"].ppc * bxf[b] + (s["spec"].ppc - 1) / 2) for b, s in enumerate(slots)]
        boxes = torch.stack([box_of(*crd[b]) for b in range(B)]).tolist()
        for b, s in enumerate(slots):
            ctf[b, 0] = sample_band(s["ct"], *crd[b], boxes[b])
        vf = Upf[:, None, None]
        ctf = torch.where(vf, ctf, torch.zeros_like(ctf))
        if train:
            ctf = I.augment(ctf, vf, gen, (1.0, L["cell_vox_min"] / p), self.icfg, depth=bdf - face.view(B, 1, 1, 1))
        else:
            ctf = I.normalise_eval(ctf, vf)
        prom_f = self._prompts(L, T, rays, uf, kf, 1.0, sp.prompt_sigma_f, sp.fk, (ya, xa, ci))
        axp = L["ax"][:, :, ci][:, :, :, ci] * Upf[:, None]
        out["x_fine"] = torch.cat([ctf, prom_f, vf.float().expand(-1, 1, kf.numel(), -1, -1),
                                   axp[:, :, None].expand(-1, -1, kf.numel(), -1, -1)], 1).to(xdt)
        sl = (slice(None), slice(c0, c0 + core), slice(c0, c0 + core))
        t_f = t_c[sl]
        w_f = (L["target_ok"] & fine_ok)[sl].float() * wt[sl]
        out.update(t_fine=t_f, w_fine=w_f, sig_fine=out["sig_ctx"][sl], outside=t_f.abs() > sp.fout,
                   verify=(T.abs() < 1.0)[sl].float(),                   # is the current frame on the true surface?
                   w_verify=L["target_ok"][sl].float(), fine_ok=fine_ok[sl])
        return out

    def _prompts(self, L, T, rays, u_grid, kk, step, psig, khalf, pix):
        """Gaussian / ramp / mask for green, provisional and rejected surfaces: [B, 9, D, H, W]."""
        B = L["B"]
        sig = rays.sigma.view(B, 1, 1)
        groups = ((L["gG"], L["G"]), (L["g_prov"], L["pres"]), (L["g_prov"], L["rej"]))
        chans = []
        for g, m in groups:
            u_t = (g - T) / rays.c                                                     # band vox along the ray
            if pix is not None:
                ya, xa, ci = pix
                u_t = S.sample_cells(u_t[:, None], ya, xa)[:, 0]
                m = m[:, ci][:, :, ci]
            k = k_of_u(u_t, u_grid, kk / 1.0)
            kd = kk.view(1, -1, 1, 1)
            mf = m[:, None].float()
            gauss = torch.exp(-((kd - k[:, None]) ** 2) / (2 * psig ** 2)) * mf
            ramp = ((kd - k[:, None]) / khalf).clamp(-1, 1) * mf
            chans += [gauss, ramp, mf.expand_as(gauss)]
        del sig
        return torch.stack(chans, 1)
