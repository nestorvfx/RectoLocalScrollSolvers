"""The SurfNet network: one model, two input streams (DeepMedic / HookNet-style two pathways; PWC-Net-style joint
coarse-to-fine heads), trained jointly.

fine stream     [B, 13, 25, 240, 240]: +-12 vox at 1 vox, 3 px per cell, 80 x 80 cells
context stream  [B, 13, 97, 160, 160]: +-96 vox at 2 vox, 1 px per cell, 160 x 160 cells (sees +-7 windings)

Context decoder features are cropped to the fine stream's region (centre half laterally, +-12.5 vox of depth) and
resampled onto the fine decoder at 4 scales (concatenation + 1 x 1 conv).
Heads:
- fine: logits over +-10 vox in 0.5-vox bins (41) + "outside" (|offset| > 10), per fine cell [B, 42, 80, 80];
- context: logits over +-80 vox in 1-vox bins (161) for the target winding and the windings k = -1 / +1, at 2-cell
  lateral resolution [B, 3, 161, 80, 80] (inference upsamples to cells);
- verify: P(the provisional surface is within 1 vox) per fine cell [B, 80, 80].
Without the fine stream (mode A, `fine=None`) only the context heads run."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GroupNormAct(nn.GroupNorm):
    """GroupNorm in the input's dtype (bf16 under autocast) instead of autocast's fp32 upcast."""

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type) and x.dtype in (torch.bfloat16, torch.float16):
            with torch.autocast(x.device.type, enabled=False):
                return F.group_norm(x, self.num_groups, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
        return super().forward(x)


def norm(c: int) -> nn.Module:
    return GroupNormAct(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride=(1, 1, 1)):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = norm(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = norm(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = None
        if cin != cout or tuple(stride) != (1, 1, 1):
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), norm(cout))

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + (x if self.skip is None else self.skip(x)))


ACT = lambda: nn.LeakyReLU(0.01, inplace=True)                         # noqa: E731


@dataclass
class NetSpec:
    in_ch: int = 13
    f_stem: int = 32
    f_ch: list = field(default_factory=lambda: [32, 64, 128, 192, 256])
    c_stem: int = 16
    c_ch: list = field(default_factory=lambda: [16, 32, 64, 128, 192])
    strides: list = field(default_factory=lambda: [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]])
    blocks: list = field(default_factory=lambda: [2, 2, 2, 2, 2])
    fk: int = 12
    fout: float = 10.0
    ck: int = 48
    cout: float = 80.0
    ppc: int = 3
    checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "NetSpec":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    @property
    def nf(self) -> int:
        return int(round(2 * self.fout / 0.5)) + 1

    @property
    def nc(self) -> int:
        return int(round(2 * self.cout)) + 1


class Trunk(nn.Module):
    def __init__(self, cin, stem_ch, chans, strides, blocks, stem_kernel, stem_stride, stem_pad, ckpt=False):
        super().__init__()
        self.ckpt = ckpt
        self.stem = nn.Sequential(nn.Conv3d(cin, stem_ch, stem_kernel, stride=stem_stride, padding=stem_pad, bias=False),
                                  norm(stem_ch), ACT())
        self.enc = nn.ModuleList()
        c = stem_ch
        for ch, st, nb in zip(chans, strides, blocks):
            self.enc.append(nn.Sequential(ResBlock(c, ch, st), *[ResBlock(ch, ch) for _ in range(nb - 1)]))
            c = ch
        self.dec = nn.ModuleList()
        for i in range(len(chans) - 1, 0, -1):
            cup, cskip = chans[i], chans[i - 1]
            self.dec.append(nn.Sequential(nn.Conv3d(cup + cskip, cskip, 1, bias=False), norm(cskip), ACT(),
                                          ResBlock(cskip, cskip)))

    def _run(self, m, x):
        return checkpoint(m, x, use_reentrant=False) if self.ckpt and self.training else m(x)

    def encode(self, x):
        y = self.stem(x)
        skips = []
        for st in self.enc:
            y = self._run(st, y)
            skips.append(y)
        return skips

    def decode_step(self, j, y, skips):
        sk = skips[-2 - j]
        y = F.interpolate(y, size=sk.shape[2:], mode="trilinear", align_corners=False)
        return self._run(self.dec[j], torch.cat([y, sk], 1))


def roi(e: torch.Tensor, size, dfrac: float, lfrac: float = 0.5) -> torch.Tensor:
    """Resample the central region of e [B, C, D, H, W] (depth +-dfrac, lateral +-lfrac of the normalised extent)
    onto `size` (D', H', W')."""
    B = e.shape[0]
    D, H, W = size
    zd = torch.linspace(-dfrac, dfrac, D, device=e.device) * (D - 1) / max(D, 1) if D > 1 else torch.zeros(1, device=e.device)
    zy = torch.linspace(-lfrac, lfrac, H, device=e.device) * (H - 1) / max(H, 1)
    zx = torch.linspace(-lfrac, lfrac, W, device=e.device) * (W - 1) / max(W, 1)
    gd, gy, gx = torch.meshgrid(zd, zy, zx, indexing="ij")
    g = torch.stack([gx, gy, gd], -1)[None].expand(B, -1, -1, -1, -1).to(e.dtype)
    return F.grid_sample(e, g, mode="bilinear", padding_mode="border", align_corners=False)


class TwoStream(nn.Module):
    def __init__(self, spec: NetSpec):
        super().__init__()
        self.spec = s = spec
        p = s.ppc
        self.fine = Trunk(s.in_ch, s.f_stem, s.f_ch, s.strides, s.blocks, (3, p, p), (1, p, p), (1, 0, 0), s.checkpointing)
        self.ctx = Trunk(s.in_ch, s.c_stem, s.c_ch, s.strides, s.blocks, 3, (2, 2, 2), 1, s.checkpointing)
        nlev = len(s.f_ch) - 1
        self.fuse = nn.ModuleList()
        for j in range(nlev):
            cf = s.f_ch[nlev - 1 - j]
            cc = s.c_ch[nlev - 1 - j]
            self.fuse.append(nn.Sequential(nn.Conv3d(cf + cc, cf, 1, bias=False), norm(cf), ACT()))
        c0f, c0c = s.f_ch[0], s.c_ch[0]
        self.head_f = nn.Conv3d(c0f, 2, 1)                              # 2 sub-positions per 1-vox sample: 0.5 bins
        self.att_f = nn.Conv3d(c0f, 1, 1)
        self.outside = nn.Conv2d(c0f, 1, 1)
        self.verify = nn.Sequential(nn.Conv2d(c0f, c0f, 1), ACT(), nn.Conv2d(c0f, 1, 1))
        self.head_c = nn.Conv3d(c0c, 4 * 3, 1)                          # 4 sub-positions per 4-vox stem sample: 1-vox bins
        self.first_f = int(round((s.fk - s.fout) / 0.5))
        self.first_c = int(round(2 * s.ck - s.cout))
        self.dfrac = (s.fk + 0.5) / (2 * s.ck + 1)

    def forward_ctx(self, xc):
        sk = self.ctx.encode(xc)
        y = sk[-1]
        feats = []
        for j in range(len(self.ctx.dec)):
            y = self.ctx.decode_step(j, y, sk)
            feats.append(y)
        B, C, D, H, W = y.shape
        with torch.autocast(y.device.type, enabled=False):
            lg = self.head_c(y.float())                                  # [B, 12, D, H, W]
        lg = lg.view(B, 3, 4, D, H, W).permute(0, 1, 3, 2, 4, 5).reshape(B, 3, D * 4, H, W)
        lg = lg[:, :, self.first_c:self.first_c + self.spec.nc]
        return lg, feats

    def forward(self, xc, xf=None):
        lc, feats = self.forward_ctx(xc)
        if xf is None:
            return {"ctx": lc}
        sk = self.fine.encode(xf)
        y = sk[-1]
        for j in range(len(self.fine.dec)):
            y = self.fine.decode_step(j, y, sk)
            e = roi(feats[j], y.shape[2:], self.dfrac)
            y = self.fuse[j](torch.cat([y, e.to(y.dtype)], 1))
        B, C, D, H, W = y.shape
        with torch.autocast(y.device.type, enabled=False):
            lf = self.head_f(y.float())
        lf = lf.permute(0, 2, 1, 3, 4).reshape(B, D * 2, H, W)[:, self.first_f:self.first_f + self.spec.nf]
        a = torch.softmax(self.att_f(y).float(), 2).to(y.dtype)
        pooled = (a * y).sum(2)
        out_l = self.outside(pooled.float())[:, 0]
        ver = self.verify(pooled)[:, 0]
        return {"ctx": lc, "fine": torch.cat([lf, out_l[:, None].float()], 1), "verify": ver}


def bins_fine(spec: NetSpec, device) -> torch.Tensor:
    return torch.arange(spec.nf, device=device, dtype=torch.float32) * 0.5 - spec.fout


def bins_ctx(spec: NetSpec, device) -> torch.Tensor:
    return torch.arange(spec.nc, device=device, dtype=torch.float32) - spec.cout


# ------------------------------------------------------------------------------------------------ losses
def _soft(t, sig, bins):
    d = bins.view(1, -1, 1, 1) - t[:, None]
    q = torch.exp(-0.5 * (d / sig[:, None]) ** 2)
    return q / q.sum(1, keepdim=True).clamp(min=1e-12)


def dist_loss(logits, t, sig, w, bins, lam_emd=0.1):
    logp = F.log_softmax(logits.float(), 1)
    q = _soft(t.clamp(float(bins[0]), float(bins[-1])), sig, bins)
    ce = -(q * logp).sum(1)
    emd = ((torch.cumsum(logp.exp(), 1) - torch.cumsum(q, 1)) ** 2).sum(1) * float(bins[1] - bins[0])
    ws = w.sum().clamp(min=1.0)
    return (ce * w).sum() / ws + lam_emd * (emd * w).sum() / ws


def fine_loss(logits, t, sig, w, outside, bins, lam_emd=0.1):
    """logits [B, nf + 1] (last = outside). In range: CE vs a Gaussian over the bins (outside 0) + EMD; out of range:
    CE vs the outside class."""
    logp = F.log_softmax(logits.float(), 1)
    q = _soft(t.clamp(float(bins[0]), float(bins[-1])), sig, bins)
    ce_in = -(q * logp[:, :-1]).sum(1)
    ce_out = -logp[:, -1]
    ce = torch.where(outside, ce_out, ce_in)
    pin = F.softmax(logits[:, :-1].float(), 1)
    emd = ((torch.cumsum(pin, 1) - torch.cumsum(q, 1)) ** 2).sum(1) * float(bins[1] - bins[0])
    ws = w.sum().clamp(min=1.0)
    win = w * (~outside).float()
    return (ce * w).sum() / ws + lam_emd * (emd * win).sum() / win.sum().clamp(min=1.0)


def total_loss(out: dict, bt: dict, spec: NetSpec, mode: str, cfg: dict) -> dict:
    dev = out["ctx"].device
    bc = bins_ctx(spec, dev)
    sub = (slice(None), slice(0, None, 2), slice(0, None, 2))       # context head: 2-cell lateral resolution
    lam_nb = float(cfg.get("lam_nb", 0.3))
    l_ctx = dist_loss(out["ctx"][:, 0], bt["t_ctx"][sub], bt["sig_ctx"][sub], bt["w_ctx"][sub], bc)
    l_nb = sum(dist_loss(out["ctx"][:, 1 + i], bt["t_nb"][:, i][sub], bt["sig_ctx"][sub], bt["w_nb"][:, i][sub], bc)
               for i in range(2))
    res = {"ctx": l_ctx, "nb": l_nb}
    if mode == "A":
        res["total"] = l_ctx + lam_nb * l_nb
        return res
    bf = bins_fine(spec, dev)
    l_f = fine_loss(out["fine"], bt["t_fine"], bt["sig_fine"], bt["w_fine"], bt["outside"], bf)
    l_v = F.binary_cross_entropy_with_logits(out["verify"].float(), bt["verify"], reduction="none")
    l_v = (l_v * bt["w_verify"]).sum() / bt["w_verify"].sum().clamp(min=1.0)
    res.update(fine=l_f, verify=l_v)
    res["total"] = l_f + float(cfg.get("lam_ctx_b", 0.5)) * l_ctx + lam_nb * l_nb + float(cfg.get("lam_verify", 0.2)) * l_v
    return res


# ------------------------------------------------------------------------------------------------ read-out
def readout(logits: torch.Tensor, bins: torch.Tensor, win: int = 2, conf_half: float = 1.0) -> dict:
    """logits [B, nb, h, w] -> pred (units), conf (probability mass within +-conf_half units of pred), entropy (nats),
    p (probabilities)."""
    p = torch.softmax(logits.float(), 1)
    nb = p.shape[1]
    m = p.argmax(1, keepdim=True)
    idx = (m + torch.arange(-win, win + 1, device=p.device).view(1, -1, 1, 1)).clamp(0, nb - 1)
    pw = torch.gather(p, 1, idx)
    bw = bins[idx]
    pred = (pw * bw).sum(1) / pw.sum(1).clamp(min=1e-12)
    near = (bins.view(1, -1, 1, 1) - pred[:, None]).abs() <= conf_half
    conf = (p * near).sum(1)
    ent = -(p * torch.log(p.clamp(min=1e-12))).sum(1)
    return {"pred": pred, "conf": conf, "entropy": ent, "p": p}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
