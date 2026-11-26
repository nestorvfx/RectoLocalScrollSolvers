"""Exponential moving average of the weights (fp32 shadow copy; Polyak averaging). Evaluation and exported models use
the EMA weights."""
from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(_unwrap(model)).float().eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))                 # warm-up of the decay
        src = _unwrap(model)
        ep = list(self.shadow.parameters())
        sp = [p.detach().float() for p in src.parameters()]
        torch._foreach_lerp_(ep, sp, 1 - d)                       # one fused kernel launch per group
        for e, b in zip(self.shadow.buffers(), src.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return getattr(m, "_orig_mod", m)
