"""Checkpoints with exact resume: model, EMA, optimiser, scheduler, step, all RNG states (torch CPU / CUDA, the
sampler generators, the pool's numpy generator), and the config. Rolling `keep` newest + `best.pt` + `last.pt`."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def rotate(run_dir: Path, keep: int) -> None:
    ck = sorted(run_dir.glob("ckpt_*.pt"))
    for p in ck[:-keep] if keep > 0 else []:
        p.unlink()


def latest(run_dir: Path) -> Path | None:
    p = run_dir / "last.pt"
    if p.exists():
        return p
    ck = sorted(run_dir.glob("ckpt_*.pt"))
    return ck[-1] if ck else None


def rng_state(sampler, pool) -> dict:
    st = {"torch": torch.get_rng_state(), "sampler_gen": sampler.gen.get_state(),
          "sampler_cpu": sampler.gen_cpu.get_state(), "pool": pool.rng.bit_generator.state}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict, sampler, pool) -> None:
    """Generator states are CPU ByteTensors; a checkpoint loaded with map_location='cuda' brings them back on the
    GPU, which set_state rejects: move them back first."""
    torch.set_rng_state(st["torch"].cpu())
    sampler.gen.set_state(st["sampler_gen"].cpu())
    sampler.gen_cpu.set_state(st["sampler_cpu"].cpu())
    pool.rng.bit_generator.state = st["pool"]
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
