"""Learning-rate schedule."""
from __future__ import annotations


def wsd_lambda(step: int, warmup: int, cool_start: int | None, cool_len: int, min_ratio: float):
    """Warmup-stable-decay: linear warm-up, constant, then a linear cooldown to min_ratio once `cool_start` is set
    (at the validation plateau or near the step cap). The run length need not be known in advance (Hu et al. 2024,
    MiniCPM; Hägele et al. 2024, "Scaling laws and compute-optimal training beyond fixed training durations")."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if cool_start is None or step < cool_start:
        return 1.0
    t = min(1.0, (step - cool_start) / max(1, cool_len))
    return 1.0 - (1.0 - min_ratio) * t
