"""Logging: a console + file logger and a JSON-lines metrics writer."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


def get_logger(name: str = "surfnet", file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_surfnet", False) for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s %(name)s] %(message)s", "%H:%M:%S"))
        h._surfnet = True  # type: ignore[attr-defined]
        log.addHandler(h)
    if file is not None:
        file = str(file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(file).resolve()) for h in log.handlers):
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(file)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
            log.addHandler(fh)
    log.propagate = False
    return log


class JsonlWriter:
    """Append-only metrics log: one JSON object per line, with a wall-clock time stamp."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **rec) -> None:
        rec.setdefault("time", round(time.time(), 3))
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
