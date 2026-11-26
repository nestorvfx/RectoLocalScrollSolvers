"""YAML configuration with dotted command-line overrides, and the data root.

    cfg = load("configs/train.yaml", ["optim.steps=2000", "data.batch=4"])

SURFNET_DATA (default ./surfnet_data) holds the downloaded meshes, the training objects (band stores) and the chunk
cache of the CT volumes. A config's `data.root: null` means this root."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]          # the repository root
CONFIGS = PKG_ROOT / "configs"


def data_root() -> Path:
    return Path(os.environ.get("SURFNET_DATA", "surfnet_data"))


def _parse_value(s: str) -> Any:
    return yaml.safe_load(s)


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    d: Any = cfg
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config (a name alone is looked up in configs/). A top-level `base: other.yaml` (relative to the
    file) is merged first."""
    path = Path(path)
    if not path.exists() and (CONFIGS / path).exists():
        path = CONFIGS / path
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load(path.parent / cfg.pop("base"))
        cfg = deep_merge(base, cfg)
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must be key=value: {o}")
        k, v = o.split("=", 1)
        set_dotted(cfg, k.strip(), _parse_value(v))
    return cfg


def dump(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
