"""Object registry: band stores under <data>/objects/<id>/, selected by id patterns."""
from __future__ import annotations

import fnmatch
from pathlib import Path

from .store import BandStore


def object_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    base = root / "objects" if (root / "objects").is_dir() else root
    return sorted(p for p in base.iterdir() if p.is_dir() and BandStore.exists(p))


def open_stores(root: str | Path, include=("*",), exclude=(), require_done: bool = True) -> list[BandStore]:
    out = []
    for d in object_dirs(root):
        oid = d.name
        if not any(fnmatch.fnmatch(oid, p) for p in include):
            continue
        if any(fnmatch.fnmatch(oid, p) for p in exclude):
            continue
        s = BandStore(d, "r")
        if require_done and not s.done.any():
            continue
        out.append(s)
    return out
