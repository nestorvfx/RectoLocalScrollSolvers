"""s5cmd wrapper for the public open-data bucket (no credentials): listings, small reads, and verified prefix copies
(list -> exact keys -> `s5cmd run` -> size check against the listing -> atomic rename). Never one HTTP request per
chunk (memory: s5cmd-for-batch-downloads)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

BUCKET = "s3://vesuvius-challenge-open-data"


class S3:
    def __init__(self, bucket: str = BUCKET, s5cmd: str = "s5cmd", workers: int = 64):
        self.bucket = bucket.rstrip("/")
        self.s5 = s5cmd
        self.workers = workers

    def _run(self, *args, check=True) -> subprocess.CompletedProcess:
        return subprocess.run([self.s5, "--no-sign-request", *args], capture_output=True, text=True, check=check)

    def url(self, key: str) -> str:
        return f"{self.bucket}/{key.lstrip('/')}"

    def ls(self, prefix: str, recursive: bool = False) -> list[dict]:
        """One level under prefix ('prefix/': objects + 'dir' entries), or every object below it (recursive:
        s5cmd's wildcard 'prefix/*' matches across '/'). Keys are relative to the bucket."""
        url = self.url(prefix.rstrip("/") + ("/*" if recursive else "/"))
        r = self._run("--json", "ls", url, check=False)
        out = []
        for line in r.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = d.get("key", "")
            k = k[len(self.bucket) + 1:] if k.startswith(self.bucket) else k
            if d.get("type") == "directory":
                if not recursive:
                    out.append({"key": k.rstrip("/") + "/", "type": "dir", "size": 0})
            else:
                out.append({"key": k, "type": "file", "size": int(d.get("size", 0))})
        return out

    def dirs(self, prefix: str) -> list[str]:
        return [o["key"] for o in self.ls(prefix) if o["type"] == "dir"]

    def files_recursive(self, prefix: str) -> list[dict]:
        return [{"key": o["key"], "size": o["size"]} for o in self.ls(prefix, recursive=True)]

    def cat(self, key: str) -> bytes:
        return subprocess.run([self.s5, "--no-sign-request", "cat", self.url(key)], capture_output=True, check=True).stdout

    def get_prefix(self, prefix: str, dest: str | Path) -> int:
        """Mirror every object under prefix into dest (same relative layout). Returns bytes downloaded."""
        dest = Path(dest)
        files = self.files_recursive(prefix)
        todo = []
        for f in files:
            rel = f["key"][len(prefix.rstrip("/")) + 1:]
            p = dest / rel
            if p.exists() and p.stat().st_size == f["size"]:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            todo.append((f, p))
        if not todo:
            return 0
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(f'cp "{self.url(f["key"])}" "{p}.part"' for f, p in todo))
            cmds = fh.name
        subprocess.run([self.s5, "--no-sign-request", "--numworkers", str(self.workers), "run", cmds],
                       capture_output=True, text=True)
        os.unlink(cmds)
        bad, n = [], 0
        for f, p in todo:
            part = Path(f"{p}.part")
            if part.exists() and part.stat().st_size == f["size"]:
                os.replace(part, p)
                n += f["size"]
            else:
                bad.append(f["key"])
        if bad:
            raise RuntimeError(f"{len(bad)} objects under {prefix} failed the size check (first {bad[:2]})")
        return n
