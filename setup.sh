#!/usr/bin/env bash
# Environment for surfnet on Linux x86_64 with an NVIDIA GPU (safe to re-run; no tokens needed: the open-data
# bucket is public).
#   bash setup.sh [--no-gpu] [--test]
# 1. s5cmd (batched S3 transfers) and uv; 2. .venv: Python 3.13, torch 2.10.0 (CUDA 12.8 wheels, 12.6 for drivers
# older than 570), requirements.txt, CuPy and cuDSS (nvmath-python; with a GPU), this package; 3. with --test, the CPU
# tests (about 15 min).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=1; TEST=0
for a in "$@"; do [[ "$a" == "--no-gpu" ]] && GPU=0; [[ "$a" == "--test" ]] && TEST=1; done
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin"
command -v s5cmd >/dev/null || curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
  | tar xz -C "$HOME/.local/bin" s5cmd
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/0.12.10/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
INDEX=https://download.pytorch.org/whl/cu128
if (( GPU )); then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  (( ${DRIVER%%.*} < 570 )) && INDEX=https://download.pytorch.org/whl/cu126
else
  INDEX=https://download.pytorch.org/whl/cpu
fi
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv --python 3.13 "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" --index-url "$INDEX" torch==2.10.0
uv pip install --python "$PY" -r "$ROOT/requirements.txt"
if (( GPU )); then
  uv pip install --python "$PY" cupy-cuda12x==14.2.0               # the recto local solver's CUDA kernels (NVRTC at run time)
  uv pip freeze --python "$PY" | grep -E '^(nvidia-|torch==)' > "$ROOT/.venv/pins.txt"   # (torch's CUDA libraries kept)
  uv pip install --python "$PY" -c "$ROOT/.venv/pins.txt" "nvmath-python[cu12]==1.0.1" \
    || echo "setup: nvmath-python (cuDSS) not installed: the recto local solver's continuous solve runs on the CPU"
fi
uv pip install --python "$PY" --no-deps -e "$ROOT"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
(( TEST )) && (cd "$ROOT" && "$PY" -m pytest -q tests)
echo "done: source $ROOT/.venv/bin/activate"
