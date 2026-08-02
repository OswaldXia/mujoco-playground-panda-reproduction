#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "GPU setup requires Linux with an NVIDIA CUDA driver."
  exit 1
fi

"$PYTHON_BIN" -c \
  'import sys; assert sys.version_info >= (3, 11), sys.version'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install \
  -c reproduction/constraints-2026-08-02.txt \
  -e '.[cuda,notebooks]'
.venv/bin/python reproduction/collect_manifest.py \
  --output reproduction/artifacts/gpu-setup-manifest.json
.venv/bin/python reproduction/vision_backend_probe.py \
  --require-gpu \
  --image reproduction/artifacts/gpu-vision-observation.png \
  --output reproduction/artifacts/gpu-vision-probe.json

echo "GPU environment and one-world RGB reset passed."
