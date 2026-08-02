#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PACKAGE_INDEX_URL="${PACKAGE_INDEX_URL:-https://pypi.org/simple}"

cd "$PROJECT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "GPU setup requires Linux with an NVIDIA CUDA driver."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Attach an NVIDIA GPU and install its driver."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in python3.12 python3.11 python3.13; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
  fi
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "CPython 3.11, 3.12, or 3.13 is required for jaxlib 0.6.2."
    echo "Install Python 3.12 or set PYTHON_BIN to a compatible interpreter."
    exit 1
  fi
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN"
    exit 1
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_TRAIN="$VENV_DIR/bin/train-jax-ppo"
if ! "$VENV_PYTHON" -c \
  'import platform, sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), sys.version; print(f"Python {platform.python_version()} on {platform.platform()}")'; then
  echo "The environment at $VENV_DIR uses an unsupported Python version."
  echo "Recreate it with Python 3.11-3.13, or choose a new VENV_DIR."
  exit 1
fi

echo "Installing packages from $PACKAGE_INDEX_URL"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "Warning: LD_LIBRARY_PATH is set and may override pip-installed CUDA libraries."
fi

"$VENV_PYTHON" -m pip install \
  --index-url "$PACKAGE_INDEX_URL" \
  --only-binary=jaxlib \
  -c reproduction/constraints-2026-08-02.txt \
  'jax[cuda12]==0.6.2'
"$VENV_PYTHON" -m pip install \
  --index-url "$PACKAGE_INDEX_URL" \
  -c reproduction/constraints-2026-08-02.txt \
  -e '.[notebooks]'
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -c \
  'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'
"$VENV_PYTHON" reproduction/collect_manifest.py \
  --output reproduction/artifacts/gpu-setup-manifest.json
"$VENV_PYTHON" reproduction/vision_backend_probe.py \
  --require-gpu \
  --image reproduction/artifacts/gpu-vision-observation.png \
  --output reproduction/artifacts/gpu-vision-probe.json

echo "GPU environment and one-world RGB reset passed."
echo "Training command: $VENV_TRAIN"
