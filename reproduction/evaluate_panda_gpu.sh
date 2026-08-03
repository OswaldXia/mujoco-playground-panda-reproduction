#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"

export JAX_DEFAULT_MATMUL_PRECISION=highest
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -x "$PYTHON" ]]; then
  echo "GPU environment not found at $VENV_DIR."
  echo "Run ./reproduction/setup_gpu.sh first."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Run this command on the Linux NVIDIA server."
  exit 1
fi

cd "$PROJECT_DIR"

ARGS=("$@")
HAS_CHECKPOINT=false
for argument in "${ARGS[@]}"; do
  if [[ "$argument" == "--checkpoint" || "$argument" == --checkpoint=* ]]; then
    HAS_CHECKPOINT=true
    break
  fi
done

if [[ "$HAS_CHECKPOINT" == false ]]; then
  ARTIFACT_DIR="${PANDA_EVAL_SOURCE_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-finetune}"
  shopt -s nullglob
  RUN_DIRS=("$ARTIFACT_DIR"/runs/PandaPickCubeCartesian-*)
  if (( ${#RUN_DIRS[@]} == 0 )); then
    echo "No fine-tune run found under $ARTIFACT_DIR/runs."
    echo "Pass --checkpoint /exact/checkpoint/path explicitly."
    exit 1
  fi
  LATEST_RUN="${RUN_DIRS[0]}"
  for run_dir in "${RUN_DIRS[@]}"; do
    if [[ "$run_dir" -nt "$LATEST_RUN" ]]; then
      LATEST_RUN="$run_dir"
    fi
  done
  CHECKPOINTS=("$LATEST_RUN"/checkpoints/[0-9]*)
  if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "No checkpoint found under $LATEST_RUN/checkpoints."
    exit 1
  fi
  LATEST_CHECKPOINT="${CHECKPOINTS[0]}"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ "$(basename "$checkpoint")" > "$(basename "$LATEST_CHECKPOINT")" ]]; then
      LATEST_CHECKPOINT="$checkpoint"
    fi
  done
  ARGS+=(--checkpoint "$LATEST_CHECKPOINT")
  echo "[selection] No checkpoint argument supplied; using the latest fine-tune checkpoint:"
  echo "  $LATEST_CHECKPOINT"
  echo
fi

echo "[preflight] Independent Panda checkpoint evaluation"
printf "  %-20s %s\n" "GPU" "$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
printf "  %-20s %s MiB\n" "VRAM" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
printf "  %-20s %s\n" "JAX" "$($PYTHON -c 'import jax; print(jax.default_backend(), jax.devices())')"
echo

set +e
"$PYTHON" reproduction/evaluate_panda_checkpoint.py "${ARGS[@]}"
STATUS=$?
set -e

if (( STATUS != 0 )); then
  echo
  echo "[failed] Independent evaluation exited with status $STATUS."
  echo "If the error is GPU out-of-memory, retry with --num-envs 128."
  exit "$STATUS"
fi
