#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"

export JAX_DEFAULT_MATMUL_PRECISION=highest
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/reproduction/artifacts/matplotlib-cache}"
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
HAS_OUTPUT=false
OUTPUT_PATH=""
for ((index = 0; index < ${#ARGS[@]}; index++)); do
  argument="${ARGS[$index]}"
  if [[ "$argument" == "--checkpoint" || "$argument" == --checkpoint=* ]]; then
    HAS_CHECKPOINT=true
  fi
  if [[ "$argument" == "--output" ]]; then
    HAS_OUTPUT=true
    if (( index + 1 >= ${#ARGS[@]} )); then
      echo "--output requires a path."
      exit 2
    fi
    OUTPUT_PATH="${ARGS[$((index + 1))]}"
  elif [[ "$argument" == --output=* ]]; then
    HAS_OUTPUT=true
    OUTPUT_PATH="${argument#--output=}"
  fi
done

EVAL_DIR="$PROJECT_DIR/reproduction/artifacts/panda-independent-eval"
mkdir -p "$EVAL_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CONSOLE_LOG="$EVAL_DIR/console-$STAMP.log"
if [[ "$HAS_OUTPUT" == false ]]; then
  OUTPUT_PATH="$EVAL_DIR/evaluation-$STAMP.json"
  ARGS+=(--output "$OUTPUT_PATH")
fi

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
  {
    echo "[selection] Automatic checkpoint selection"
    printf "  %-22s%s\n" "Checkpoint" "$LATEST_CHECKPOINT"
  } | tee -a "$CONSOLE_LOG"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
JAX_SUMMARY="$($PYTHON -c 'import jax; print(jax.default_backend(), jax.devices())')"
{
  echo
  echo "[preflight] Linux NVIDIA evaluation"
  printf "  %-22s%s\n" "GPU" "$GPU_NAME"
  printf "  %-22s%s MiB\n" "VRAM" "$GPU_MEMORY"
  printf "  %-22s%s\n" "JAX" "$JAX_SUMMARY"
  printf "  %-22s%s\n" "Console log" "$CONSOLE_LOG"
} | tee -a "$CONSOLE_LOG"

set +e
"$PYTHON" reproduction/evaluate_panda_checkpoint.py "${ARGS[@]}" \
  2>&1 | tee -a "$CONSOLE_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if (( STATUS != 0 )); then
  echo
  echo "[failed] Independent evaluation exited with status $STATUS." \
    | tee -a "$CONSOLE_LOG"
  if grep -Eqi "RESOURCE_EXHAUSTED|out of memory" "$CONSOLE_LOG"; then
    echo "[action] GPU memory exhausted; retry with --num-envs 128." \
      | tee -a "$CONSOLE_LOG"
  else
    echo "[action] Inspect the traceback above or $CONSOLE_LOG." \
      | tee -a "$CONSOLE_LOG"
  fi
  exit "$STATUS"
fi

echo | tee -a "$CONSOLE_LOG"
echo "[analysis] Position-stratified failure analysis" | tee -a "$CONSOLE_LOG"
ANALYSIS_DIR="${OUTPUT_PATH%.*}-analysis"
set +e
"$PYTHON" reproduction/analyze_panda_evaluation.py \
  --report "$OUTPUT_PATH" \
  --output-dir "$ANALYSIS_DIR" \
  2>&1 | tee -a "$CONSOLE_LOG"
ANALYSIS_STATUS=${PIPESTATUS[0]}
set -e
if (( ANALYSIS_STATUS != 0 )); then
  echo "[failed] Evaluation passed, but analysis exited with status $ANALYSIS_STATUS." \
    | tee -a "$CONSOLE_LOG"
  echo "[action] The raw evaluation report is intact at $OUTPUT_PATH." \
    | tee -a "$CONSOLE_LOG"
  exit "$ANALYSIS_STATUS"
fi

{
  echo
  echo "Evaluation command complete."
  printf "  %-22s%s\n" "Evaluation report" "$OUTPUT_PATH"
  printf "  %-22s%s\n" "Analysis directory" "$ANALYSIS_DIR"
  printf "  %-22s%s\n" "Console log" "$CONSOLE_LOG"
} | tee -a "$CONSOLE_LOG"
