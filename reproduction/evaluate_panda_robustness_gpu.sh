#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"
ROBUSTNESS_DIR="${PANDA_ROBUSTNESS_SOURCE_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-robustness}"
BASELINE="$PROJECT_DIR/reproduction/results/linux-position-stratified-analysis.json"

cd "$PROJECT_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "GPU environment not found at $VENV_DIR."
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$PROJECT_DIR/reproduction/artifacts/panda-robustness-eval/$STAMP"
mkdir -p "$RUN_DIR"
export PANDA_EVAL_ARTIFACT_DIR="$RUN_DIR"

echo "[1/5] Selecting the robustness checkpoint"
if [[ -n "${PANDA_ROBUSTNESS_EVAL_CHECKPOINT:-}" ]]; then
  CHECKPOINT="$PANDA_ROBUSTNESS_EVAL_CHECKPOINT"
else
  CHECKPOINT="$("$PYTHON" reproduction/select_best_checkpoint.py \
    --summary "$ROBUSTNESS_DIR/evaluation-summary.json" \
    --runs-dir "$ROBUSTNESS_DIR/runs" \
    --output "$RUN_DIR/checkpoint-selection.json" \
    --path-only)"
fi
printf "  %-22s%s\n" "Checkpoint" "$CHECKPOINT"
printf "  %-22s%s\n" "Output" "$RUN_DIR"

echo
echo "[2/5] Original-distribution regression"
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/original.json" \
  --target-success 0.95 \
  --minimum-seed-success 0.90

echo
echo "[3/5] Left-side regression: y in [-0.05, -0.02)"
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/left.json" \
  --box-y-min=-0.05 \
  --box-y-max=-0.02 \
  --target-success 0.95 \
  --minimum-seed-success 0.90

echo
echo "[4/5] Hard-bin regression: y in [-0.03, -0.02)"
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/hard.json" \
  --box-y-min=-0.03 \
  --box-y-max=-0.02 \
  --target-success 0.93 \
  --minimum-seed-success 0.85

echo
echo "[5/5] Controlled-experiment decision"
"$PYTHON" reproduction/summarize_robustness_evaluation.py \
  --original "$RUN_DIR/original.json" \
  --left "$RUN_DIR/left.json" \
  --hard "$RUN_DIR/hard.json" \
  --baseline "$BASELINE" \
  --output "$RUN_DIR/robustness-summary.json" \
  --require-pass

echo
echo "Robustness evaluation suite complete."
printf "  %-22s%s\n" "Run directory" "$RUN_DIR"
printf "  %-22s%s\n" "Decision summary" "$RUN_DIR/robustness-summary.json"
