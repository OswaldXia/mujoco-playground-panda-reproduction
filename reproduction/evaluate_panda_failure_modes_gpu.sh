#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"
ROBUSTNESS_DIR="${PANDA_ROBUSTNESS_SOURCE_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-robustness}"

cd "$PROJECT_DIR"
if [[ ! -x "$PYTHON" ]]; then
  echo "GPU environment not found at $VENV_DIR."
  echo "Run ./reproduction/setup_gpu.sh first."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Run this command on the Linux NVIDIA server."
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${PANDA_FAILURE_MODE_ARTIFACT_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-failure-modes/$STAMP}"
mkdir -p "$RUN_DIR"

echo "[1/3] Selecting the completed robustness checkpoint"
if [[ -n "${PANDA_FAILURE_MODE_CHECKPOINT:-}" ]]; then
  CHECKPOINT="$PANDA_FAILURE_MODE_CHECKPOINT"
  printf "  %-22s%s\n" "Selection" "explicit environment override"
else
  CHECKPOINT="$($PYTHON reproduction/select_best_checkpoint.py \
    --summary "$ROBUSTNESS_DIR/evaluation-summary.json" \
    --runs-dir "$ROBUSTNESS_DIR/runs" \
    --output "$RUN_DIR/checkpoint-selection.json" \
    --path-only)"
  printf "  %-22s%s\n" "Selection" "best completed robustness evaluation"
fi
printf "  %-22s%s\n" "Checkpoint" "$CHECKPOINT"
printf "  %-22s%s\n" "Output" "$RUN_DIR"

echo
echo "[2/3] Collecting trajectory diagnostics on the missed left-side gate"
REPORT="$RUN_DIR/left-trajectory.json"
PANDA_EVAL_ARTIFACT_DIR="$RUN_DIR" \
  ./reproduction/evaluate_panda_gpu.sh \
  --checkpoint "$CHECKPOINT" \
  --output "$REPORT" \
  --box-y-min=-0.05 \
  --box-y-max=-0.02 \
  --target-success 0.95 \
  --minimum-seed-success 0.90

ANALYSIS_DIR="${REPORT%.*}-analysis"
CLASSIFICATION="$ANALYSIS_DIR/failure-classification.json"
if [[ ! -s "$CLASSIFICATION" ]]; then
  echo "Trajectory classification was not generated: $CLASSIFICATION"
  exit 1
fi

echo
echo "[3/3] Failure-mode collection complete"
printf "  %-22s%s\n" "Evaluation report" "$REPORT"
printf "  %-22s%s\n" "Classification" "$CLASSIFICATION"
printf "  %-22s%s\n" "Episode table" "$ANALYSIS_DIR/episodes.csv"
printf "  %-22s%s\n" "Console log" "$RUN_DIR/console-*.log"
echo
echo "Use the class counts to select one intervention; do not retrain yet."
