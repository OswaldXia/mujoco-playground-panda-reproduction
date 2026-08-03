#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PANDA_ARTIFACT_DIR="${PANDA_ARTIFACT_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-robustness-smoke}"
export PANDA_ROBUSTNESS_TIMESTEPS="${PANDA_ROBUSTNESS_TIMESTEPS:-100000}"
export PANDA_ROBUSTNESS_NUM_EVALS="${PANDA_ROBUSTNESS_NUM_EVALS:-3}"

echo "[smoke] Targeted robustness training pipeline"
printf "  %-22s%s\n" "Timesteps" "$PANDA_ROBUSTNESS_TIMESTEPS"
printf "  %-22s%s\n" "Artifact root" "$PANDA_ARTIFACT_DIR"

exec "$PROJECT_DIR/reproduction/train_panda_gpu.sh" robustness
