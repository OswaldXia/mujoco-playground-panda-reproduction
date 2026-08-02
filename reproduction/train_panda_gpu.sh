#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-}"
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "full" ]]; then
  echo "Usage: $0 {smoke|full}"
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$PROJECT_DIR/reproduction/artifacts/panda-vision-$RUN_KIND"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"
TRAIN="$VENV_DIR/bin/train-jax-ppo"

export JAX_DEFAULT_MATMUL_PRECISION=highest
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
mkdir -p "$ARTIFACT_DIR"

if [[ ! -x "$PYTHON" || ! -x "$TRAIN" ]]; then
  echo "GPU environment not found at $VENV_DIR."
  echo "Run ./reproduction/setup_gpu.sh first."
  exit 1
fi

echo "[1/4] Validating the JAX GPU backend..."
"$PYTHON" -c \
  'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'

echo "[2/4] Recording the reproducibility manifest..."
"$PYTHON" reproduction/collect_manifest.py \
  --output "$ARTIFACT_DIR/manifest.json"

COMMON_ARGS=(
  --env_name=PandaPickCubeCartesian
  --impl=warp
  --vision
  --seed=1
  --num_videos=4
  --reward_scaling=1.0
  --progress_interval_seconds=30
  --render_camera_lookat=0.45,0.0,0.20
  --render_camera_distance=1.20
  --render_camera_azimuth=140
  --render_camera_elevation=-25
  --logdir="$ARTIFACT_DIR/runs"
  --suffix="vision-$RUN_KIND-seed1"
  --use_tb
)

if [[ "$RUN_KIND" == "smoke" ]]; then
  echo "[3/4] Starting the 100k-step visual PPO smoke run..."
  "$TRAIN" \
    "${COMMON_ARGS[@]}" \
    --num_timesteps=100000 \
    --num_envs=64 \
    --num_eval_envs=8 \
    --batch_size=32 \
    --num_evals=3 \
    --playground_config_overrides='{"naconmax":1536,"naccdmax":1536}' \
    2>&1 | tee "$ARTIFACT_DIR/console.log"
else
  echo "[3/4] Starting the 10M-step visual PPO full run..."
  "$TRAIN" \
    "${COMMON_ARGS[@]}" \
    --num_timesteps=10000000 \
    --num_envs=1024 \
    --num_eval_envs=128 \
    --batch_size=256 \
    --num_evals=5 \
    2>&1 | tee "$ARTIFACT_DIR/console.log"
fi

echo "[4/4] Summarizing TensorBoard evaluation metrics..."
"$PYTHON" reproduction/summarize_tensorboard.py \
  --logdir "$ARTIFACT_DIR/runs" \
  --output "$ARTIFACT_DIR/evaluation-summary.json"

echo "Run complete. Artifacts: $ARTIFACT_DIR"
