#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-}"
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "full" ]]; then
  echo "Usage: $0 {smoke|full}"
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$PROJECT_DIR/reproduction/artifacts/panda-vision-$RUN_KIND"

export JAX_DEFAULT_MATMUL_PRECISION=highest
export MUJOCO_GL=egl

cd "$PROJECT_DIR"
mkdir -p "$ARTIFACT_DIR"

uv run python -c \
  'import jax; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices())'

uv run python reproduction/collect_manifest.py \
  --output "$ARTIFACT_DIR/manifest.json"

COMMON_ARGS=(
  --env_name=PandaPickCubeCartesian
  --impl=warp
  --vision
  --seed=1
  --num_videos=4
  --reward_scaling=1.0
  --logdir="$ARTIFACT_DIR/runs"
  --suffix="vision-$RUN_KIND-seed1"
)

if [[ "$RUN_KIND" == "smoke" ]]; then
  uv run train-jax-ppo \
    "${COMMON_ARGS[@]}" \
    --num_timesteps=100000 \
    --num_envs=64 \
    --num_eval_envs=8 \
    --batch_size=32 \
    --num_evals=2 \
    --playground_config_overrides='{"naconmax":1536,"naccdmax":1536}' \
    2>&1 | tee "$ARTIFACT_DIR/console.log"
else
  uv run train-jax-ppo \
    "${COMMON_ARGS[@]}" \
    --num_timesteps=10000000 \
    --num_envs=1024 \
    --num_eval_envs=128 \
    --batch_size=256 \
    --num_evals=5 \
    --use_tb \
    2>&1 | tee "$ARTIFACT_DIR/console.log"
fi
