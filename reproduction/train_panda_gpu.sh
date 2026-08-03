#!/usr/bin/env bash
set -euo pipefail

RUN_KIND="${1:-}"
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "full" && "$RUN_KIND" != "official" && "$RUN_KIND" != "finetune" ]]; then
  echo "Usage: $0 {smoke|full|official|finetune}"
  echo "  smoke:    100k-step pipeline validation"
  echo "  full:     10M steps with GPU-memory-aware parallelism"
  echo "  official: exact upstream 1024-env configuration (high-memory GPU)"
  echo "  finetune: continue from the best full-run checkpoint at a lower learning rate"
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
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

stage() {
  echo
  echo "[$1/5] $2"
}

item() {
  printf "  %-20s %s\n" "$1" "$2"
}

cd "$PROJECT_DIR"
mkdir -p "$ARTIFACT_DIR"

if [[ ! -x "$PYTHON" || ! -x "$TRAIN" ]]; then
  echo "GPU environment not found at $VENV_DIR."
  echo "Run ./reproduction/setup_gpu.sh first."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. A Linux NVIDIA GPU is required."
  exit 1
fi

PANDA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
PANDA_GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
PANDA_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
if ! [[ "$PANDA_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "Could not determine GPU memory from nvidia-smi: $PANDA_GPU_MEMORY_MIB"
  exit 1
fi

stage 1 "Hardware preflight"
item "GPU" "$PANDA_GPU_NAME"
item "VRAM" "$PANDA_GPU_MEMORY_MIB MiB"
item "Driver" "$PANDA_GPU_DRIVER"
PANDA_JAX_SUMMARY="$($PYTHON -c 'import jax; devices = jax.devices(); assert jax.default_backend() == "gpu", devices; print("{} ({})".format(jax.default_backend(), ", ".join(map(str, devices))))')"
item "JAX backend" "$PANDA_JAX_SUMMARY"
item "Memory allocator" "$TF_GPU_ALLOCATOR"

stage 2 "Reproducibility record"
"$PYTHON" reproduction/collect_manifest.py \
  --output "$ARTIFACT_DIR/manifest.json" >/dev/null
item "Git commit" "$(git rev-parse --short HEAD)"
item "Manifest" "$ARTIFACT_DIR/manifest.json"

PANDA_NUM_TIMESTEPS=10000000
PANDA_NUM_EVALS=5
PANDA_CONFIG_OVERRIDES=""
PANDA_LEARNING_RATE=""
PANDA_RESTORE_CHECKPOINT=""

if [[ "$RUN_KIND" == "smoke" ]]; then
  PANDA_PROFILE="smoke"
  PANDA_NUM_TIMESTEPS=100000
  PANDA_NUM_ENVS=64
  PANDA_NUM_EVAL_ENVS=8
  PANDA_BATCH_SIZE=32
  PANDA_NUM_EVALS=3
  PANDA_CONTACT_CAPACITY=$((24 * PANDA_NUM_ENVS))
  PANDA_CONFIG_OVERRIDES="{\"naconmax\":$PANDA_CONTACT_CAPACITY,\"naccdmax\":$PANDA_CONTACT_CAPACITY}"
elif [[ "$RUN_KIND" == "official" ]]; then
  if (( PANDA_GPU_MEMORY_MIB < 20000 )); then
    echo
    echo "Official profile blocked before allocation: $PANDA_GPU_MEMORY_MIB MiB VRAM is below the 20,000 MiB safety threshold."
    echo "This 1024-env profile is known to exceed an 11 GiB RTX 2080 Ti."
    echo "Run './reproduction/train_panda_gpu.sh full' for the same 10M timesteps with memory-aware parallelism."
    exit 1
  fi
  PANDA_PROFILE="official-exact"
  PANDA_NUM_ENVS=1024
  PANDA_NUM_EVAL_ENVS=128
  PANDA_BATCH_SIZE=256
else
  if (( PANDA_GPU_MEMORY_MIB >= 20000 )); then
    PANDA_PROFILE="auto-large (official parallelism)"
    PANDA_NUM_ENVS=1024
    PANDA_NUM_EVAL_ENVS=128
    PANDA_BATCH_SIZE=256
  elif (( PANDA_GPU_MEMORY_MIB >= 10000 )); then
    PANDA_PROFILE="auto-10-19GiB"
    PANDA_NUM_ENVS=512
    PANDA_NUM_EVAL_ENVS=64
    PANDA_BATCH_SIZE=128
  elif (( PANDA_GPU_MEMORY_MIB >= 7000 )); then
    PANDA_PROFILE="auto-7-9GiB"
    PANDA_NUM_ENVS=256
    PANDA_NUM_EVAL_ENVS=32
    PANDA_BATCH_SIZE=64
  else
    PANDA_PROFILE="auto-under-7GiB"
    PANDA_NUM_ENVS=128
    PANDA_NUM_EVAL_ENVS=16
    PANDA_BATCH_SIZE=32
  fi

  if [[ "$RUN_KIND" == "finetune" ]]; then
    if [[ -n "${PANDA_FINETUNE_NUM_ENVS:-}${PANDA_FINETUNE_NUM_EVAL_ENVS:-}${PANDA_FINETUNE_BATCH_SIZE:-}" ]]; then
      PANDA_PROFILE="finetune-custom"
    else
      PANDA_PROFILE="finetune-$PANDA_PROFILE"
    fi
    PANDA_NUM_ENVS="${PANDA_FINETUNE_NUM_ENVS:-$PANDA_NUM_ENVS}"
    PANDA_NUM_EVAL_ENVS="${PANDA_FINETUNE_NUM_EVAL_ENVS:-$PANDA_NUM_EVAL_ENVS}"
    PANDA_BATCH_SIZE="${PANDA_FINETUNE_BATCH_SIZE:-$PANDA_BATCH_SIZE}"
    PANDA_NUM_TIMESTEPS="${PANDA_FINETUNE_TIMESTEPS:-10000000}"
    PANDA_NUM_EVALS="${PANDA_FINETUNE_NUM_EVALS:-9}"
    PANDA_LEARNING_RATE="${PANDA_FINETUNE_LEARNING_RATE:-0.0005}"
  else
    if [[ -n "${PANDA_FULL_NUM_ENVS:-}${PANDA_FULL_NUM_EVAL_ENVS:-}${PANDA_FULL_BATCH_SIZE:-}" ]]; then
      PANDA_PROFILE="custom"
    fi
    PANDA_NUM_ENVS="${PANDA_FULL_NUM_ENVS:-$PANDA_NUM_ENVS}"
    PANDA_NUM_EVAL_ENVS="${PANDA_FULL_NUM_EVAL_ENVS:-$PANDA_NUM_EVAL_ENVS}"
    PANDA_BATCH_SIZE="${PANDA_FULL_BATCH_SIZE:-$PANDA_BATCH_SIZE}"
  fi
  for value in "$PANDA_NUM_ENVS" "$PANDA_NUM_EVAL_ENVS" "$PANDA_BATCH_SIZE" "$PANDA_NUM_TIMESTEPS" "$PANDA_NUM_EVALS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "Training profile values must be positive integers: $value"
      exit 1
    fi
  done
  PANDA_CONTACT_CAPACITY=$((48 * PANDA_NUM_ENVS))
  PANDA_CONFIG_OVERRIDES="{\"naconmax\":$PANDA_CONTACT_CAPACITY,\"naccdmax\":$PANDA_CONTACT_CAPACITY}"
fi

if [[ "$RUN_KIND" == "finetune" ]]; then
  PANDA_FULL_ARTIFACT_DIR="${PANDA_FINETUNE_SOURCE_DIR:-$PROJECT_DIR/reproduction/artifacts/panda-vision-full}"
  PANDA_FINETUNE_SELECTION="$ARTIFACT_DIR/finetune-source.json"
  if [[ -n "${PANDA_FINETUNE_CHECKPOINT:-}" ]]; then
    PANDA_RESTORE_CHECKPOINT="$PANDA_FINETUNE_CHECKPOINT"
    if [[ ! -f "$PANDA_RESTORE_CHECKPOINT/ppo_network_config.json" && ! -f "$PANDA_RESTORE_CHECKPOINT/config.json" ]]; then
      echo "The requested fine-tune checkpoint is not an exact checkpoint directory:"
      echo "  $PANDA_RESTORE_CHECKPOINT"
      echo "Expected ppo_network_config.json or config.json inside that directory."
      exit 1
    fi
    "$PYTHON" -c 'import json, pathlib, sys; path = pathlib.Path(sys.argv[1]).resolve(); output = pathlib.Path(sys.argv[2]); output.write_text(json.dumps({"checkpoint": str(path), "selection": "explicit override"}, indent=2) + "\n", encoding="utf-8")' \
      "$PANDA_RESTORE_CHECKPOINT" "$PANDA_FINETUNE_SELECTION"
  else
    PANDA_FULL_SUMMARY="$PANDA_FULL_ARTIFACT_DIR/evaluation-summary.json"
    PANDA_FULL_RUNS="$PANDA_FULL_ARTIFACT_DIR/runs"
    if [[ ! -f "$PANDA_FULL_SUMMARY" || ! -d "$PANDA_FULL_RUNS" ]]; then
      echo "A completed full run was not found at:"
      echo "  $PANDA_FULL_ARTIFACT_DIR"
      echo "Run './reproduction/train_panda_gpu.sh full' first, or set PANDA_FINETUNE_CHECKPOINT."
      exit 1
    fi
    PANDA_RESTORE_CHECKPOINT="$("$PYTHON" reproduction/select_best_checkpoint.py \
      --summary "$PANDA_FULL_SUMMARY" \
      --runs-dir "$PANDA_FULL_RUNS" \
      --output "$PANDA_FINETUNE_SELECTION" \
      --path-only)"
  fi
fi

if (( (PANDA_BATCH_SIZE * 8) % PANDA_NUM_ENVS != 0 )); then
  echo "Invalid profile: batch_size * num_minibatches must be divisible by num_envs."
  echo "Received batch_size=$PANDA_BATCH_SIZE, num_minibatches=8, num_envs=$PANDA_NUM_ENVS."
  exit 1
fi

stage 3 "Training plan"
item "Mode / profile" "$RUN_KIND / $PANDA_PROFILE"
item "Total timesteps" "$PANDA_NUM_TIMESTEPS"
item "Train environments" "$PANDA_NUM_ENVS"
item "Eval environments" "$PANDA_NUM_EVAL_ENVS"
item "Batch size" "$PANDA_BATCH_SIZE"
item "Evaluations" "$PANDA_NUM_EVALS"
if [[ -n "$PANDA_LEARNING_RATE" ]]; then
  item "Learning rate" "$PANDA_LEARNING_RATE"
  item "Restore checkpoint" "$PANDA_RESTORE_CHECKPOINT"
  item "Selection record" "$PANDA_FINETUNE_SELECTION"
fi
item "Artifact root" "$ARTIFACT_DIR"
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "official" && "$PANDA_NUM_ENVS" -lt 1024 ]]; then
  echo "  Note: parallelism and batch size are reduced to fit available VRAM."
fi
if [[ "$RUN_KIND" == "finetune" ]]; then
  echo "  Note: policy/value parameters are restored; the optimizer state starts fresh."
fi

COMMON_ARGS=(
  --env_name=PandaPickCubeCartesian
  --impl=warp
  --vision
  --quiet_warp
  --noprint_config
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

TRAIN_ARGS=(
  --num_timesteps="$PANDA_NUM_TIMESTEPS"
  --num_envs="$PANDA_NUM_ENVS"
  --num_eval_envs="$PANDA_NUM_EVAL_ENVS"
  --batch_size="$PANDA_BATCH_SIZE"
  --num_evals="$PANDA_NUM_EVALS"
)
if [[ -n "$PANDA_CONFIG_OVERRIDES" ]]; then
  TRAIN_ARGS+=(--playground_config_overrides="$PANDA_CONFIG_OVERRIDES")
fi
if [[ -n "$PANDA_LEARNING_RATE" ]]; then
  TRAIN_ARGS+=(
    --learning_rate="$PANDA_LEARNING_RATE"
    --load_checkpoint_path="$PANDA_RESTORE_CHECKPOINT"
  )
fi

stage 4 "PPO training and replay generation"
echo "  The first evaluation and JIT compilation may take several minutes."
set +e
"$TRAIN" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$ARTIFACT_DIR/console.log"
PANDA_TRAIN_STATUS=${PIPESTATUS[0]}
set -e
if (( PANDA_TRAIN_STATUS != 0 )); then
  echo
  echo "[failed] Training exited with status $PANDA_TRAIN_STATUS."
  item "Full log" "$ARTIFACT_DIR/console.log"
  if grep -Eq "RESOURCE_EXHAUSTED|Out of memory|out of memory" "$ARTIFACT_DIR/console.log"; then
    echo "  Cause: GPU memory exhausted. Retry with a smaller custom profile:"
    if [[ "$RUN_KIND" == "finetune" ]]; then
      echo "  PANDA_FINETUNE_NUM_ENVS=256 PANDA_FINETUNE_NUM_EVAL_ENVS=32 PANDA_FINETUNE_BATCH_SIZE=64 ./reproduction/train_panda_gpu.sh finetune"
    else
      echo "  PANDA_FULL_NUM_ENVS=256 PANDA_FULL_NUM_EVAL_ENVS=32 PANDA_FULL_BATCH_SIZE=64 ./reproduction/train_panda_gpu.sh full"
    fi
  fi
  exit "$PANDA_TRAIN_STATUS"
fi

stage 5 "Evaluation summary and output index"
shopt -s nullglob
PANDA_RUN_DIRS=("$ARTIFACT_DIR"/runs/PandaPickCubeCartesian-*)
if (( ${#PANDA_RUN_DIRS[@]} == 0 )); then
  echo "No timestamped training run was found under $ARTIFACT_DIR/runs."
  exit 1
fi
PANDA_CURRENT_RUN_DIR="${PANDA_RUN_DIRS[0]}"
for run_dir in "${PANDA_RUN_DIRS[@]}"; do
  if [[ "$run_dir" -nt "$PANDA_CURRENT_RUN_DIR" ]]; then
    PANDA_CURRENT_RUN_DIR="$run_dir"
  fi
done

"$PYTHON" reproduction/summarize_tensorboard.py \
  --logdir "$PANDA_CURRENT_RUN_DIR" \
  --output "$ARTIFACT_DIR/evaluation-summary.json" \
  --concise

VIDEOS=("$PANDA_CURRENT_RUN_DIR"/rollout*.mp4)

echo
echo "Run complete. Output locations:"
item "Artifact root" "$ARTIFACT_DIR"
item "Console log" "$ARTIFACT_DIR/console.log"
item "Environment" "$ARTIFACT_DIR/manifest.json"
if [[ "$RUN_KIND" == "finetune" ]]; then
  item "Fine-tune source" "$PANDA_FINETUNE_SELECTION"
fi
item "Evaluation summary" "$ARTIFACT_DIR/evaluation-summary.json"
item "TensorBoard root" "$ARTIFACT_DIR/runs"
item "Current run" "$PANDA_CURRENT_RUN_DIR"
item "Checkpoints" "$PANDA_CURRENT_RUN_DIR/checkpoints"
if (( ${#VIDEOS[@]} > 0 )); then
  echo "  Replay videos:"
  for video in "${VIDEOS[@]}"; do
    echo "    $video"
  done
else
  item "Replay videos" "none found"
fi
