# Panda Vision PPO Reproduction

This directory reproduces the official MuJoCo Playground
`PandaPickCubeCartesian` vision PPO baseline without modifying the upstream
environment implementation.

The upstream source is pinned to:

```text
4db186a5b53427c9d313b9c7200480144894ada1
```

## Stages

1. Run the state-based JAX/CPU smoke test on macOS.
2. Run the vision/Warp smoke test on a Linux NVIDIA GPU.
3. Run the official 10M-step vision PPO configuration.
4. Evaluate fixed-seed episode success and archive the artifacts.

## macOS smoke test

Create a Python 3.12 environment and install the dependency resolution that was
validated on 2026-08-02:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  -c reproduction/constraints-2026-08-02.txt \
  -e '.[notebooks]'
MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/collect_manifest.py \
  --output reproduction/artifacts/macos-manifest.json
MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/smoke_test_macos.py \
  --output-dir reproduction/artifacts/macos-smoke
python reproduction/vision_backend_probe.py \
  --require-ready \
  --image reproduction/artifacts/macos-vision-observation.png \
  --output reproduction/artifacts/macos-vision-probe.json
```

This stage validates state and one-world RGB environment loading, JIT
reset/step, finite observations, and native MuJoCo rendering. It is not
intended to train a converged policy.

On macOS, MuJoCo rendering needs access to CoreGraphics. Run the smoke test
from a normal Terminal session if a sandboxed shell reports
`invalid CoreGraphics connection`.

The `uv.lock` file at the pinned upstream revision is stale relative to
`pyproject.toml`: it contains Warp 1.11.0 while the project requires Warp
1.15.0 or newer. It was tested separately for the CPU state smoke test, but it
is not used for the GPU visual run.

The recorded constraints combine JAX 0.6.2 with MuJoCo 3.11.0 and Warp 1.15.0.
JAX 0.11.0 can run the environment, but it removed
`jax.device_put_replicated`, which Brax 0.14.2 still uses in PPO training.

## Linux NVIDIA GPU

Follow the upstream CUDA installation instructions, then run:

```bash
./reproduction/setup_gpu.sh
./reproduction/train_panda_gpu.sh smoke
./reproduction/train_panda_gpu.sh full
# Exact upstream parallelism; intended for a high-memory GPU.
./reproduction/train_panda_gpu.sh official
```

The setup script creates the machine-local `.venv`, selects CPython 3.12, 3.11,
or 3.13 in that order, and installs the CUDA 12 JAX plugin from the official
PyPI index before installing the project. A macOS checkout and a Linux GPU
checkout each need their own `.venv`; virtual environments are not portable
between operating systems. JAX 0.6.2 does not publish a CPython 3.14 wheel.

The modes have distinct purposes:

| Mode | Timesteps | Parallelism |
| --- | ---: | --- |
| `smoke` | 100k | 64 train / 8 eval, batch 32 |
| `full` | 10M | Selected from GPU memory; see below |
| `official` | 10M | Exact upstream 1024 train / 128 eval, batch 256 |

`full` preserves the complete 10M-step workload but scales concurrency to fit
the detected GPU memory. The automatic profiles are:

| GPU memory | Train envs | Eval envs | Batch size |
| ---: | ---: | ---: | ---: |
| 20,000 MiB or more | 1024 | 128 | 256 |
| 10,000–19,999 MiB | 512 | 64 | 128 |
| 7,000–9,999 MiB | 256 | 32 | 64 |
| Less than 7,000 MiB | 128 | 16 | 32 |

An 11 GiB RTX 2080 Ti therefore uses 512/64/128. The exact 1024-env profile
was observed to exhaust that GPU after the initial evaluation while allocating
the first training batch. `official` is guarded below 20,000 MiB so this fails
early with an actionable message instead of during JIT execution.

To reduce an adaptive run further or set an explicit profile:

```bash
PANDA_FULL_NUM_ENVS=256 \
PANDA_FULL_NUM_EVAL_ENVS=32 \
PANDA_FULL_BATCH_SIZE=64 \
  ./reproduction/train_panda_gpu.sh full
```

The script also uses CUDA's asynchronous allocator to reduce fragmentation and
scales the MJWarp contact capacity with the selected environment count.

Both modes use seed 1, write TensorBoard scalars, and extract the final mean
episode reward and `reward/success` metric to `evaluation-summary.json`.
The console prints five named phases, a compact hardware/training plan, reward
and ETA at evaluation boundaries, plus a heartbeat every 30 seconds during
long JIT or training intervals. Complete configuration dumps and cached Warp
module messages are hidden in this launcher so the important state remains
visible. Training output is unbuffered and preserved in `console.log`.

Rollout videos use a task-oriented oblique free camera instead of the model's
default view or the tightly cropped `front` policy camera. The selected view
keeps the arm, gripper, cube, and lift region visible without adding another
model camera and therefore does not change the visual policy observation.

If a previous setup attempt left an incompatible `.venv`, deactivate and
remove that failed environment before running the setup script again. The
script never deletes an existing environment automatically. To select an
interpreter or a non-default environment explicitly:

```bash
PYTHON_BIN=python3.12 VENV_DIR="$PWD/.venv" \
  ./reproduction/setup_gpu.sh
```

Do not enable the generic `--domain_randomization` flag for this environment.
The visual randomizer exists in `randomize_vision.py` but is not registered in
the generic manipulation randomizer registry at the pinned revision.

## Output locations

Runtime outputs are not committed to Git. The smoke command writes them under:

```text
reproduction/artifacts/panda-vision-smoke/
├── console.log                 # Complete training console output
├── manifest.json               # Git, Python, package, and GPU environment
├── evaluation-summary.json     # Final reward and success metrics
└── runs/
    └── PandaPickCubeCartesian-<timestamp>-vision-smoke-seed1/
        ├── checkpoints/        # Restorable policy checkpoints
        ├── rollout0.mp4        # Replay videos (normally rollout0-3)
        └── events.out.tfevents.* # TensorBoard event data
```

The adaptive full command uses the same layout under
`reproduction/artifacts/panda-vision-full/`; the exact profile uses
`reproduction/artifacts/panda-vision-official/`. At the end of any command,
the script prints the exact paths to the log, manifest, evaluation summary,
checkpoint directories, TensorBoard root, and every replay video.

To inspect the main smoke outputs:

```bash
cat reproduction/artifacts/panda-vision-smoke/evaluation-summary.json
ls -lh reproduction/artifacts/panda-vision-smoke/runs/*/rollout*.mp4
tensorboard --logdir reproduction/artifacts/panda-vision-smoke/runs
```

Keep at least:

- environment manifest;
- console log;
- checkpoint directory;
- rollout videos;
- fixed-seed evaluation report;
- training curves or TensorBoard logs.
