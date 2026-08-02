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
```

The setup script creates the machine-local `.venv`, selects CPython 3.12, 3.11,
or 3.13 in that order, and installs the CUDA 12 JAX plugin from the official
PyPI index before installing the project. A macOS checkout and a Linux GPU
checkout each need their own `.venv`; virtual environments are not portable
between operating systems. JAX 0.6.2 does not publish a CPython 3.14 wheel.

The `smoke` mode uses 64 visual environments and 100k steps. The `full` mode
uses the official tuned parameters: 1024 environments, 10M steps, 128
evaluation environments, and a batch size of 256.

Both modes use seed 1, write TensorBoard scalars, and extract the final mean
episode reward and `reward/success` metric to `evaluation-summary.json`.

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

## Expected artifacts

Runtime outputs are ignored by Git and stored under `reproduction/artifacts/`.
Keep at least:

- environment manifest;
- console log;
- checkpoint directory;
- rollout videos;
- fixed-seed evaluation report;
- training curves or TensorBoard logs.
