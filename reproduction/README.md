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

Create a Python 3.12 environment and install the exact versions from the
upstream lock file:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install uv==0.12.1
uv sync --frozen --active --extra notebooks
MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/collect_manifest.py \
  --output reproduction/artifacts/macos-manifest.json
MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/smoke_test_macos.py \
  --output-dir reproduction/artifacts/macos-smoke
```

This stage validates environment loading, JIT reset/step, finite observations,
and native MuJoCo rendering. It is not intended to train a converged policy.

On macOS, MuJoCo rendering needs access to CoreGraphics. Run the smoke test
from a normal Terminal session if a sandboxed shell reports
`invalid CoreGraphics connection`.

## Linux NVIDIA GPU

Follow the upstream CUDA installation instructions, then run:

```bash
./reproduction/train_panda_gpu.sh smoke
./reproduction/train_panda_gpu.sh full
```

The `smoke` mode uses 64 visual environments and 100k steps. The `full` mode
uses the official tuned parameters: 1024 environments, 10M steps, 128
evaluation environments, and a batch size of 256.

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
