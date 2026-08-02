# Reproduction Status

## Pinned baseline

- Environment: `PandaPickCubeCartesian`
- Upstream commit: `4db186a5b53427c9d313b9c7200480144894ada1`
- Branch: `reproduce/panda-vision`
- Official vision steps: `10,000,000`
- Official parallel environments: `1,024`
- Official evaluation environments: `128`
- Image observation: single-camera `64 x 64 RGB`

## Checklist

- [x] Upstream source pinned
- [x] Reproduction harness added
- [x] macOS environment installed with recorded constraints
- [x] macOS state reset/step/render passed
- [x] macOS MJWarp vision capability boundary recorded
- [ ] Linux NVIDIA vision smoke test passed
- [ ] Official 10M-step vision training completed
- [ ] Fixed-seed success evaluation completed
- [ ] Results documented

## macOS state-smoke result (2026-08-02)

- Hardware: MacBook Air (M1, 8 cores, 16 GB)
- OS/Python: macOS 15.7.5 arm64 / Python 3.12.4
- JAX backend: CPU (`TFRT_CPU_0`)
- Primary packages: JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.11.0, Brax 0.14.2,
  Warp 1.15.0, Playground 0.2.0
- MuJoCo Menagerie: `1b86ece576591213e2b666ebf59508454200ca97`
- Test: JIT reset, 2 zero-action steps, finite observation/state assertions,
  and 640 x 480 native rendering
- Result: passed; reward `0.3532326221`, done `0.0`
- First-run timing: reset `0.190 s`; steps `21.449 s` and `21.192 s`
- Evidence: `reproduction/results/macos-state-smoke.json`; runtime image and
  manifest are in the ignored `reproduction/artifacts/` directory
- Training-stack self-test: a 128-timestep update on Brax `fast` passed,
  confirming the constrained JAX/Brax PPO path executes on CPU

The stale upstream `uv.lock` profile (JAX 0.6.2, MuJoCo 3.6.0, Warp 1.11.0)
was also exercised and passed the same assertions. That secondary run emitted
an `overflow encountered in cast` warning during compilation; the explicit
finiteness checks still passed.

## Known hardware boundary

Apple Silicon is used for source inspection, state-environment debugging, and
native rendering. The official visual training path uses the MJWarp batch
renderer and requires an NVIDIA GPU for practical throughput.

With the current declared dependencies (MuJoCo 3.11.0 and Warp 1.15.0), a
one-world 64 x 64 RGB reset and step succeed on the M1 CPU. With cached Warp
kernels, reset took `1.546 s` and one step took `2.025 s`; the initial cold
reset took `19.134 s`. This is useful for visual pipeline debugging, but JAX
still exposes only the CPU backend; the official 1,024-environment PPO run
remains an NVIDIA GPU task. The structured result is in
`reproduction/results/macos-vision-probe.json`.
