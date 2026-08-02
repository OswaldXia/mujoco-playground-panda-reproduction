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
- [x] macOS environment installed from `uv.lock`
- [x] macOS state reset/step/render passed
- [ ] Linux NVIDIA vision smoke test passed
- [ ] Official 10M-step vision training completed
- [ ] Fixed-seed success evaluation completed
- [ ] Results documented

## macOS state-smoke result (2026-08-02)

- Hardware: MacBook Air (M1, 8 cores, 16 GB)
- OS/Python: macOS 15.7.5 arm64 / Python 3.12.4
- JAX backend: CPU (`TFRT_CPU_0`)
- Locked packages: JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.6.0, Brax 0.14.2,
  Warp 1.11.0, Playground 0.2.0
- MuJoCo Menagerie: `1b86ece576591213e2b666ebf59508454200ca97`
- Test: JIT reset, 2 zero-action steps, finite observation/state assertions,
  and 640 x 480 native rendering
- Result: passed; reward `0.3532326221`, done `0.0`
- First-run timing: reset `0.199 s`; steps `22.469 s` and `22.460 s`
- Evidence: `reproduction/results/macos-state-smoke.json`; runtime image and
  manifest are in the ignored `reproduction/artifacts/` directory

JAX emitted an `overflow encountered in cast` warning during compilation, but
the explicit observation, reward, position, and velocity finiteness checks all
passed.

## Known hardware boundary

Apple Silicon is used for source inspection, state-environment debugging, and
native rendering. The official visual training path uses the MJWarp batch
renderer and requires an NVIDIA GPU for practical throughput.
