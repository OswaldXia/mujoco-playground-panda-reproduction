# Reproduction Status

## Pinned baseline

- Environment: `PandaPickCubeCartesian`
- Upstream commit: `4db186a5b53427c9d313b9c7200480144894ada1`
- Project branch: `main`
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
- [x] Linux NVIDIA vision smoke test passed
- [x] Adaptive 10M-step vision training completed
- [x] Fixed-seed success evaluation completed
- [x] Initial full-run results documented
- [x] Best-checkpoint fine-tuning completed
- [x] Training evaluation target (`>= 0.9`) reached
- [ ] Independent held-out multi-seed evaluation completed

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

## Linux NVIDIA vision smoke (2026-08-02)

- Hardware: NVIDIA GeForce RTX 2080 Ti, 11 GiB, CUDA toolkit 12.9
- Runtime: Linux x86_64, Python 3.12.3, JAX/JAXLIB 0.6.2, Warp 1.15.0
- Run commit: `35adbdf60613acf7b123299db6bd3f0e576381bb`
- Test: `PandaPickCubeCartesian` visual PPO smoke mode, 100,000 timesteps
- Result: passed, as confirmed from the Linux GPU run
- Artifact root: `reproduction/artifacts/panda-vision-smoke/`
- Main outputs: `console.log`, `manifest.json`, and
  `evaluation-summary.json`
- Checkpoints, TensorBoard events, and replay videos: timestamped directory
  under `reproduction/artifacts/panda-vision-smoke/runs/`
- Storage policy: runtime outputs remain local and are ignored by Git

An initial exact-profile full attempt used 1024 train environments, 128 eval
environments, and batch size 256. The initial evaluation passed, but the first
training epoch exhausted the 11 GiB GPU while requesting another 5.09 GiB.
The launcher now distinguishes the guarded exact `official` mode from the
10M-step, memory-aware `full` mode; this GPU selects 512/64/128 for `full`.

## Linux NVIDIA adaptive full run (2026-08-02)

- Hardware: NVIDIA GeForce RTX 2080 Ti, 11 GiB
- Workload: 10,035,200 effective timesteps, 512 train environments, 64 eval
  environments, batch size 128
- Best evaluation: step 5,017,600, mean reward `5.077643`, success `0.046875`
  (3/64)
- Final evaluation: step 10,035,200, mean reward `4.318500`, success `0.015625`
  (1/64)
- Video review: all four deterministic rollouts approach the cube, but none
  visibly completes a lift
- Assessment: pipeline reproduction succeeded; the learned policy has not
  converged to a reliable grasp-and-lift behavior
- Next step: run `./reproduction/train_panda_gpu.sh finetune`, which selects
  the step-5,017,600 checkpoint and writes new artifacts separately
- Runtime artifacts remain local and are not committed to Git

## Linux NVIDIA fine-tune run (2026-08-03)

- Source checkpoint: adaptive full step 5,017,600
- Additional workload: 10,076,160 effective timesteps
- Final evaluation: mean reward `9.594509`, success `0.96875` (62/64)
- Late evaluations: `0.828125` at step 6,297,600, `0.953125` at step
  8,816,640, and `0.96875` at the final step
- Video review: rollouts 0, 2, and 3 visibly reach the lift target; rollout 1
  exposes a remaining hard initial condition
- Assessment: the fine-tuned policy meets the training-evaluation target;
  independent held-out evaluation is still required before the final claim
- Next step: back up the artifact tree, then evaluate four held-out seeds with
  `reproduction/evaluate_panda_gpu.sh`

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
