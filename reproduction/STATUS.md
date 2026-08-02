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
- [ ] macOS environment installed
- [ ] macOS state reset/step/render passed
- [ ] Linux NVIDIA vision smoke test passed
- [ ] Official 10M-step vision training completed
- [ ] Fixed-seed success evaluation completed
- [ ] Results documented

## Known hardware boundary

Apple Silicon is used for source inspection, state-environment debugging, and
native rendering. The official visual training path uses the MJWarp batch
renderer and requires an NVIDIA GPU for practical throughput.

