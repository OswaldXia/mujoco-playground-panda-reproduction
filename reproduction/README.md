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
3. Run the 10M-step vision PPO configuration.
4. If necessary, continue from the strongest checkpoint with a lower learning
   rate.
5. Evaluate fixed-seed episode success and archive the artifacts.

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
# Continue from the best checkpoint if the first full run has not converged.
./reproduction/train_panda_gpu.sh finetune
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
| `finetune` | 10M additional | Best `full` checkpoint; lower learning rate |
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

## Continue an incomplete full run

If the first full run approaches the cube but rarely lifts it, do not continue
from the final checkpoint automatically. Run:

```bash
./reproduction/train_panda_gpu.sh finetune
```

The launcher reads
`reproduction/artifacts/panda-vision-full/evaluation-summary.json`, selects the
checkpoint with the highest evaluation success rate (then the highest reward
when success ties), and trains for another 10M timesteps with learning rate
`0.0005`. New logs, checkpoints, and videos are written separately under
`reproduction/artifacts/panda-vision-finetune/`; the original full run is not
changed. The selected source is recorded in `finetune-source.json`.

For the 2026-08-02 RTX 2080 Ti run, this rule selects step `5,017,600`:
success was `0.046875` (3/64 episodes) and mean reward was `5.077643`. The
final checkpoint was weaker (`0.015625`, 1/64), so it should not be the source.

This is parameter fine-tuning rather than an exact optimizer-state resume.
Brax restores the observation normalizer, policy, and value parameters but
initializes a new optimizer. Useful overrides are:

```bash
PANDA_FINETUNE_TIMESTEPS=5000000 \
PANDA_FINETUNE_NUM_EVALS=9 \
PANDA_FINETUNE_LEARNING_RATE=0.0003 \
  ./reproduction/train_panda_gpu.sh finetune
```

To select an exact checkpoint manually:

```bash
PANDA_FINETUNE_CHECKPOINT="$PWD/reproduction/artifacts/panda-vision-full/runs/<run>/checkpoints/<step>" \
  ./reproduction/train_panda_gpu.sh finetune
```

For an 11 GiB RTX 2080 Ti, the default fine-tune profile remains 512 train
environments, 64 evaluation environments, and batch size 128. A reasonable
acceptance target is at least `0.8` fixed-seed evaluation success, preferably
`0.9`; rollout videos should visibly show grasp and lift to the target before
the environment's immediate success reset.

## Back up a completed run

Create a verified archive on the Linux server before any further experiment:

```bash
./reproduction/backup_panda_run.sh finetune
```

The script validates the manifest, console log, evaluation summary, latest run,
and latest checkpoint. It then archives the complete
`panda-vision-finetune` artifact tree without overwriting an existing file,
reopens the archive to validate it, and writes a SHA-256 checksum. The default
destination is:

```text
~/panda-reproduction-archives/
├── panda-vision-finetune-<UTC timestamp>.tar.gz
└── panda-vision-finetune-<UTC timestamp>.tar.gz.sha256
```

Select another destination with `PANDA_BACKUP_DIR=/path/to/backups`. The
archive remains outside Git because it contains large checkpoints and videos.

## Independent held-out evaluation

Run this after backing up the converged fine-tune result:

```bash
./reproduction/evaluate_panda_gpu.sh
```

With no arguments, the launcher selects the newest numeric checkpoint from the
latest fine-tune run. It evaluates a deterministic policy on four held-out
seeds (`101,202,303,404`), with 256 independent episodes per seed and 1,024
episodes in total. This is inference-only: it never updates or overwrites the
policy.

The schema-version-2 report includes successes and rewards by seed, aggregate
success rate, a 95% Wilson confidence interval, worst-seed success, and one
record for every episode. Each record contains its seed, environment index,
initial cube position, outcome, reward, and episode length. The default
acceptance rule is aggregate success `>= 0.90` and worst-seed success `>= 0.85`.
The terminal groups related fields in aligned, readable sections. Each
completed seed reports its own and cumulative success, reward, elapsed time,
and estimated remaining time. A heartbeat is printed during the compiled
rollout without inventing an unavailable step percentage.

After evaluation, the launcher automatically analyzes success against initial
cube y position. The environment fixes x and only samples y in
`[-0.05, 0.05]`, so a one-dimensional position plot is more informative than
an artificial two-dimensional heatmap. Reports, analysis, and complete console
logs are written to:

```text
reproduction/artifacts/panda-independent-eval/
├── evaluation-<UTC timestamp>-<process>.json
├── evaluation-<UTC timestamp>-<process>-analysis/
│   ├── analysis-summary.json
│   ├── episodes.csv
│   ├── position-bins.csv
│   ├── position-success-rate.png
│   └── failure-cases.json
└── console-<UTC timestamp>-<process>.log
```

`episodes.csv` provides the denominator that was missing from the first
evaluation, so success rate can be computed for each position bin rather than
inferring difficulty from failures alone. `failure-cases.json` retains the seed
and environment index needed to identify each failed rollout under the same
evaluation batch configuration.

The completed 2026-08-03 evaluation of fine-tune step 10,076,160 passed:
990/1,024 episodes succeeded (`96.68%`), the 95% Wilson interval was
`95.40%`–`97.61%`, and the worst seed achieved `95.70%`. A compact,
machine-readable record is committed at
`reproduction/results/linux-independent-evaluation.json`; the full runtime
report and console log remain in the ignored artifact directory.

That first report used schema version 1 and recorded failure positions only.
The schema-version-2 repeat evaluated the same checkpoint over another 1,024
episodes and achieved 988/1,024 (`96.48%`). It found 265/290 success (`91.38%`)
for `y < -0.02`, compared with 723/734 (`98.50%`) elsewhere. The weakest bin,
`[-0.03, -0.02)`, achieved 67/76 (`88.16%`). A two-sided Fisher exact test for
the broader left region versus the remainder gives `p = 2.17e-7`, supporting a
relative spatial weakness while leaving the aggregate acceptance unchanged.
The curated evidence is in
`reproduction/results/linux-position-stratified-analysis.json`.

### Trajectory-level failure classification

Schema version 4 adds stage-level diagnostics without changing the policy,
checkpoint, reward, or training configuration. It records approach, the
environment's official 12 mm `reached_box` condition, close/open commands,
physical finger aperture, left/right/bilateral finger-pad contact, lifting,
dropping, target-height error, and the first step of each key event. A failure
is assigned to exactly one class:

- `out_of_bounds_or_invalid`
- `never_approached`
- `approached_not_reached`
- `reached_no_lift`
- `lifted_then_dropped`
- `lifted_timeout`

`ever_hand_box_collision` specifically means collision with the hand capsule;
it is retained as a diagnostic but is not treated as successful grasp contact.
The evaluator prints class counts at completion, adds all trajectory fields to
`episodes.csv`, and writes a standalone `failure-classification.json` in the
analysis directory.

The first schema-version-3 trajectory run recorded 963/1,024 left-side success
(`94.04%`) and classified 52 of 61 failures (`85.25%`) as
`reached_no_lift`. Successful and failed episodes both issued their first close
command at step 1, so that event alone does not distinguish the outcome. The
same report exposed 50 step-1 reached episodes, all successful, matching the
environment's hard-coded 5% guide-state exploration aid. Historical schema-2
and schema-3 evaluations therefore mix policy performance with training
assistance and are retained as development evidence, not guide-free acceptance.
The compact analysis record is committed at
`reproduction/results/linux-trajectory-failure-analysis.json`.

The guide-state probability is now configurable: training preserves the 5%
default, while periodic training evaluation, replay inference, and independent
formal evaluation force it to zero. The independent evaluator verifies the
effective value and records it in every schema-version-4 report. A robustness
resume also rejects legacy reports so old and corrected evaluations cannot be
combined.

Run the focused collection on the Linux NVIDIA server with:

```bash
git pull
./reproduction/evaluate_panda_failure_modes_gpu.sh
```

The command automatically selects the completed robustness checkpoint and
evaluates 1,024 new guide-free episodes over the left-side range
`[-0.05, -0.02)`, which was the only missed gate. Results are stored under
`reproduction/artifacts/panda-failure-modes/<timestamp>/`. This must be a new
rollout: earlier reports either lack intermediate states or include the
guide-state aid. The new finger-contact and aperture fields allow successful
and failed grasp acquisition to be compared directly.
Use the dominant class to choose one controlled intervention; do not combine
multiple reward, sampling, and policy changes in the first follow-up run.

To evaluate a particular checkpoint or reduce memory use:

```bash
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint /absolute/path/to/checkpoints/000010076160 \
  --num-envs 128
```

The evaluation requires the Linux NVIDIA server. On an 11 GiB RTX 2080 Ti,
start with the default 256 environments; if it reports GPU out-of-memory,
retry with 128. Each seed is evaluated sequentially, and the command prints a
heartbeat during the first JIT compilation.

## Targeted left-side robustness experiment

The position-stratified evaluation identified `y < -0.02` as a relative weak
region: 265/290 success (`91.38%`) versus 723/734 (`98.50%`) elsewhere. First
validate checkpoint restore and the new sampling path with an isolated 100k
smoke run:

```bash
./reproduction/smoke_panda_robustness_gpu.sh
```

Its outputs remain under
`reproduction/artifacts/panda-vision-robustness-smoke/` and do not overwrite the
formal experiment. After it completes, run one controlled fine-tune from the
converged step-10,076,160 checkpoint:

The completed smoke reached effective step 102,400 and recorded success rates
of `0.9375`, `0.9375`, and `0.90625` at its three evaluations. This passes the
pipeline gate: restored-policy evaluation, targeted updates, TensorBoard data,
and final outputs all completed. The two-episode success change is not treated
as evidence for or against robustness; that claim requires the full regression
suite. Compact evidence is committed at
`reproduction/results/linux-robustness-smoke.json`.

```bash
./reproduction/train_panda_gpu.sh robustness
```

The default profile is intentionally smaller than the converged fine-tune:

- 3,000,000 timesteps;
- learning rate `0.0001`;
- the same VRAM-aware 512/64/128 environment and batch profile on an 11 GiB
  RTX 2080 Ti;
- 50% original uniform resets over `[-0.05,0.05]` and 50% targeted uniform
  resets over `[-0.05,-0.02]`.

Because the original component also lands left of `-0.02` 30% of the time,
the resulting training mixture places approximately 65% of resets in the weak
left region while preserving coverage of the complete original distribution.
Artifacts are isolated under
`reproduction/artifacts/panda-vision-robustness/`. Override only when running a
deliberate ablation, for example:

```bash
PANDA_ROBUSTNESS_TIMESTEPS=2000000 \
PANDA_ROBUSTNESS_LEARNING_RATE=0.0001 \
PANDA_ROBUSTNESS_TARGET_PROBABILITY=0.5 \
  ./reproduction/train_panda_gpu.sh robustness
```

After training, run the fixed regression suite:

```bash
./reproduction/evaluate_panda_robustness_gpu.sh
```

It selects the strongest robustness checkpoint and evaluates 1,024 episodes on
each of three distributions:

| Distribution | Range | Acceptance |
| --- | --- | --- |
| Original | `[-0.05,0.05]` | aggregate `>=0.95`, worst seed `>=0.90` |
| Left | `[-0.05,-0.02]` | aggregate `>=0.95`, worst seed `>=0.90` |
| Hard bin | `[-0.03,-0.02]` | aggregate `>=0.93`, worst seed `>=0.85` |

The suite writes all raw reports and position plots under a timestamped
`reproduction/artifacts/panda-robustness-eval/` directory, then produces
`robustness-summary.json` with absolute gains and two-sided Fisher exact tests
against the recorded baseline. A failed criterion is a valid experimental
result and must not be hidden by selecting only favorable rollouts.

If plotting or a later distribution fails after a raw report has been written,
resume the same directory instead of repeating completed GPU rollouts:

```bash
./reproduction/evaluate_panda_robustness_gpu.sh \
  --resume /absolute/path/to/panda-robustness-eval/<run>
```

Resume mode validates and reuses each existing 1,024-episode report, regenerates
its analysis with overwrite enabled, and runs only the missing distributions.

The completed experiment selected robustness checkpoint step 2,007,040. It
preserved the original distribution at 1003/1,024 (`97.95%`), improved the left
region from `91.38%` to 968/1,024 (`94.53%`), and improved the hard bin from
`88.16%` to 970/1,024 (`94.73%`). The hard-bin comparison was statistically
significant (`p=0.0343`); the left comparison was borderline (`p=0.0533`). The
pre-registered overall decision remains **FAIL** because the left point estimate
missed its 95% threshold by five episodes. This is recorded as a partial success,
without changing the threshold or choosing another checkpoint after seeing the
test results. Evidence is in
`reproduction/results/linux-targeted-robustness-evaluation.json`.

Back up a completed robustness run separately with:

```bash
./reproduction/backup_panda_run.sh robustness
```

All training modes use seed 1, write TensorBoard scalars, and extract the final
mean episode reward and `reward/success` metric to `evaluation-summary.json`.
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
`reproduction/artifacts/panda-vision-full/`; fine-tuning uses
`reproduction/artifacts/panda-vision-finetune/`; the exact profile uses
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
