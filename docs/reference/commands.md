# 命令速查

所有命令从仓库根目录运行。运行前先用 `git status --short --branch` 确认分支。

## Mac 验证

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  -c reproduction/constraints-2026-08-02.txt \
  -e '.[notebooks]'
MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/smoke_test_macos.py \
  --output-dir reproduction/artifacts/macos-smoke
python reproduction/vision_backend_probe.py \
  --require-ready \
  --image reproduction/artifacts/macos-vision-observation.png \
  --output reproduction/artifacts/macos-vision-probe.json
```

## Linux GPU 安装与训练

```bash
./reproduction/setup_gpu.sh
./reproduction/train_panda_gpu.sh smoke
./reproduction/train_panda_gpu.sh full
./reproduction/train_panda_gpu.sh finetune
```

仅高显存精确并行度：

```bash
./reproduction/train_panda_gpu.sh official
```

## 备份

```bash
./reproduction/backup_panda_run.sh finetune
./reproduction/backup_panda_run.sh robustness
sha256sum -c ~/panda-reproduction-archives/<archive>.sha256
```

## 独立评估

```bash
./reproduction/evaluate_panda_gpu.sh
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint /absolute/path/to/checkpoints/<step> \
  --num-envs 128
```

## 鲁棒性与轨迹诊断

```bash
./reproduction/smoke_panda_robustness_gpu.sh
./reproduction/train_panda_gpu.sh robustness
./reproduction/evaluate_panda_robustness_gpu.sh
./reproduction/evaluate_panda_failure_modes_gpu.sh
```

恢复未完成三分布评估：

```bash
./reproduction/evaluate_panda_robustness_gpu.sh \
  --resume /absolute/path/to/panda-robustness-eval/<run>
```

## 查看结果

```bash
python3 -m json.tool reproduction/artifacts/<run>/evaluation-summary.json
find reproduction/artifacts -name 'rollout*.mp4' -print
tensorboard --logdir reproduction/artifacts/panda-vision-finetune/runs
```

## 课程 lab 与测试

```bash
source .venv/bin/activate
python docs/labs/01_transform_2d.py
python docs/labs/02_ppo_clipping.py --epsilon 0.3
python docs/labs/09_wilson_interval.py
python docs/labs/03_mujoco_state.py
python docs/labs/04_jax_batching.py
python docs/labs/05_panda_inspect.py
python -m pytest reproduction/tests -q
```
