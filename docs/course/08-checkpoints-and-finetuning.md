# 08｜Checkpoint、选择与 Fine-tune

建议时间：6 小时，不含训练。硬件：Linux NVIDIA GPU。

## 学习目标

- 根据预先定义的规则选 checkpoint，而不是默认最后一步；
- 区分参数恢复与完整 optimizer-state resume；
- 从最佳 full checkpoint 进行独立 fine-tune；
- 创建可验证、不可覆盖的归档。

## 本章知识清单

- **checkpoint 选择集**：按预先规则从开发评估中选模型，不偷看最终 held-out；
- **最后一步不等于最佳**：on-policy 训练会波动，应按 success/reward 规则选择；
- **参数 fine-tune**：恢复网络参数但重置 optimizer，不能称为完整 resume；
- **评估隔离**：训练、选择和最终判断使用不同职责的数据；
- **不可覆盖归档**：产物带时间/step/SHA-256，写入后重新打开验证。

## 为什么最后 checkpoint 不一定最好

on-policy 训练会波动。本项目首轮 full 在 5,017,600 步达到 3/64，而最终
10,035,200 步降到 1/64。续训脚本先按 success，再按 reward 打破平局，选择
中间 checkpoint。它恢复网络/normalizer 参数，但重新初始化 optimizer，因此
应称为 parameter fine-tune，而不是无缝续跑。

## 核心知识

checkpoint 选择集、训练集和最终 held-out 评估应隔离。反复查看 held-out 后
挑 checkpoint 会把它变成开发集。fine-tune 使用更低学习率 `0.0005` 并写入
新的 artifact root，不覆盖 full。备份包含日志、manifest、summary、events、
视频和 checkpoint，生成 SHA-256 并重新打开归档验证。

本项目 fine-tune 追加约 10.08M 步，最终训练评估 62/64；它随后仍需独立
1,024 回合评估才能成为更可靠的性能证据。

## 最小实验

不启动训练，先运行 checkpoint 选择器读取已有 full summary，手工核对选中步。
若服务器上没有本项目产物，阅读已提交结果并在纸面完成选择。

只运行选择器（不会启动训练）：

```bash
python reproduction/select_best_checkpoint.py \
  --summary /absolute/path/to/training-summary.json \
  --runs-dir /absolute/path/to/panda-vision-full/runs
```

先用 `find reproduction/artifacts -name 'training-summary.json' -print` 定位 summary。
选择器输出 step、success、reward 与 checkpoint 路径；确认后才进入下面的训练操作。

## 源码定位

- [`select_best_checkpoint.py`](../../reproduction/select_best_checkpoint.py)；
- `train_panda_gpu.sh` 的 `finetune` 分支；
- [`backup_panda_run.sh`](../../reproduction/backup_panda_run.sh)；
- [`linux-independent-evaluation.json`](../../reproduction/results/linux-independent-evaluation.json)。

## 运行前预测

根据 summary 预测选中的 checkpoint。写下如果 success 相同，为什么用 reward
作次级规则；再说明为什么不能查看最终 held-out 后重新选一个“更好”的 checkpoint。

## 操作

```bash
./reproduction/train_panda_gpu.sh finetune
./reproduction/backup_panda_run.sh finetune
```

人工核验归档：

```bash
sha256sum -c ~/panda-reproduction-archives/*.sha256
```

## 预期结果

fine-tune 输出位于独立 `panda-vision-finetune/`，并记录 source checkpoint。
备份目录出现时间戳 tar.gz 和同名 `.sha256`，脚本拒绝覆盖已有文件。

## 常见错误

- 自动使用最新数字目录；
- 把 fine-tune 步数写成从零开始的完整基线步数；
- 认为恢复参数也恢复了 optimizer 动量；
- 归档后不校验，或把唯一 checkpoint 留在临时服务器；
- 将大 checkpoint 提交普通 Git。

## 修改练习

写一个三行 checkpoint 选择策略：主指标、平局规则、仍相同时的规则。用项目
历史三个 evaluation 点手工执行，并保存为 `notes/08-checkpoint-policy.md`。

## 自测

1. 为什么“最后一步”不是合理的默认选择规则？
2. parameter fine-tune 与 exact resume 差在哪里？
3. SHA-256 能证明什么，不能证明什么？

## 通过标准

能从 summary 独立复现 checkpoint 选择；能恢复并生成新 artifact root；归档
校验通过，且原 full 产物未改变。

## Git 节点

```bash
git add notes/08-checkpoint-policy.md
git commit -m "docs: define checkpoint selection and archive policy"
```
