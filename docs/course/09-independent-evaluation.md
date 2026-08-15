# 09｜独立评估与统计判断

建议时间：8 小时。硬件：Linux NVIDIA GPU；统计 lab 可在 Mac 运行。

本章有两条明确路径：Mac/无 checkpoint 先完成“离线路径”，得到 Gate 4
`PRACTICED`；只有在 GPU 上对自己的冻结 checkpoint 采集完整记录，才能得到
`READY`。评分边界见 [`GATE_RUBRIC.md`](GATE_RUBRIC.md)。

## 学习目标

- 恢复冻结策略进行 inference-only 评估；
- 使用多种子、固定分布、充分分母和 Wilson 区间；
- 区分 point estimate、置信区间、worst seed 与验收门槛；
- 从 JSON/CSV 追溯每个聚合数字。

## 为什么不能只看四个视频

视频适合发现行为模式，不足以估计成功率。4/4 成功的样本仍很小，而且相机
视角可能遮挡关键接触。独立评估冻结 policy，不更新参数，以四个种子各 256
回合形成 1,024 个二项结果，同时保存每回合位置、奖励和长度。

## 核心知识

成功率 `p_hat=k/n`。简单的 `p_hat ± 1.96 sqrt(p(1-p)/n)` 在接近 0/1 或
小样本时表现差；Wilson 区间把中心和宽度作校正。本项目既看 aggregate，也看
worst seed，避免平均数掩盖某个随机批次的弱点。

门槛是决策规则，不是置信区间。点估计 94.14% 可拥有上界超过 95%，但固定
`>=95%` 点估计门槛仍然 FAIL。不能因“统计上可能达到”改写工程验收。

## 最小实验

先完成
[`09_evaluation_statistics.ipynb`](../notebooks/09_evaluation_statistics.ipynb)。它读取
已提交 compact evidence，可视化 per-seed 和样本量对区间宽度的影响，并用断言
阻止把 8 条精选轨迹误作 50% 成功率样本。

运行 [`labs/09_wilson_interval.py`](../labs/09_wilson_interval.py)，比较 62/64、
964/1,024 与 990/1,024 的区间。再从 guide-free compact JSON 手算总数和最差
种子。

无需 GPU 的完整练习：

```bash
python docs/labs/09_offline_evaluation.py
```

脚本从已提交 compact evidence 重算 aggregate、分 seed、Wilson 与固定门槛。随后
打开 [`课程数据说明`](../data/README.md)，解释为何 8 条精选 episode 不能计算
正式成功率。

## 源码定位

- [`evaluate_panda_gpu.sh`](../../reproduction/evaluate_panda_gpu.sh)；
- [`evaluate_panda_checkpoint.py`](../../reproduction/evaluate_panda_checkpoint.py)；
- [`analyze_panda_evaluation.py`](../../reproduction/analyze_panda_evaluation.py)；
- [`linux-guide-free-left-trajectory-analysis.json`](../../reproduction/results/linux-guide-free-left-trajectory-analysis.json)。

## 运行前预测

预测 62/64 和 990/1,024 哪个区间更窄；说明样本量和点估计各如何影响宽度。
写下评估运行中唯一允许变化的状态。

## 操作

```bash
python docs/labs/09_wilson_interval.py
python docs/labs/09_offline_evaluation.py

# 以下仅在 Linux NVIDIA 服务器、且已有 checkpoint 时运行。
./reproduction/evaluate_panda_gpu.sh
```

指定 checkpoint 或降低并行度：

```bash
./reproduction/evaluate_panda_gpu.sh \
  --checkpoint /absolute/path/to/checkpoints/<step> \
  --num-envs 128
```

## 预期结果

终端依次验证 runtime/checkpoint、加载确定性策略、按 seed 汇报累计成功与 ETA，
最后写入时间戳 JSON、分析目录、CSV、图、失败列表和完整日志。执行过程中没有
optimizer update，checkpoint 不被覆盖。

## 常见错误

- 用训练期间的 64 回合评估作最终结论；
- 只提交 success rate，不提交 successes/episodes；
- 多次运行只挑最好一次；
- 混用不同分布或 guide probability 的结果；
- 把置信区间上界超过门槛解释为验收通过。

## 修改练习

给 964/1,024 写一段四句结论，必须同时包含点估计、Wilson 区间、worst seed、
固定门槛结论。不得使用“基本通过”。

## 自测

1. 为什么 62/64 不如 990/1,024 稳健？
2. 固定 seed 是否等于无随机性？
3. 为什么评估必须记录初始位置和每回合结果？
4. 94.14% 的区间覆盖 95% 时，95% 门槛为何仍 FAIL？

## 通过标准

离线路径：Notebook 与脚本通过、能正确解释区间/门槛/精选样例边界，记为 Gate 4 PRACTICED。
正式路径：再从自己的原始 episode records 重算聚合与 per-seed，artifact 完整且
策略无更新，才记为 Gate 4 READY。

## Git 节点

```bash
git add notes/09-evaluation-conclusion.md
git commit -m "docs: report independent Panda evaluation"
```
