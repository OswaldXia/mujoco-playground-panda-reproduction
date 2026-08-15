# 11｜评估完整性：Guide-state 案例

建议时间：5 小时。硬件：Mac 阅读与测试，GPU 用于最终确认。

## 学习目标

- 识别训练辅助泄漏到评估的路径；
- 用源码、配置和轨迹异常三角验证；
- 修复评估而不改变训练语义；
- 正确降级历史结论并重新采集证据。

## 本章知识清单

- **评估泄漏**：训练辅助、挑 seed 或协议差异进入评估，使数字不再代表目标分布；
- **effective config**：评估真正采用的值必须从运行对象读取并写进报告；
- **三角验证**：源码、配置和轨迹异常三类证据相互印证；
- **schema 版本**：协议字段变化必须升级结构并拒绝不兼容的旧输入；
- **历史结论降级**：保留旧结果但缩小主张，修复后重新采集，不能删样本补救。

## 为什么需要完整性审计

策略、仿真和日志都可以正常运行，但评估协议中的一条训练捷径仍可能让数字失真。
完整性审计回答“这个成功真的是标准初始条件下由策略造成的吗”，它应发生在继续
调 reward 或扩大训练预算之前。

## 案例背景

环境原本在每个新回合以 5% 概率把仿真状态换到 `picked` keyframe，以帮助训练
探索稀疏成功奖励。早期评估复用了同一配置。schema-3 轨迹报告出现 50 个在
step 1 已 reached 的回合，恰接近 1,024 的 5%，且 50 个全部成功。这不是
策略从普通 reset 一步完成抓取，而是训练辅助进入了评估。

## 核心知识

评估泄漏不只指训练数据出现在测试数据，也包括：自动示范/起始状态、奖励捷径、
环境随机化差异、反复挑 seed/checkpoint、终止后 autoreset 数据被当末状态。

本项目的修复原则：

1. 训练默认仍为 0.05，保持已训练 checkpoint 的语义；
2. periodic evaluation、replay inference、独立评估显式覆盖为 0.0；
3. evaluator 读取 effective value 并写入 schema-4；
4. robustness resume 拒绝旧 schema，防止混合协议；
5. 历史结果保留但标记 `historical_guide_assisted`；
6. 重新采集 1,024 回合，而不是从旧数据“扣掉疑似 5%”。

不能简单删除 50 条 step-1 轨迹，因为 guide-state 会改变批量状态、终止与后续
autoreset；筛除后的样本也不再是预定义评估分布。

## 最小实验

若尚未完成，先运行
[`10_failure_analysis.ipynb`](../notebooks/10_failure_analysis.ipynb) 的“Guide-state
完整性审计”部分；它并排展示 schema、status、guide probability 与 step-1 异常。

对比
[`linux-trajectory-failure-analysis.json`](../../reproduction/results/linux-trajectory-failure-analysis.json)
与
[`linux-guide-free-left-trajectory-analysis.json`](../../reproduction/results/linux-guide-free-left-trajectory-analysis.json)。
列出 schema、guide probability、step-1 异常、success 和正式状态。

先在 Mac 运行协议对比：

```bash
python docs/labs/11_offline_integrity_audit.py
```

它不会把两份数字合并，而是验证 historical 标签、非零 guide、step-1 异常与
schema-4 guide-free 修复链。

## 源码定位

- `pick_cartesian.py::sample_guide_swap/default_config/step`；
- `learning/train_jax_ppo.py` 的 evaluation/inference override；
- `evaluate_panda_checkpoint.py` 的 config validation；
- `evaluate_panda_robustness_gpu.sh` 的 resume schema validation；
- 相关测试：`test_panda_position_sampling.py` 与 evaluator tests。

## 运行前预测

如果修复正确，训练默认 config、正式 evaluator effective config、step-1 reached
各应是什么值？预测 guide-free 成功率一定下降吗，并解释为什么不能保证单次
采样严格单调。

## 操作

```bash
python docs/labs/11_offline_integrity_audit.py
python -m unittest discover -s reproduction/tests -p 'test_*.py' -v

# 首选；若没有 rg，使用下一条 grep 命令。
rg -n "guide_swap_probability" mujoco_playground learning reproduction
grep -RIn "guide_swap_probability" mujoco_playground learning reproduction
```

随后在 GPU 上用第 10 章命令重新采集，不复用旧 report。

## 预期结果

测试覆盖默认训练值 0.05、正式评估值 0.0 和旧 schema 拒绝。新 compact report
明确写 `guide_free_formal_evaluation`，不会出现由 guide 造成的 step-1 reached。

## 常见错误

- 为修评估把训练默认也改成 0，导致 checkpoint 训练协议被追溯性改写；
- 从旧报告筛数据代替重新采样；
- 删除历史结果，失去问题发现链；
- 修复代码后仍引用旧数字作正式结论；
- 只在 shell 传 override，不在输出中记录 effective value。

## 修改练习

写 `notes/11-integrity-audit.md`，检查 observation、reset、reward、termination、
policy stochasticity、checkpoint selection 六类泄漏。每项给出源码证据或
“未发现”的检查依据。

## 自测

1. guide-state 为什么能帮助训练，却不应出现在正式评估？
2. 50/1,024 的异常为什么是强线索但还需要源码验证？
3. 为什么历史数据要保留并降级，而不是删除？
4. effective config 为什么必须写入报告？

## 通过标准

能完整复述“异常→源码→修复→测试→重采集→降级历史结论”的链条；能独立审计
至少六类泄漏。仅运行离线 lab 得到 Gate 4 PRACTICED；自己的 guide-free 全量
报告通过审计后才是 READY，至此具备进入毕业实验的评估完整性基础。

## Git 节点

```bash
git add notes/11-integrity-audit.md
git commit -m "docs: audit Panda evaluation integrity"
```
