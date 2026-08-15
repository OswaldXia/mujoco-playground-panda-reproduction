# 10｜失败分析与鲁棒性实验

建议时间：10 小时。硬件：分析可在 Mac，采集需 Linux NVIDIA GPU。

## 学习目标

- 从聚合成功率下钻到位置分层与轨迹阶段；
- 区分探索性失败分析与预注册鲁棒性实验；
- 设计 original/weak-region/hard-bin 回归；
- 用失败机制而非直觉选择下一项改动。

## 为什么聚合成功率不够

整体 96% 可能隐藏某个位置只有 88%。反过来，看到失败集中在负 y 也不能直接
断言负 y 更难，因为没有成功样本作分母。正确路径是先记录每回合起点，再按
预定义 bin 计算条件成功率和区间；随后用新固定分布验证，而不是持续切 bin
寻找最差结果。

## 核心知识

本项目 schema-2 发现 `y<-0.02` 为 265/290（91.38%），其余为 723/734
（98.50%）。这属于位置分层的探索性发现。随后受控训练把约 65% reset 放入
左区，checkpoint 在历史含 guide 协议下保持 original 97.95%，left 94.53%，
hard bin 94.73%，但固定 left 95% 门槛仍差 5 次，整体 FAIL。

轨迹分类进一步把失败互斥地分为：非法/越界、从未接近、接近未到达、到达未
抬起、抬起后掉落、抬起但超时。guide-free schema-4 中 55/60 是
`reached_no_lift`。成功与失败都 100% 曾双指接触，说明“再奖一次接触”缺乏
区分力；瓶颈更像接触后的稳定抬升。

Fisher exact test 可比较两个二项组，但 p 值不是效应大小；多 bin 探索还要
考虑多重比较。最终工程决策应同时看效果量、区间、回归门槛和失败类型。

## 最小实验

先完成共享的
[`10_failure_analysis.ipynb`](../notebooks/10_failure_analysis.ipynb)。它按分类守恒、
事件时间线、接触对照、位置分母与 guide 协议顺序分析，避免从单个视频直接猜 reward。

用已提交 compact JSON 回答：改进前左区、历史鲁棒 checkpoint、guide-free
确认三者协议是否相同？哪些数字能直接对比，哪些只能作为开发证据？然后从
60 个失败重算各类占比。

Mac 离线路径直接运行：

```bash
python docs/labs/10_offline_failure_analysis.py
```

再从 [`精选 episode fixture`](../data/guide-free-left-episodes-fixture.json) 选择一条
成功与一条 `reached_no_lift`，按 approached→reached→contact→lift→success
画时间线。记住该 fixture 被有意平衡，不能估计成功率。

## 源码定位

- [`analyze_panda_evaluation.py`](../../reproduction/analyze_panda_evaluation.py)；
- [`panda_failure_classification.py`](../../reproduction/panda_failure_classification.py)；
- [`evaluate_panda_robustness_gpu.sh`](../../reproduction/evaluate_panda_robustness_gpu.sh)；
- [`evaluate_panda_failure_modes_gpu.sh`](../../reproduction/evaluate_panda_failure_modes_gpu.sh)；
- [`linux-guide-free-left-trajectory-analysis.json`](../../reproduction/results/linux-guide-free-left-trajectory-analysis.json)。

## 运行前预测

只看一条 `reached_no_lift` 轨迹前，预测最小距离、首次接触、首次 reached 和
最大高度的关系。写下若所有失败都曾接触，contact bonus 的可证伪预测是什么。

## 操作

先在任意机器完成离线分析；在服务器采集自己的 guide-free 轨迹：

```bash
python docs/labs/10_offline_failure_analysis.py

# 以下切换会离开课程分支；先提交当前 notes，并确认目标分支存在。
git switch analysis/guide-free-grasp-diagnostics
./reproduction/evaluate_panda_failure_modes_gpu.sh
```

历史鲁棒性三分布回归的入口为：

```bash
./reproduction/evaluate_panda_robustness_gpu.sh
```

失败后使用同目录恢复，避免重跑已完成分布：

```bash
./reproduction/evaluate_panda_robustness_gpu.sh \
  --resume /absolute/path/to/panda-robustness-eval/<run>
```

## 预期结果

每回合只有一个 failure class；分类总数等于总 failures；summary 给出 success
与 failure 的抓取特征对照。当前正式 guide-free 证据应为 964/1,024、60 个
失败、55 个 `reached_no_lift`，而非历史 schema-3 的 963/1,024。

## 常见错误

- 从失败位置数量推断条件失败率；
- 事后缩小 bin 直到得到显著 p 值；
- 同时改奖励、采样和网络，结果无法归因；
- 因 hard bin 提升就忽略 original regression；
- 把历史 guide-assisted 结果与 guide-free 结果拼成一条提升曲线。

## 修改练习

写一份 `notes/10-failure-hypothesis.md`：只选择一个 dominant class，只提出一个
机制假设、一个可观察中间指标、一个最小改动和一个会否定假设的结果。

## 自测

1. 失败聚集与条件失败率有什么区别？
2. 为什么 failure classes 要互斥且覆盖所有失败？
3. 为什么“双指接触率 100%”反对通用 contact bonus？
4. 鲁棒性改进为何必须包含原分布回归？

## 通过标准

能从 episode records 复核分类总数；能把“发现问题”和“验证改进”写成两个
实验；下一假设由数据支持且只改变一个因素。

## Git 节点

```bash
git add notes/10-failure-hypothesis.md
git commit -m "docs: derive one Panda failure hypothesis"
```
