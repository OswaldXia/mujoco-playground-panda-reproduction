# 12｜毕业实验：接触后的稳定抬升

建议时间：12–20 小时，不含等待。硬件：Linux NVIDIA GPU。

## 学习目标

- 从失败证据提出可证伪机制假设；
- 预注册一个单变量 reward 改动；
- 完成 smoke、训练、三分布回归与轨迹复检；
- 无论 PASS/FAIL，都形成可用于求职展示的工程报告。

## 为什么选择这个问题

guide-free 左侧评估为 964/1,024（94.14%），固定 95% 门槛差 9 次。60 次失败
中 55 次是 `reached_no_lift`；成功与失败均 100% 曾有双指接触，首次接近、
首次 reached 和首次双指接触时间也相近。差异主要出现在 reached 到 lift：成功
平均约 40.8 步，失败仅 5 条最终 lift，平均约 92.8 步。

**预注册假设：** 在双指接触后，对方块从桌面高度向 `lift_height=0.05 m`
移动的连续进度提供奖励，会降低 `reached_no_lift`，并使 guide-free 左侧成功率
达到至少 95%，同时不破坏原分布性能。

这不是“结果保证”。若失败类型不变或原分布退化，假设被否定或需要修订。

## 核心知识

机制实验的最小单位是“一个可观察瓶颈、一个有方向的干预、一个固定对照协议”。
reward shaping 既可能改善 credit assignment，也可能制造新捷径，所以必须同时
检查任务成功、原分布回归和失败类型转移。机制指标支持解释，但不能取代预注册
的最终成功率门槛。

## 唯一改动

新 reward raw term：

\[
r_{stable\_lift}=I(\text{bilateral contact})\cdot
clip\left(\frac{z_{box}-z_{reset}}{0.05-z_{reset}},0,1\right).
\]

沿用环境已有 progress-reward 机制，只给历史最好值的正增量。权重先固定一个
保守值并在 smoke 前写入契约；首个正式运行不同时修改采样、PPO、网络、成功
条件、guide probability 或 episode length。通用 contact bonus 被排除，因为
成功/失败都已经取得接触。

## 控制项

| 项目 | 固定值 |
| --- | --- |
| 起始 checkpoint | robustness step 2,007,040 |
| 环境 | `PandaPickCubeCartesian`, vision/Warp |
| 训练 sampling | 与对照 robustness run 相同的 50% targeted mixture |
| PPO/网络 | 与对照完全相同 |
| 训练预算 | 3M effective timesteps |
| train guide | 0.05（保持 checkpoint 训练语义） |
| formal eval guide | 0.0 |
| seeds | 101, 202, 303, 404 |
| episodes | 每分布 4×256 |
| checkpoint rule | success 优先、reward 次级，运行前固定 |

## 验收标准

- original aggregate `>=95%`，worst seed `>=90%`；
- left `[-0.05,-0.02]` aggregate `>=95%`，worst seed `>=90%`；
- hard bin `[-0.03,-0.02]` aggregate `>=93%`，worst seed `>=85%`；
- guide-free left 的 `reached_no_lift` incidence 少于固定 baseline 55/1,024，
  并另报其占全部失败的比例；
- 所有报告 schema 4、guide probability 0.0、无策略更新。

前三组与历史预注册门槛一致。最后一项是机制指标，不能用百分比下降替代前三组
工程门槛。不得在看见结果后调整阈值或只选有利 seed。

## 最小实验

先只实现并单元测试 reward 函数，不启动训练。构造无接触、接触且高度递增、
高于 lift threshold 三组输入，验证门控、单调性和封顶；再做 100k smoke。
只有测试和 smoke 均通过，才授权一次正式 3M 运行。

## 源码定位

- `pick_cartesian.py::step`：增加 raw reward 与 metric 的唯一代码位置；
- `train_panda_gpu.sh`：复制为新模式或显式参数化，产物根目录必须独立；
- failure classifier/evaluator：复用现有 schema-4，不修改分类规则；
- [`reference/configurations.md`](../reference/configurations.md)：对照配置表。

## 运行前预测

把以下内容写进实验 README：reward 权重、期望下降的 failure class、可能出现的
副作用（夹住不抬/过早抬升/碰撞）、会否定假设的观察。先冻结后写代码。

## 操作

从功能分析分支创建独立实验分支，不从旧 left 分支继续堆第二个概念：

```bash
git switch analysis/guide-free-grasp-diagnostics
git pull --ff-only
git switch -c experiment/reach-to-lift-stability
```

实现后按四步推进：

1. 单元测试 reward 在无接触时为 0、随高度单调、达到阈值封顶；
2. 100k smoke 验证恢复、更新、metric、checkpoint、视频；
3. 单次 3M 正式训练，不并行扫多个事后权重；
4. original/left/hard 三分布 guide-free 回归，再跑 left trajectory 分类。

本课程不预先给出尚未实施的命令名。实现 launcher 时应采用清晰的新模式与
`panda-reach-to-lift-*` artifact root，并补入 `reference/commands.md`。

## 预期结果

预期是“得到可解释结论”，不是预设 PASS。理想结果为 left 超过 95%、original
保持、`reached_no_lift` 明显减少；如果只提高 reward 或训练评估而正式回归
不变，结论是失败；若 left 提升但 original 退化，结论是局部收益/整体不接受。

## 常见错误

- 为尽快过线同时增加 reward、继续 targeted sampling、改学习率；
- 看到 smoke 成功率好就停止正式训练；
- 用新训练选过的 seed 做“held-out”；
- 只比较 55 个绝对失败数，不比较总失败/回合数；
- 把一次 94.14→95.0 的点估计变化写成普遍改进而不报区间。

## 修改练习

真正完成该实验，并按
[`exercises/capstone-report-template.md`](../exercises/capstone-report-template.md)
填写报告。报告必须包含失败结果路径，不能只展示最好视频。

## 自测

1. 为什么这个 reward 要由 bilateral contact 门控？
2. 为什么训练仍保留 0.05 guide，正式评估必须为 0？
3. 哪个结果会否定“post-contact stability”假设？
4. 为什么新实验必须有新分支和 artifact root？

## 通过标准

代码测试通过、smoke 完整、正式训练只一次、三分布评估和轨迹分类齐全；报告
能区分事实、推断和限制，并如实给 PASS/FAIL。完成后通过 Gate 5。

## Git 节点

建议至少四次提交：

```text
test: specify contact-gated lift progress
feat: add contact-gated lift progress experiment
docs: preregister reach-to-lift evaluation
docs: report reach-to-lift experiment result
```
