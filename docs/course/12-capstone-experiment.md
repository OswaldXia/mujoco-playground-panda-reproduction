# 12｜毕业实验：接触后的稳定抬升

建议时间：12–20 小时，不含等待。硬件：Linux NVIDIA GPU。

## 学习目标

- 从失败证据提出可证伪机制假设；
- 预注册一个单变量 reward 改动；
- 完成 smoke、训练、三分布回归与轨迹复检；
- 无论 PASS/FAIL，都形成可用于求职展示的工程报告。

## 本章知识清单

- **机制瓶颈**：从轨迹证据定位“接触后未稳定抬升”，而非凭直觉改 reward；
- **可证伪假设**：提前写明改动应改变什么、什么结果会否定它；
- **单变量**：treatment 只增加 stable-lift reward，其余协议冻结；
- **等计算量 control**：两臂从同一 checkpoint 各训练相同 3M 步；
- **配对种子与回归**：相同 seeds 比较 original/left/hard 三分布；
- **诚实结论**：FAIL 也报告效果量、失败类型和下一步，不改写门槛。

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

**因果对照必须等计算量。** 起始 checkpoint 的 94.14% 只能描述训练前水平。
若 treatment 在它之上多训练 3M 步，却直接与未继续训练的起点比较，就无法区分
“新 reward 有效”和“原 reward 再训练 3M 也会变化”。因此从同一 checkpoint
分叉两臂：control 保持原 reward 训练 3M，treatment 仅增加新 reward 训练 3M。

## 唯一改动

新 reward raw term：

$$
r_{stable\_lift}=I(\text{bilateral contact})\cdot
clip\left(\frac{z_{box}-z_{reset}}{0.05-z_{reset}},0,1\right).
$$

沿用环境已有 progress-reward 机制，只给历史最好值的正增量。权重先固定一个
保守值并在 smoke 前写入契约；首个正式运行不同时修改采样、PPO、网络、成功
条件、guide probability 或 episode length。通用 contact bonus 被排除，因为
成功/失败都已经取得接触。

## 配对对照设计

| 项目 | Control | Treatment |
| --- | --- | --- |
| 起始 checkpoint | 同一个 robustness step 2,007,040 + SHA-256 | 完全相同 |
| 唯一差异 | 原 reward | 原 reward + contact-gated lift progress |
| 环境/sampling | vision/Warp，50% targeted mixture | 完全相同 |
| PPO/网络/学习率 | 冻结的共同配置 | 完全相同 |
| 追加预算 | 3M effective timesteps | 3M effective timesteps |
| train seed | 配对使用相同 seed | 配对使用相同 seed |
| train guide | 0.05 | 0.05 |
| checkpoint rule | 相同候选频率；success→reward→较早 step | 完全相同 |
| formal eval | guide 0.0；101/202/303/404，各 256 回合 | 完全相同 |

资源受限时先做一个 train seed 的配对 pilot，只能标记 Gate 5 `PILOT`。若要写成
“reward 改进在训练随机性下稳定”，预注册至少三个 train seeds；每个 seed 都从
同一输入 checkpoint 产生一对 control/treatment，并分别完成 original/left/hard
评估。不要只给 treatment 多跑 seed，也不要从 pilot 中挑最好的 seed 进入报告。

## 验收标准

- original aggregate `>=95%`，worst seed `>=90%`；
- left `[-0.05,-0.02]` aggregate `>=95%`，worst seed `>=90%`；
- hard bin `[-0.03,-0.02]` aggregate `>=93%`，worst seed `>=85%`；
- guide-free left 的 `reached_no_lift` incidence 少于固定 baseline 55/1,024，
  并另报其占全部失败的比例；
- 所有报告 schema 4、guide probability 0.0、无策略更新。

此外，reward 的因果效果以**配对 treatment − control**报告：每个 train seed 给
三分布成功率差、`reached_no_lift` incidence 差，再汇总跨 seed 均值与范围/区间。
如果只有单 seed，必须写 `pilot; training-seed uncertainty not estimated`，不进行
普遍提升表述。

前三组与历史预注册门槛一致。最后一项是机制指标，不能用百分比下降替代前三组
工程门槛。不得在看见结果后调整阈值或只选有利 seed。

## 最小实验

先只实现并单元测试 reward 函数，不启动训练。构造无接触、接触且高度递增、
高于 lift threshold 三组输入，验证门控、单调性和封顶；再做 100k smoke。
只有测试和 smoke 均通过，才授权一次 control 3M 与一次 treatment 3M 配对运行。
smoke 也应两臂同预算，验证 metric 只在 treatment 出现且公共指标 schema 一致。

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

实现后按五步推进：

1. 单元测试 reward 在无接触时为 0、随高度单调、达到阈值封顶；
2. 从同一 checkpoint 做 control/treatment 各 100k smoke，验证恢复、更新、metric、
   checkpoint、视频；
3. 冻结 reward scale、train seeds 和 checkpoint rule；
4. 对每个预注册 seed 运行 control 3M 与 treatment 3M，不扫事后权重；
5. 对两臂各自选中的 checkpoint 做 original/left/hard guide-free 回归，再跑 left
   trajectory 分类，按 seed 形成配对差。

本课程不预先给出尚未实施的命令名。实现 launcher 时应采用清晰的新模式与
`panda-reach-to-lift-*` artifact root，并补入 `reference/commands.md`。

## 预期结果

预期是“得到可解释结论”，不是预设 PASS。理想结果为 treatment 相对等预算
control 的 left 成功率更高、original 保持、`reached_no_lift` 更少；如果两臂
相近，不能把 treatment 相对训练前起点的变化归因于 reward；若 left 提升但
original 退化，结论是局部收益/整体不接受。

## 常见错误

- 为尽快过线同时增加 reward、继续 targeted sampling、改学习率；
- 让 treatment 多训练 3M，却用没有追加训练的起始 checkpoint 当唯一对照；
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
4. 为什么 control 也必须从相同 checkpoint 继续训练 3M？
5. 为什么新实验必须有新分支和 artifact root？

## 通过标准

代码测试通过、两臂 smoke 完整、同 seed/预算的配对训练完成、三分布评估和轨迹
分类齐全；报告能区分事实、推断和限制，并如实给 PASS/FAIL。单 train seed 只到
PILOT；至少三个预注册 train seeds 且证据齐全才通过 Gate 5 READY。

## Git 节点

建议至少四次提交：

```text
test: specify contact-gated lift progress
feat: add contact-gated lift progress experiment
docs: preregister reach-to-lift evaluation
docs: report reach-to-lift experiment result
```
