# Reach-to-lift 毕业实验报告

## 1. 结论摘要

- 决策：PASS / FAIL / PARTIAL（按预注册规则）
- 最重要的一个数字：
- 最重要的失败机制变化：
- 适用范围与不适用范围：

## 2. 预注册

- 分支与起始 commit：
- 假设：
- 唯一改动：
- reward 公式与固定 scale：
- 起始 checkpoint 与 SHA-256：
- 训练预算/seed/PPO/sampling：
- formal eval seeds/episodes/guide probability：
- original/left/hard/failure-class 验收标准：
- 会否定假设的结果：

## 3. 实现与测试

- 代码位置：
- 无接触、高度单调、上界测试：
- 100k smoke 结果与产物：
- 与对照相比唯一变化的 diff：

## 4. 正式结果

| 分布 | 成功/回合 | 成功率 | Wilson 95% | worst seed | 门槛 | 决策 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| Original | | | | | | |
| Left | | | | | | |
| Hard | | | | | | |

## 5. 机制验证

| Failure class | Baseline | New | 变化 |
| --- | ---: | ---: | ---: |
| reached_no_lift | 55/1,024 episodes; 55/60 failures | | |
| lifted_then_dropped | 4/60 | | |
| lifted_timeout | 1/60 | | |

补充 reach-to-lift latency、接触率和 aperture；总数变化时同时给出分子和分母。

## 6. 失败案例

至少展示一个成功、一个 dominant failure、一个新副作用（若有），写 episode id、
初始位置、关键事件 step 与视频路径。

## 7. 限制与下一步

- 模拟与真实硬件边界：
- 统计/分布限制：
- 如果只允许再做一个实验：

## 8. 可复现材料

- commit / manifest / console log：
- checkpoints / TensorBoard / videos：
- evaluation JSON / CSV / plots：
- archive SHA-256：
