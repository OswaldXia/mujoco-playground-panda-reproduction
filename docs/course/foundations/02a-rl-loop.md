# 02A｜强化学习闭环与 MDP

建议 60–90 分钟。

## 先区分三种学习方式

监督学习拿到输入与目标标签；模仿学习的标签通常是专家动作；强化学习让策略与
环境交互，以累积奖励评价动作序列。本项目没有“这张图应该输出哪个动作”的专家
数据，PPO 用 policy 自己采集的 rollout 更新，因此属于 on-policy 强化学习。

guide-state 会让少量训练回合从较容易的 picked 状态开始，但它没有提供专家动作，
所以不是模仿学习。不过它改变了初始状态分布；若泄漏到正式评估，会高估标准 reset
下的能力。

## MDP 五元组

MDP 写作 `(S,A,P,R,gamma)`：状态、动作、转移、奖励、折扣。每一步：

1. 环境给 observation `o_t`；
2. policy 按 `pi(a_t|o_t)` 选择动作；
3. 物理转移到新状态；
4. 环境返回 `r_t` 与 `done_t`；
5. policy 用许多完整或截断的轨迹更新。

严格说视觉单帧未包含速度与接触历史，所以 observation 不一定满足完整 Markov
性。本配置的 `action_history_length=1` 提供有限动作历史，仍应称为部分可观测的
近似，而不是把 RGB 与仿真真状态混为一谈。

## Panda MDP 草图

- `S`：仿真 qpos/qvel、方块姿态、接触等隐藏真状态；
- `O`：64×64 RGB 和配置允许的历史；
- `A`：y/z 末端增量与夹爪；
- `P`：MuJoCo/MJX 接触动力学加 autoreset；
- `R`：接近、目标、碰撞、抬升、成功等 shaping；
- `gamma=0.97`：越远的奖励权重越低。

episode 最长 200 control steps，即 10 秒仿真时间。成功、超时或非法状态可结束
回合；Brax 包装器可能随即 autoreset，因此记录 terminal state 必须在 reset 覆盖
之前完成。

## 暂停并判断

1. 固定 checkpoint 做 1,024 回合评估时还在学习吗？
2. 环境返回奖励是否等于“业务成功”？
3. 使用自己刚产生的数据更新一次后，这批数据还严格来自新策略吗？

答案：没有，只在推理；不等于，成功是独立 metric/判定；不再是，因此 PPO 不会
无限复用旧 rollout。

## 通过标准

不用照抄即可为本项目写出 S/O/A/P/R/gamma/done；能解释它为何是 RL、为何是
on-policy、为何单帧视觉只是 Markov 近似。
