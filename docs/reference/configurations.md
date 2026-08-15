# 固定配置与实验差异

## 环境基线

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `ctrl_dt` | 0.05 s | 策略/控制周期 |
| `sim_dt` | 0.005 s | 物理步长 |
| `episode_length` | 200 | 最长 10 s 控制时间 |
| `action_scale` | 0.005 m | 单步归一化位移尺度 |
| `box_init_range` | 0.05 m | 原始 y 均匀范围 |
| `success_threshold` | 0.05 m | 到目标的高度误差阈值 |
| `vision.cam_res` | 64×64 | 单相机 RGB |
| train guide | 0.05 | 仅训练探索辅助 |
| formal-eval guide | 0.0 | 正式评估必须禁用 |

## 视觉 PPO 基线

| 参数 | 官方视觉配置 |
| --- | ---: |
| timesteps | 10,000,000 |
| train/eval envs | 1,024 / 128 |
| batch size | 256 |
| unroll length | 10 |
| minibatches / updates | 8 / 8 |
| discount | 0.97 |
| learning rate | 0.001 |
| entropy cost | 0.01 |
| clipping epsilon | 0.3 |
| CNN channels | 16, 32 |
| hidden layers | 128, 128, 128 |

本仓库 `full` 在 11 GiB GPU 保持 10M 工作量，但使用 512/64/128；应表述为
memory-aware reproduction，而不是 exact parallelism。

## 已完成实验矩阵

| 阶段 | 起点 | 主要变化 | 正式状态 |
| --- | --- | --- | --- |
| Full | 随机初始化 | 10M, 512/64/128 | 未收敛 |
| Fine-tune | full 5,017,600 | 再约 10M, lr 0.0005 | 训练评估收敛 |
| Position analysis | fine-tune final | 仅记录位置 | 发现左侧弱区；历史 guide |
| Left robustness | fine-tune final | 3M, lr 1e-4, 65% 左区 | 部分成功；历史 guide |
| Guide-free diagnosis | robustness 2,007,040 | 仅评估/轨迹字段 | 964/1,024，FAIL |
| Capstone | robustness 2,007,040 | 仅 contact-gated lift progress | 待实施 |

## 正式评估协议

| 项目 | 值 |
| --- | --- |
| deterministic policy | true |
| seeds | 101, 202, 303, 404 |
| episodes per seed | 256 |
| total per distribution | 1,024 |
| guide probability | 0.0 |
| original y | `[-0.05,0.05]` |
| left y | `[-0.05,-0.02]` |
| hard y | `[-0.03,-0.02]` |

任何比较先检查 checkpoint、schema、guide、范围、种子、回合数是否一致。
