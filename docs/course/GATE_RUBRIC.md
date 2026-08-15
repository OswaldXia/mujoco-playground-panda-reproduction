# 五道能力门：客观评分表

每道 Gate 只取最低满足等级。`READY` 代表可以进入下一阶段；`PRACTICED` 代表已用
课程数据练习，但不能冒充自己的正式证据。口头回答与机器可读产物都需要，二者
缺一不通过。

## Gate 1｜理论与环境

| 等级 | 可观察证据 |
| --- | --- |
| NOT READY | 不能独立写出 MDP；混淆像素与世界坐标或 FK/IK |
| PRACTICED | 三个 starter 之外的 01/02 练习通过，但源码链仍依赖答案 |
| READY | 三维变换与 advantage starter 通过；能从源码口述 observation→action→IK→reward→done，并提交 source audit |

允许查看文档，不允许照读答案。审阅者随机改变一个 ratio、done 或坐标变换后仍能
正确推导，才算 READY。

## Gate 2｜计算栈

| 等级 | 可观察证据 |
| --- | --- |
| NOT READY | 把 MuJoCo、Warp、Brax 的职责混在一起；不能预测 batch shape |
| PRACTICED | JAX 演示与 starter 通过 |
| READY | 能手画 XML→MJX/Warp→RGB→CNN→PPO；正确解释 jit 冷/热计时、vmap/scan/PyTree、显存峰值，并提交 shape audit |

## Gate 3｜复现

| 等级 | 可观察证据 |
| --- | --- |
| NOT READY | 只有安装成功或单张截图 |
| LOCAL READY | Mac manifest、state smoke、RGB probe 均通过且能陈述边界 |
| READY | 再加 Linux GPU smoke；manifest 含 commit/dirty/version/device，产物路径和失败日志可追溯 |

Mac-only 学习者在 `LOCAL READY` 停留是正确结论，不扣减已掌握的理论能力。

## Gate 4｜独立评估

| 等级 | 可观察证据 |
| --- | --- |
| NOT READY | 只看视频或训练期 64 回合数字 |
| PRACTICED | 三个 offline lab 通过，能解释精选轨迹不能估计成功率 |
| READY | 自己冻结 checkpoint，按预注册协议运行 4×256；能从 episode records 重算 aggregate/per-seed/Wilson/失败类别；schema、guide 与 SHA-256 齐全 |

没有 checkpoint/GPU 时最多到 PRACTICED。课程自带 compact evidence 只能证明学习者
会分析，不能证明其复现了策略性能。

## Gate 5｜毕业实验

| 等级 | 可观察证据 |
| --- | --- |
| NOT READY | 只训练 treatment，拿它与较早 baseline 比；或看结果后改门槛 |
| PILOT | 同一起点完成一对等预算 control/treatment，协议固定，但只有一个 train seed |
| READY | 至少 3 个预注册 train seeds 的配对 control/treatment；每臂等 3M steps、同 checkpoint rule；每个 arm/seed 完成三分布评估；报告跨 seed effect 与失败机制，无论结果正负均归档 |

资源不足时可以把 PILOT 作为求职中的工程试验，但必须明确“单种子 pilot，尚未证明
训练随机性下稳定提升”。不能把多跑 treatment 的额外计算量解释为 reward 因果效应。

## 审阅记录

| Gate | 日期 | 等级 | commit / artifact | 审阅者复问与结果 |
| --- | --- | --- | --- | --- |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
