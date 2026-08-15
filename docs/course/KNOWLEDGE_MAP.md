# Panda 课程知识地图

这张表回答“本课程到底要学会哪些知识”。它不是目录，也不是需要背诵的术语表。
每进入一章，先看对应行；学完后遮住“一句话解释”，尝试用自己的话复述，并用
最后一列的产物证明自己会用。

标记自己的掌握状态：`🔴 没听过`、`🟡 能解释但不会用`、`🟢 能独立解释并验证`。
只有达到绿色，才把该知识点计为掌握。

## 全课程主线

1. **任务是什么**：坐标、动作、状态、奖励和成功条件；
2. **系统怎么运行**：MuJoCo 物理、JAX 批处理、视觉 PPO 更新；
3. **结果是否可信**：冻结 checkpoint、独立评估、统计区间和证据链；
4. **怎样改进**：定位轨迹瓶颈、提出可证伪假设、做等计算量单变量对照。

## 逐章知识点

| 章 | 必须会说清的知识点 | 一句话解释 | 项目中的落点 | 掌握证据 |
| --- | --- | --- | --- | --- |
| 00 | 基线、适配、实验 | 复现原方案、为硬件做适配、提出新改动是三种不同主张 | 分支、commit、manifest、`results/` | 一页复现契约 |
| 00 | 代码、产物、证据 | 代码能重跑；大产物留本机；小证据进入 Git | `reproduction/`、`artifacts/`、`results/` | 能给任一结论定位 JSON 和 commit |
| 01 | 坐标系与坐标 | 同一个几何点在不同坐标系中有不同数字 | world/base/tip/camera/box | 画出坐标链并标注表达系 |
| 01 | 旋转、平移、齐次变换 | 齐次矩阵把旋转和平移放进一个可组合运算 | `T_world_base @ T_base_tip` | 手算并用 Notebook 验证一个点 |
| 01 | FK、IK、笛卡尔动作 | FK 由关节求末端，IK 由末端目标求关节 | `_move_tip`、`action_scale` | 解释 3 维动作如何变成控制量 |
| 02 | MDP 与强化学习闭环 | 策略行动、环境转移并给奖励，数据来自交互而非专家标签 | observation/action/reward/done | 写出本任务的 S/A/P/R/终止 |
| 02 | reward、return、value、advantage | 即时分数、未来累计、未来估计、动作相对好坏是四个不同量 | reward、critic、GAE | 用两步轨迹手算 return/advantage |
| 02 | 概率比与 PPO clipping | PPO 限制新策略对旧数据的概率变化，不能修复错误奖励 | Brax PPO 配置 | 推导正负优势各一个样本 |
| 03 | MJCF、MjModel、MjData | XML 是源模型，Model 是编译结构，Data 是运行时状态 | Panda XML、`qpos/qvel/ctrl` | 最小模型 reset/control/step 报告 |
| 03 | 物理步与控制步 | 一个控制命令可覆盖多个更小的物理积分步 | `sim_dt=0.005`、`ctrl_dt=0.05` | 算出 action repeat 并解释影响 |
| 04 | 纯函数、显式 PRNG、PyTree | JAX 把状态和随机性显式放进可变换的数据结构 | env state、keys、params | 不复用 key，画出数据树 |
| 04 | `jit`、`vmap`、`scan` | 分别表达编译、环境批维和时间递推 | 并行 rollout 与 PPO | 写出 `[time, env, feature]` 来源 |
| 04 | MuJoCo/MJX/Warp/Brax 边界 | 物理模型、设备批处理、NVIDIA 实现和训练算法各负其责 | 训练数据流 | 从 XML 追到 gradient update |
| 05 | reset/step 数据流 | 环境定义初态、观察、动作转换、物理、奖励和终止顺序 | `pick_cartesian.py` | 手画一次 step 的调用链 |
| 05 | reward progress 与 guide-state | 前者只给潜势新高，后者是训练探索辅助 | reward history、`picked` keyframe | 判断哪些机制不能进入正式评估 |
| 06 | 分层 smoke | 从导入、状态到图像逐层扩大验证范围 | Mac scripts、manifest | 保存 JSON 与渲染图并写能力边界 |
| 06 | 平台绑定环境 | `.venv` 含解释器和本机二进制，不能跨系统复制 | macOS/Linux 各自 `.venv` | 能解释 backend 与安装约束 |
| 07 | smoke/full/official | 管线验证、完整工作量、官方并行度是三种不同实验 | 100k/10M/profile | 训练前写明主张和验收线 |
| 07 | 显存与 JIT | reset 能跑不代表首次梯度更新装得下全部中间张量 | `num_envs`、batch、eval envs | 解释 OOM 并选择适配 profile |
| 08 | checkpoint 选择 | 选择规则必须先定义，不能默认最后一步或偷看 held-out | summary、selector | 给出选中 step 的可复核依据 |
| 08 | fine-tune 与 resume | 只恢复参数而重置优化器不是无缝续训 | checkpoint loader | 写清恢复了什么、未恢复什么 |
| 09 | 点估计、Wilson 区间、门槛 | 观察比例、不确定性范围、工程判定回答三个不同问题 | 1,024 回合报告 | 重算 aggregate、区间、worst seed |
| 09 | 随机样本与精选轨迹 | 精选视频可解释行为，不能估计总体成功率 | compact report vs fixture | 能指出每类证据可支持的结论 |
| 10 | 分层率与分母 | 失败数量没有成功样本作分母时不能证明某区域更难 | y-position bins | 重算条件成功率并注明探索性 |
| 10 | 轨迹失败分类 | 互斥、完备的阶段分类把失败定位到行为链节点 | `reached_no_lift` 等 | 验证分类守恒并画事件时间线 |
| 11 | 评估泄漏与 effective config | 训练辅助进入评估会让正常运行的数字仍然失真 | guide probability、schema | 源码/配置/轨迹三角审计 |
| 11 | 历史结论降级 | 协议错误要保留历史证据、降级主张并重新采集 | schema-3 → schema-4 | 写出异常到重采集的完整链 |
| 12 | 可证伪机制假设 | 假设必须预测干预后哪项指标改变，以及什么结果会否定它 | stable-lift reward | 预注册唯一改动与否定条件 |
| 12 | 等计算量单变量对照 | control 与 treatment 从同一 checkpoint 获得相同训练预算 | paired seeds、3M/3M | 完整对照表和 PASS/FAIL 报告 |

## 每章的学习动作

每个知识点都用同一套四步闭环，不再把“读完”当作掌握：

1. **认名词**：先读知识清单和概念卡，能指出每个符号或术语指什么；
2. **做预测**：不运行代码，先写出数值、shape、行为或判定；
3. **看证据**：运行最小实验，对比预测与输出，解释差异；
4. **脱稿复述**：关闭正文，用一句话解释概念，再指出项目代码和证据位置。

若只能复述定义但不能完成第 3、4 步，状态仍是黄色。若连知识点名称都说不出，
不要重读整章；只回到该章的“本章知识清单”和对应微课。

