# 术语表

| 术语 | 本课程中的含义 |
| --- | --- |
| action | 策略每个控制步输出的 y、z、夹爪三个量 |
| actuator / ctrl | MuJoCo 执行器及写入 `MjData.ctrl` 的控制输入 |
| advantage | 某动作相对状态下平均选择的估计收益 |
| artifact | 日志、checkpoint、视频、完整报告等本机运行产物 |
| checkpoint | 可恢复的策略、价值和观察归一化等参数快照 |
| deterministic policy | 评估时选分布代表动作而非继续随机采样 |
| domain randomization | 随机化视觉/物理参数以扩大训练分布 |
| episode | 从 reset 到成功、失败或超时的一条交互序列 |
| guide-state | 本环境训练期把少量新回合替换到 picked 状态的探索辅助 |
| held-out seed | 未用于策略更新或 checkpoint 选择的评估随机种子 |
| JIT | 将数组程序即时编译到目标设备；首次调用通常较慢 |
| MJCF | MuJoCo 的 XML 模型描述格式 |
| MJX | 允许 MuJoCo 模型/数据在 JAX 设备数组上批处理的接口 |
| MJX-Warp | 面向 NVIDIA GPU 优化的 MuJoCo 实现与批量渲染路径 |
| MDP | 状态、动作、转移、奖励、折扣构成的决策过程 |
| on-policy | 使用当前附近策略采集的数据更新当前策略 |
| PPO | 使用 clipped surrogate 控制策略更新幅度的策略梯度算法 |
| PyTree | JAX 可遍历和变换的嵌套数组容器结构 |
| regression | 改进弱区时确认原分布性能没有不可接受退化 |
| reward shaping | 添加中间奖励以改善学习信号，但可能改变行为偏好 |
| rollout | 用固定策略运行得到的一条或一批轨迹 |
| seed | 控制伪随机序列起点；不是覆盖所有随机性的证明 |
| smoke test | 低成本确认管线能运行，不证明最终性能 |
| success metric | 环境定义的成功指标，本项目为 `reward/success` |
| Wilson interval | 二项成功率置信区间，小样本时优于简单正态近似 |
