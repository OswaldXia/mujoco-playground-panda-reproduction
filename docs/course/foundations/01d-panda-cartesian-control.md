# 01D｜Panda 笛卡尔控制的数据流

建议 60–90 分钟。前置：01A–01C。

## 策略实际控制什么

`PandaPickCubeCartesian` 的策略输出长度为 3，但它不是三个关节角。环境将其解释
为：tip 的 y 增量、tip 的 z 增量、夹爪命令。x 增量被固定为 0，以降低学习难度。
归一化平移动作乘 `action_scale=0.005`，所以幅值 1 在一个 control step 中最多
请求 5 mm 位移。

`ctrl_dt=0.05 s` 表示策略每 50 ms 给一次命令；`sim_dt=0.005 s` 表示物理积分
每 5 ms 一次。因此每个策略动作通常保持 10 个物理子步。动作缩放是“每控制步
位移”，`ctrl_dt` 是时间，不能混为速度单位。

```mermaid
flowchart LR
  O["64×64 RGB"] --> P["policy: 3 values"]
  P --> S["scale y/z; map gripper"]
  S --> C["workspace clipping"]
  C --> I["Panda IK"]
  I --> U["joint actuator controls"]
  U --> M["10 MuJoCo substeps"]
  M --> O
```

## 夹爪维度为何特殊

前两维是连续位置增量；第三维最终映射为开/关语义。策略网络可能输出连续值，
环境却通过符号或阈值生成夹爪 target。分析 policy output 时，必须追到环境映射，
不能仅从网络最后一层推断动作物理含义。

## 相机左与世界 y

固定 x 不代表图像中水平方向固定不动。相机外参将 world 轴旋转到 camera 轴，再由
内参投影到像素。因此针对 initial cube y 的失败分层是世界位置分析；针对像素列
的分析是视觉输入分析，两者相关但不等价。

## 暂停并计算

1. 连续 20 个 control step 都请求 y=+1，忽略裁剪时累计请求多少米？
2. 这 20 步跨越多少秒、多少 physics step？
3. 如果第 5 步已到边界，剩余命令还会产生相同位移吗？

答案：0.1 m；1 s、200 个物理步；不会，目标会被工作空间裁剪。

## 与失败分类连接

`never_reached` 可能来自视觉定位或移动；`reached_no_lift` 已说明接近阶段成功，
重点转向夹爪时序、接触、上提动作或奖励；`lifted_then_dropped` 则要求查看接触
保持与运动平滑性。坐标与动作链为后续“机制级分类”提供了语言。

## 通过标准

从三维 policy output 口头追踪到七关节控制和物理子步；正确算出上面的时间/位移；
解释为何 world-y 分层不能直接叫作“画面左侧分层”。
