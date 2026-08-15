# 03｜MuJoCo 模型、状态与控制

建议时间：8 小时。硬件：Mac 即可。

## 学习目标

- 区分 MJCF、`MjModel`、`MjData` 与 MJX 对象；
- 理解 body、joint、geom、site、sensor、actuator 和 keyframe；
- 能读取 `qpos`、`qvel`、`ctrl`、`xpos` 与 `sensordata`；
- 用最小模型完成 reset、control、step 和状态检查。

## 本章知识清单

- **MJCF**：描述机器人、场景、碰撞、传感器与执行器的 XML 源模型；
- **`MjModel` 与 `MjData`**：前者保存编译后的结构，后者保存每次运行的可变状态；
- **body/joint/geom/site**：分别组织刚体、自由度、碰撞形状和测量/目标标记；
- **`qpos/qvel/ctrl`**：广义位置、广义速度和 actuator 输入；
- **物理步与控制步**：`ctrl_dt/sim_dt=10`，一个动作会推进十个物理子步。

## 为什么需要这些概念

策略输出最终必须变成 actuator control，成功与接触则来自仿真状态/传感器。
不了解模型和数据的边界，就无法判断问题在 XML、控制器、物理、观察还是策略。

## 核心知识

MJCF 是可维护的模型源文件；加载并编译后成为只读结构为主的 `MjModel`；每次
运行的可变状态位于 `MjData`。`qpos` 是广义位置，`qvel` 是广义速度，`ctrl`
是 actuator 输入。`xpos` 是前向计算得到的世界位置。`mj_step` 用模型和当前
数据推进一个仿真步。

Panda XML 通过 include 组合机器人和场景。geom 参与形状/碰撞，site 常作为
测量或目标标记，sensor 输出进入 `sensordata`，keyframe 保存命名姿态。环境
使用 `low_home` 初始化，并用 `picked` 作为训练期 guide-state 源。

`sim_dt=0.005`，`ctrl_dt=0.05`，所以一个环境控制步包含 10 个物理子步。提高
控制频率和减小物理步长不是同一件事。

## 最小实验

在已安装 MuJoCo 的 `.venv` 中运行
[`labs/03_mujoco_state.py`](../labs/03_mujoco_state.py)。它创建一个带滑动关节
和位置 actuator 的最小模型，打印维度并推进 20 步。

## 源码定位

- `mjx_single_cube_camera.xml`：场景和相机；
- `sensor.xml`：触碰/碰撞传感器；
- `panda.py` 与 `pick.py`：模型资产和基础任务；
- MuJoCo 官方建模指南：<https://mujoco.readthedocs.io/en/stable/modeling.html>；
- 官方仿真说明：<https://mujoco.readthedocs.io/en/stable/programming/simulation.html>。

## 运行前预测

预测最小模型的 `nq`、`nv`、`nu`。把 actuator 的控制目标从正值改为负值前，
先预测 `qpos` 最终方向。

## 操作

```bash
source .venv/bin/activate
python docs/labs/03_mujoco_state.py
rg -n "camera|sensor|actuator|keyframe|box" \
  mujoco_playground/_src/manipulation/franka_emika_panda/xmls
```

## 预期结果

最小模型有一个位置维度、一个速度维度和一个控制输入，关节位置朝控制目标移动；
输出保持有限。Panda XML 搜索能定位相机、接触传感器、执行器和命名 keyframe。

## 常见错误

- 修改 `MjData` 后忘记调用 forward，读取了过期的 `xpos`；
- 认为一个 body 必然对应一个 joint；
- 把 sensor 的索引当成 `sensordata` 地址；
- 用渲染帧率推断物理步长。

## 修改练习

给最小模型增加 joint position sensor，打印 sensor 地址和值；再把 timestep 减半，
保持总模拟时间相同，比较需要的步数。

## 自测

1. `MjModel` 和 `MjData` 的职责分别是什么？
2. `qpos` 为什么不总与 `qvel` 等长？
3. 200 个环境步对应多少秒控制时间、多少个物理步？

## 通过标准

能从 XML 找到一个 actuator 对应的 joint，并从环境代码追到 `ctrl`；能解释
`data.xpos` 与 `data.qpos` 的不同来源。

## Git 节点

```bash
git add notes/03-mujoco-state.md
git commit -m "docs: inspect MuJoCo model state and control"
```
