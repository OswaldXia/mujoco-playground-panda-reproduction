# 01｜坐标、姿态与运动学

建议时间：8 小时。硬件：Mac 即可。

本章不是一次读完的摘要。按顺序完成四节微课，每节都先做“暂停并预测”：

1. [`01A 向量、点与坐标系`](foundations/01a-vectors-and-frames.md)（1h）；
2. [`01B 齐次变换与组合`](foundations/01b-homogeneous-transforms.md)（1h）；
3. [`坐标系交互 Notebook`](../notebooks/01_frames_and_transforms.ipynb)（1–1.5h）；
4. [`01C FK、雅可比与 IK`](foundations/01c-fk-ik.md)（1.5h）；
5. [`01D Panda 笛卡尔控制`](foundations/01d-panda-cartesian-control.md)（1h）；
6. starter 实验、源码审计与复盘（2h）。

## 学习目标

- 理解世界、机器人基座、末端、相机和物体坐标系；
- 能用齐次变换组合旋转与平移；
- 区分正运动学、逆运动学与动力学；
- 解释三维动作为什么控制 `y`、`z` 和夹爪，而非三个关节。

## 本章知识清单

- **点、向量与坐标系**：点有位置，方向向量不受平移；数字必须注明在哪个系表达；
- **旋转矩阵 `R`**：改变表达方向，满足 `R.T @ R = I` 且行列式为 1；
- **齐次变换 `T`**：在 4×4 矩阵中统一旋转和平移，并可沿坐标链组合；
- **FK 与 IK**：FK 由关节角求末端姿态，IK 由末端目标反求关节控制；
- **笛卡尔动作**：策略输出末端 y/z 增量与夹爪命令，再由 IK 变成关节控制。

## 为什么需要这些概念

策略输出的不是“抓住方块”，而是每个控制周期的末端增量与夹爪命令。错误的
轴、单位或坐标系会表现成看似随机的策略失败。视觉图像是相机坐标系的投影，
碰撞和成功条件却在仿真世界坐标中计算。

## 核心知识

点从局部系 B 表达到世界系 A：

$$
{}^A p = {}^A R_B {}^B p + {}^A t_B,
\qquad
{}^A T_B = \begin{bmatrix}R&t\\0&1\end{bmatrix}.
$$

连续变换按路径相乘：`T_world_tip = T_world_base @ T_base_tip`。旋转矩阵应满足
`R.T @ R = I` 且行列式为 1。正运动学由关节角求末端姿态；逆运动学由目标
末端姿态求关节控制，本环境的 `_move_tip` 调用 Panda IK，并在无解时保留旧
控制。动力学则继续考虑质量、速度、力和接触。

环境把动作扩展为 `[x, y, z, gripper]`，其中 `x` 增量固定为 0，所以策略的
三个输出实际是 `y`、`z`、夹爪。位移乘 `action_scale=0.005`，即归一化动作
1.0 对应单个控制步 5 mm。

## 最小实验

先通过统一入口打开
[`01_frames_and_transforms.ipynb`](../notebooks/01_frames_and_transforms.ipynb)，
完成预测、可视化、自测与反思。Notebook 从空内核可完整执行，但不要跳过纸面预测。

运行 [`labs/01_transform_2d.py`](../labs/01_transform_2d.py)。先手算一个点旋转
90°再平移的结果，再让脚本比较“先旋转后平移”和“先平移后旋转”。

演示通过后，补全
[`01_transform_3d_exercise.py`](../labs/starter/01_transform_3d_exercise.py)。starter
失败是预期起点；按分级提示完成，最后才看
[`参考实现`](../solutions/labs/01_transform_3d_solution.py)。

## 源码定位

- `pick_cartesian.py::_post_init`：从关节控制计算初始末端变换；
- `pick_cartesian.py::_move_tip`：动作缩放、工作空间裁剪与逆运动学；
- `panda_kinematics.py`：Panda 正/逆运动学实现；
- `mjx_single_cube_camera.xml`：相机、机器人、方块和目标的模型关系。

## 运行前预测

预测 `[1, 0]` 旋转 90°再平移 `[2, 1]` 的结果，以及改变组合顺序后的结果。
解释为什么变换乘法不可交换。

## 操作

```bash
source .venv/bin/activate
./reproduction/start_course_notebooks.sh  # 新终端启动；完成后 Ctrl-C 关闭
python docs/labs/01_transform_2d.py
python docs/labs/starter/01_transform_3d_exercise.py
rg -n "increment|action_scale|compute_franka_ik|new_tip_pos" mujoco_playground/_src/manipulation/franka_emika_panda/pick_cartesian.py
```

若没有 `rg`，将最后一条替换为 `grep -RIn "action_scale\|compute_franka_ik"`
加同一文件路径。

## 预期结果

脚本输出 `[2, 2]` 与 `[-1, 3]` 两种不同结果，并验证旋转矩阵正交。源码显示
动作 0/1 两维进入增量的 y/z，动作 2 控制夹爪。

## 常见错误

- 把角度直接传给要求弧度的函数；
- 混淆坐标系中的点与坐标系自身的位置；
- 把 `qpos` 的长度理解为自由度数；含四元数时二者可能不同；
- 把运动学无解当成 PPO 算法问题。

## 修改练习

把 lab 中的点换成两个点组成的“夹爪”，再增加一个 45°中间坐标系；写出
组合矩阵并用断言验证。

## 自测

1. 正运动学、逆运动学、动力学分别回答什么问题？
2. `action_scale=0.005` 与 `ctrl_dt=0.05` 分别代表什么？
3. 为什么相机看到的“左”不一定等于世界坐标负 y？

## 通过标准

不用查看答案即可手算两次二维齐次变换，能从 `_move_tip` 说明动作到关节控制
的路径，并能指出 IK 失败的处理方式。

## Git 节点

```bash
git add notes/01-frames.md
git commit -m "docs: explain Panda frames and Cartesian control"
```
