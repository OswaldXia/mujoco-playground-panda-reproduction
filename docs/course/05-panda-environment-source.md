# 05｜Panda 环境源码精读

建议时间：8 小时。硬件：Mac 即可。

## 学习目标

- 从注册名追到环境类、配置、XML 和 PPO 网络；
- 逐项解释 reset、observation、action、reward、done；
- 发现训练专用机制并判断它是否应进入正式评估；
- 输出一份可复核的环境审计表。

## 本章知识清单

- **注册与配置**：从环境名追到类、默认参数、XML 和 PPO 网络配置；
- **`reset`**：采样初始方块/目标并创建观察、历史量与 episode 状态；
- **`step`**：动作处理 → IK → 物理 → 奖励/终止 → 下一帧观察；
- **reward progress**：只奖励当前潜势超过本回合历史最佳值的正增量；
- **guide-state**：训练期探索辅助，不是策略行为，正式评估必须关闭；
- **success 与 reward**：成功判定是最终指标，dense reward 只是学习信号。

## 为什么需要源码精读

配置打印只能说明“传入了什么”，不能完整说明环境真正做了什么。本项目最重要
的完整性问题——5% guide-state 也进入评估——正是从 step 源码和轨迹数据交叉
验证发现的。

## 核心知识

`reset` 把方块 x 固定在末端平面附近，y 从区间采样，z 为 0；目标高度固定为
0.20 m。视觉 observation 只有一幅 RGB 字典，不把 `target_pos` 或机器人状态
暴露给策略。`step` 首先处理动作历史和 guide-state，然后做 Cartesian IK、
物理子步、奖励、终止和下一帧渲染。

奖励分三层：靠近/目标/无碰撞等 dense potential；抬升和成功 sparse bonus；
最后只保留超过历史最好 potential 的正向 progress。成功在视觉模式只比较
方块与目标的高度差，小于 0.05 m 即结束。`reward/success` 是评估主指标。

训练期 guide-state 以 5% 概率把新回合状态替换为 `picked` keyframe，帮助命中
稀疏成功奖励。它是探索辅助，不是策略动作结果；正式评估必须强制为 0。

## 最小实验

运行 [`labs/05_panda_inspect.py`](../labs/05_panda_inspect.py)。默认只加载
state observation，打印关键配置、action size、obs shape、初始方块位置，并做
一次零动作 step。第一次 JIT 可较慢。

## 源码定位

- `registry.py` 与 manipulation `__init__.py`：名称注册；
- `pick_cartesian.py::default_config/reset/step/_get_success/_move_tip`；
- `manipulation_params.py::brax_vision_ppo_config`；
- `evaluate_panda_checkpoint.py`：正式评估如何覆盖 guide probability。

## 运行前预测

写下 observation 类型、action shape、200 步对应秒数、方块 y 范围、成功高度
条件。再预测零动作一步是否会成功。

## 操作

```bash
source .venv/bin/activate
python docs/labs/05_panda_inspect.py
rg -n "guide_swap_probability|reward/success|box_init|success_threshold" \
  mujoco_playground reproduction
```

## 预期结果

action size 为 3；状态 observation 是一维数组；方块 y 在约 `[-0.05,0.05]`；
guide probability 默认 0.05。正式 evaluator 的环境覆盖值为 0.0。

## 常见错误

- 只看 `default_config`，忽略 launcher 的 overrides；
- 把 `no_box_collision` 写入 raw reward 后仍忘记它是否进入已缩放字典；
- 混淆训练随机策略、确定性 replay 和独立评估；
- 因为目标在画面中不可见，就认为 success 不可计算；环境有特权仿真状态。

## 修改练习

建立 `notes/environment-audit.md`，列出每个 observation、action、reward、done、
reset 随机项的源码位置、单位、策略是否可见。指出 `guide_swap_probability` 在
训练和评估中的期望值。

## 自测

1. 三维 action 各代表什么？
2. 视觉策略为何能在看不到显式目标坐标时学习固定高度任务？
3. `lifted` 与 `success` 的高度条件有什么不同？
4. guide-state 为什么不是模仿学习？为什么仍会污染评估？

## 通过标准

能从环境名追到全部四类源码；能不看文档画出 reset→obs→policy→action→IK→
physics→reward/done 的数据流；审计表中训练/评估机制没有混淆。

## Git 节点

```bash
git add notes/environment-audit.md
git commit -m "docs: audit Panda environment data flow"
```
