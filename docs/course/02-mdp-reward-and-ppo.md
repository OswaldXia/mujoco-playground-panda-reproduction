# 02｜MDP、奖励与 PPO

建议时间：10 小时。硬件：Mac 即可。

## 学习目标

- 把 Panda 任务写成 MDP；
- 区分策略、价值、回报、优势和 on-policy 数据；
- 解释 PPO clipped objective 的作用与局限；
- 判断 reward、success metric 和真实任务完成之间的差别。

## 为什么需要这些概念

这个项目属于强化学习，不是模仿学习：策略通过与环境交互得到奖励，没有专家
动作标签。PPO 的日志是训练代理指标；最后结论仍必须来自冻结策略的独立评估。

## 核心知识

MDP 写作 `(S, A, P, R, gamma)`。本项目的视觉策略状态近似为 64×64 RGB，
动作是连续 y/z 加离散化夹爪语义，转移由 MuJoCo 物理决定，奖励由靠近、目标、
碰撞、抬升和成功项组成，`gamma=0.97`。

折扣回报 `G_t = r_t + gamma r_(t+1) + ...`。价值网络估计未来回报；优势
`A_t` 表示该动作相对当前平均选择更好还是更差。PPO 用新旧策略概率比
`r_t(theta)`，优化

\[
L^{CLIP}=E[\min(r_t A_t,\; clip(r_t,1-\epsilon,1+\epsilon)A_t)].
\]

裁剪限制单批数据上的过大更新，但不保证单调改进，也不能修复错误奖励或评估
泄漏。连续动作策略通常输出分布参数，训练时采样，正式评估使用确定性动作。

本环境还有“reward progress”：只奖励当前总潜势超过历史最好值的增量。这能
减少来回刷分，但意味着日志中的逐步 reward 不是原始各项之和。

## 最小实验

运行 [`labs/02_ppo_clipping.py`](../labs/02_ppo_clipping.py)，观察正/负优势下，
概率比超出 `[0.8, 1.2]` 后目标如何变化。随后把 epsilon 改为 0.3，与本项目
视觉 PPO 配置一致。

## 源码定位

- `pick_cartesian.py::step`：dense、sparse、progress reward 与 done；
- `manipulation_params.py::brax_vision_ppo_config`：PPO 超参数；
- `learning/train_jax_ppo.py`：环境包装、网络和训练入口；
- PPO 原论文：<https://arxiv.org/abs/1707.06347>。

## 运行前预测

对优势 `+1` 和 `-1`，分别预测概率比为 0.5、1.0、1.5 时 clipped surrogate
取哪个分支。写下“更大的奖励”是否必然表示更高成功率。

## 操作

```bash
python3 docs/labs/02_ppo_clipping.py
rg -n "reward_scaling|clipping_epsilon|discounting|entropy_cost" \
  mujoco_playground/config/manipulation_params.py
```

## 预期结果

有利动作的概率比上升超过上界后收益被截住；不利动作的概率比下降超过下界时
惩罚也被限制在保守目标内。epsilon 变大允许单次更新偏离更远。

## 常见错误

- 把 PPO 理解成每步选择最高 reward 的搜索；
- 把 value network 当作部署时必需的控制器；
- 用训练评估种子反复挑 checkpoint 后仍称其为 held-out；
- 奖励上升就声称抓取成功，不检查 `reward/success` 和视频。

## 修改练习

为 lab 增加四个 ratio/advantage 组合，标注哪些样本真正被裁剪。再写 150 字说明
为什么“接触奖励”无法解决当前所有轨迹都曾双指接触的问题。

## 自测

1. 为什么本项目是 on-policy 强化学习而不是模仿学习？
2. entropy cost 在训练中提供什么作用？
3. PPO 裁剪能否阻止错误成功条件？
4. reward shaping 为什么可能改变最优行为？

## 通过标准

能根据 ratio 和 advantage 手算 clipped objective；能列出本环境 MDP 的五个
部分；能解释 reward、metric、acceptance 三层含义。

## Git 节点

```bash
git add notes/02-mdp-ppo.md
git commit -m "docs: derive Panda MDP and PPO objective"
```
