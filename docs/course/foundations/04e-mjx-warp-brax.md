# 04E｜MuJoCo、MJX、Warp 与 Brax 的职责边界

建议 60–90 分钟。前置：04A–04D。

## 同一项目中的五层

| 层 | 主要职责 | 典型问题 |
| --- | --- | --- |
| MuJoCo/MJCF | 模型语义、CPU 物理、原生渲染 | 关节/相机/接触是否正确 |
| MJX | JAX 兼容的模型与 data、批量仿真接口 | state tree/shape 是否正确 |
| MJX-Warp | NVIDIA 上的物理与批量视觉内核 | kernel、contact capacity、显存 |
| 视觉网络 | 将 RGB 编码为 policy/value 表示 | 图像 shape、activation 显存 |
| Brax PPO | rollout、GAE、minibatch 与更新 | 训练预算、on-policy 协议 |

Playground 环境把这些层连接起来。说“MuJoCo 在训练策略”不准确；MuJoCo 提供
物理，Brax 实现 PPO，JAX/XLA 执行变换后的数值程序，Warp 提供特定 GPU 内核。

## 一次视觉 step

MJCF 先加载为模型；环境 state 在设备上批量存在；动作经 IK/控制后推进多个物理
子步；renderer 为每个 world 生成 RGB；Brax wrapper 记录 transition、done 与
autoreset。训练时 CNN 和 value/policy backward 又增加大量中间 activation。

`vision_config.nworld` 在 renderer 创建时固定。`num_envs`、`num_eval_envs`、
`batch_size` 与 `nworld` 必须兼容，且共同决定峰值显存。只把一个值减半可能产生
shape 不一致或仍在另一阶段 OOM，因此仓库用成组 profile。

## 故障按层定位

- XML/CPU smoke 失败：先看 MuJoCo model、路径、状态；
- state 正常但 RGB probe 失败：看 camera、renderer、Warp；
- reset/eval 正常但 update OOM：看 CNN backward、batch 与 XLA 峰值；
- 训练正常但统计异常：看 evaluator、成功定义、reset/guide/autoreset；
- policy 不抬升：再看 reward、动作/接触和训练，而非先归因 GPU。

## 暂停并分类

“`RESOURCE_EXHAUSTED` 在第一轮 PPO update 请求 5.09 GiB”优先属于训练激活/批量
峰值，不是 MJCF 语义错误；“相机全黑但 state finite”优先检查渲染层；“step 1
已成功约 5%”优先检查评估初始状态分布。

## 通过标准

能把一个错误先定位到上述层之一，并给出最小隔离实验；能解释 Mac 一世界 RGB
通过为何不等于 Linux 10M 视觉 PPO 可行。
