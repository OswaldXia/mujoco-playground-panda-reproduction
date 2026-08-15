# 最小实验

这些脚本用于隔离一个概念，不替代正式 reproduction 流程。

| Lab | 依赖 | 设备 | 对应章节 |
| --- | --- | --- | --- |
| `00_course_preflight.py` | 标准库 | CPU | 课程入口与路线选择 |
| `01_transform_2d.py` | NumPy | CPU | 坐标变换 |
| `02_ppo_clipping.py` | NumPy | CPU | PPO 裁剪 |
| `03_mujoco_state.py` | MuJoCo | CPU | 模型与状态 |
| `04_jax_batching.py` | JAX | CPU/GPU | JIT、vmap、PRNG |
| `05_panda_inspect.py` | 项目完整环境 | CPU | Panda 数据流 |
| `09_wilson_interval.py` | 标准库 | CPU | 二项成功率区间 |
| `09_offline_evaluation.py` | 标准库 + compact JSON | CPU | 聚合、分 seed 与验收 |
| `10_offline_failure_analysis.py` | 标准库 + fixture | CPU | 轨迹失败机制 |
| `11_offline_integrity_audit.py` | 标准库 + compact JSON | CPU | guide-state 协议审计 |

## 主动练习

[`starter/`](starter/README.md) 中的文件故意保留 TODO：

| Starter | 依赖 | 训练技能 |
| --- | --- | --- |
| `01_transform_3d_exercise.py` | NumPy | 三维变换、组合、逆变换 |
| `02_advantage_exercise.py` | NumPy | return、done mask、GAE、PPO |
| `04_jax_transforms_exercise.py` | JAX | vmap、scan、PRNG key |

先运行已完成的最小演示，再补全对应 starter。卡住时按 Level 1→3 提示；最后才
查看 [`solutions/labs/`](../solutions/labs/)。

建议在项目 `.venv` 中运行。前两个和 Wilson lab 可用普通 Python 环境运行，
但统一环境能减少版本歧义。每次先写预测，再执行并解释与预测不一致之处。
