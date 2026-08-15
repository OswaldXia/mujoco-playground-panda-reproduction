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

建议在项目 `.venv` 中运行。前两个和 Wilson lab 可用普通 Python 环境运行，
但统一环境能减少版本歧义。每次先写预测，再执行并解释与预测不一致之处。
