# 04｜JAX、MJX、Warp 与 Brax

建议时间：12 小时。硬件：前半 Mac，GPU 部分只读或在服务器运行。

按以下微课逐层学习，不需要第一遍就读懂训练入口全部实现：

1. [`04A 函数式 JAX`](foundations/04a-functional-jax.md)（1–1.5h）；
2. [`04B 显式 PRNG`](foundations/04b-prng.md)（0.75–1h）；
3. [`04C JIT 与 tracing`](foundations/04c-jit-and-tracing.md)（1–1.5h）；
4. [`04D vmap、scan 与 PyTree`](foundations/04d-vmap-scan-pytree.md)（1.5h）；
5. [`JAX 执行模型交互 Notebook`](../notebooks/04_jax_execution_model.ipynb)（1.5h）；
6. [`04E MuJoCo/MJX/Warp/Brax 边界`](foundations/04e-mjx-warp-brax.md)（1–1.5h）；
7. 演示、starter、源码 shape 追踪与复盘（3–4h）。

## 学习目标

- 理解纯函数、不可变数组、显式 PRNG key；
- 能解释 `jit`、`vmap`、`scan` 与 PyTree；
- 区分 MuJoCo、MJX、MJX-Warp、批量渲染器和 Brax PPO；
- 解释首次等待、重编译和 GPU OOM 的原因。

## 本章知识清单

- **纯函数与 PyTree**：状态显式输入/输出，嵌套数组结构可被 JAX 整体变换；
- **PRNG key**：随机状态显式传递并拆分，复用同一个 key 会复现同一随机数；
- **`jit`**：按 shape、dtype 和静态结构 trace/编译，首次调用包含编译成本；
- **`vmap`**：给单样本函数增加环境 batch 维；
- **`scan`**：把时间递推写成可编译循环，产生 time 维；
- **栈边界**：MJX/Warp 负责设备物理与渲染，Brax 负责 rollout 和 PPO 更新。

## 为什么需要这些概念

视觉 PPO 的吞吐来自“许多相同世界并行”，不是单场景更快。JAX 会先 trace 并
编译数组程序；MJX 提供可在设备上批处理的物理状态；MJX-Warp 针对 NVIDIA
实现物理/渲染；Brax 提供训练算法和网络。

## 核心知识

- `jit(f)`：按输入 shape/dtype 和静态结构编译；第一次慢，之后复用；
- `vmap(f)`：给函数自动增加批维，不等同 Python for；
- `lax.scan`：把固定长度循环表达为可编译的数据流；
- PRNG key：随机状态显式传入并 `split`，重复使用同一 key 会重复样本；
- PyTree：嵌套的 dataclass/dict/tuple，叶子为数组，可整体变换。

MuJoCo CPU 适合单场景调试和原生渲染。MJX 对象包含设备数组，可增加 batch
维。MJX-Warp 的 batch renderer 创建时固定 `nworld`，显存随世界数、图像、
接触容量、训练 batch 和中间激活共同增长。Brax PPO 再对环境进行 autoreset、
rollout、优势估计和多轮 minibatch 更新。

```mermaid
flowchart LR
  XML["MJCF"] --> M["MuJoCo MjModel"]
  M --> X["MJX/Warp device model"]
  X --> E["batched env states"]
  E --> P["64×64 RGB policy"]
  P --> A["Brax PPO updates"]
```

## 最小实验

先完成
[`04_jax_execution_model.ipynb`](../notebooks/04_jax_execution_model.ipynb)，观察
PRNG、batch/time shape 和同步计时。Mac 的 CPU backend 是预期结果。

在项目 `.venv` 运行 [`labs/04_jax_batching.py`](../labs/04_jax_batching.py)。
比较 Python 循环、`vmap`、`jit(vmap(...))`，并故意复用一次 key 观察重复随机数。

再补全
[`04_jax_transforms_exercise.py`](../labs/starter/04_jax_transforms_exercise.py)，
分别实现 batch、时间 scan 与独立 key；最后才对照
[`参考实现`](../solutions/labs/04_jax_transforms_solution.py)。

## 源码定位

- `mjx_env.py::step`：`lax.scan` 执行物理子步；
- `pick_cartesian.py`：`mjx.put_model`、render context、BVH 与 RGB；
- `train_jax_ppo.py`：训练函数、网络工厂、progress callback；
- JAX quickstart：<https://docs.jax.dev/en/latest/quickstart.html>；
- JAX 随机数：<https://docs.jax.dev/en/latest/random-numbers.html>；
- MJX 官方文档：<https://mujoco.readthedocs.io/en/latest/mjx.html>。

## 运行前预测

预测 `vmap` 后数组 shape；预测同一个 key 调用两次 `random.uniform` 的结果；
说明把 `nworld` 从 512 加到 1024 为什么可能不只是显存翻倍那么简单。

## 操作

```bash
source .venv/bin/activate
python docs/labs/04_jax_batching.py
python docs/labs/starter/04_jax_transforms_exercise.py
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## 预期结果

`vmap` 和显式逐项结果数值一致；首次 jitted 调用包含编译开销；同 key 生成相同
样本，split 后不同。Mac 显示 CPU，Linux CUDA 环境应显示 GPU。

## 常见错误

- 在 jitted 函数内依赖 Python 可变全局状态；
- 改变 batch shape 后误以为会复用旧编译；
- 把日志停顿当死机，或把 JIT 心跳当训练步进度；
- 只看显卡总显存，不看 JAX 预分配和最大瞬时分配；
- 把 `jax` 与带平台二进制的 `jaxlib` 当成同一个包。

## 修改练习

把 lab 的 batch 从 8 改到 16，并用 `block_until_ready()` 正确计时。再画出一次
环境 step 中 Python、JAX、MJX、Warp、GPU 的职责边界。

## 自测

1. 为什么 JAX 需要显式 key？
2. `jit` 的首次耗时与训练速度有什么关系？
3. 为什么 Mac 可以做一世界 RGB probe，却不适合正式视觉 PPO？
4. Brax 与 MuJoCo 谁负责物理、谁负责 PPO？

## 通过标准

能预测基本 `jit/vmap` 程序的 shape 和随机行为；能解释项目中的五层计算栈，
并能根据 GPU OOM 指向并行环境、评估环境、batch 或接触容量，而非盲目重试。

## Git 节点

```bash
git add notes/04-jax-stack.md
git commit -m "docs: map JAX MJX Warp and Brax stack"
```
