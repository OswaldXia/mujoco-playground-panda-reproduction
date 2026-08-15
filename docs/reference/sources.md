# 一手资料与版本纪律

课程优先引用本仓库固定源码和官方资料。在线文档会更新；遇到与固定提交不一致
时，以本仓库源码/constraints 为实验事实，以新文档为背景，不静默升级。

## 项目与算法

- MuJoCo Playground 官方仓库：<https://github.com/google-deepmind/mujoco_playground>
- 本项目固定上游 commit：`4db186a5b53427c9d313b9c7200480144894ada1`
- Brax 官方仓库：<https://github.com/google/brax>
- PPO 原论文：<https://arxiv.org/abs/1707.06347>

## MuJoCo / MJX

- MuJoCo 概览：<https://mujoco.readthedocs.io/en/stable/overview.html>
- MJCF 建模：<https://mujoco.readthedocs.io/en/stable/modeling.html>
- Simulation：<https://mujoco.readthedocs.io/en/stable/programming/simulation.html>
- MJX 与 batch rendering：<https://mujoco.readthedocs.io/en/latest/mjx.html>

## JAX

- 安装与平台 wheel：<https://docs.jax.dev/en/latest/installation.html>
- JAX quickstart：<https://docs.jax.dev/en/latest/quickstart.html>
- JIT：<https://docs.jax.dev/en/latest/jit-compilation.html>
- `vmap` API：<https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>
- 随机数：<https://docs.jax.dev/en/latest/random-numbers.html>

## 阅读顺序

先运行本课程 lab 形成直觉，再读官方章节，最后回到项目源码指出对应行。PPO
论文重点读 algorithm、clipped objective 和实验设置；不要把论文伪代码直接
当成本仓库 Brax 实现的全部细节。
