# 07｜GPU 训练与资源管理

建议时间：10 小时，不含等待。硬件：Linux NVIDIA GPU。

## 学习目标

- 完成 CUDA/JAX/MJWarp preflight 和 100k smoke；
- 区分 `smoke`、`full`、`official` 的实验含义；
- 根据显存选择并行环境、评估环境和 batch；
- 阅读 JIT、训练进度、TensorBoard、checkpoint 和视频。

## 为什么 GPU 配置也是实验设计

减少并行环境可以保持 10M 总步数，但会改变采样/优化批次结构和墙钟时间；
因此它是“内存适配的完整工作量复现”，不应冒充精确官方并行度复现。11 GiB
RTX 2080 Ti 上官方 1024/128/256 配置在首轮更新申请额外约 5.09 GiB 时 OOM，
而 512/64/128 完成训练。

## 核心知识

- `smoke`：100k 步，只证明安装、编译、更新、保存、回放管线；
- `full`：完整 10M 步，按显存适配并行度；
- `official`：固定 1024 train / 128 eval / batch 256，需要高显存；
- JAX 首次 evaluation/update 会编译，0% 停留不等于卡死；
- OOM 受训练 batch、视觉中间张量、评估并行度和 contact capacity 共同影响。

启动脚本设置异步 CUDA allocator，按显存选择 profile，并打印阶段、心跳、评估
进度、ETA 与产物路径。完整底层 Warp 输出仍保存在日志中。

## 最小实验

只运行 `smoke`。在命令前写下预期耗时范围、显存 profile 和验收条件。运行后
检查 `manifest.json`、`console.log`、`evaluation-summary.json`、checkpoint、
TensorBoard event 与至少一个 mp4 是否同时存在。

## 源码定位

- [`setup_gpu.sh`](../../reproduction/setup_gpu.sh)：Linux 环境与 CUDA JAX；
- [`train_panda_gpu.sh`](../../reproduction/train_panda_gpu.sh)：模式、显存与输出；
- [`train_jax_ppo.py`](../../learning/train_jax_ppo.py)：Brax 训练入口；
- JAX 官方安装：<https://docs.jax.dev/en/latest/installation.html>。

## 运行前预测

预测 11 GiB GPU 的 `full` profile；说明 smoke 即使成功率为 0，何时仍可 PASS。
写下若 OOM，首先减少哪个量，以及减少后结论应如何表述。

## 操作

```bash
git status --short --branch
./reproduction/setup_gpu.sh
./reproduction/train_panda_gpu.sh smoke
```

确认 smoke 的所有产物后才运行：

```bash
./reproduction/train_panda_gpu.sh full
```

高显存且确实需要精确上游并行度时才使用：

```bash
./reproduction/train_panda_gpu.sh official
```

## 预期结果

preflight 显示 `gpu` 与 `cuda:0`。smoke 完成三个或预设次数的评估、至少一个
checkpoint 和视频。full 在 2080 Ti 自动选择 512/64/128，总有效步数约
10,035,200；这不是失败，而是整数 rollout/batch 对齐后的结果。

## 常见错误

- 直接跑 full，安装/保存错误数小时后才暴露；
- 看到 `[progress] steps=0` 就终止首次编译；
- OOM 后只重跑，不减少 profile 或记录失败配置；
- 用 `official` 名称描述自动缩小的运行；
- 只保留视频，不保留 checkpoint、manifest 和日志。

## 修改练习

用环境变量把 full profile 改为 256/32/64，但不要真正启动 10M；只阅读脚本并
写出预期内存、吞吐和统计变化。把答案记录到 `notes/07-gpu-plan.md`。

## 自测

1. 100k smoke 成功能证明哪些工程链路？不能证明什么？
2. OOM 为什么可能发生在初始评估之后？
3. `num_envs` 减半但 timesteps 不变，哪些量保持、哪些量改变？
4. 为什么训练视频不等于独立评估？

## 通过标准

GPU smoke 全部产物齐全；能读出一次 evaluation boundary；能给 OOM 提出有
记录、可比较的调整，而不是随机改多个参数。至此通过 Gate 3。

## Git 节点

```bash
git add notes/07-gpu-plan.md
git commit -m "docs: record GPU smoke and memory profile"
```
