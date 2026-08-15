# 04D｜vmap、scan 与 PyTree

建议 90 分钟。前置：04A–04C。

## vmap：把单样本函数提升为批量函数

若 `energy(x)` 接收 shape `[3]` 并返回标量，`jax.vmap(energy)` 可接收 `[N,3]`
并返回 `[N]`。它描述的是对批维的同构映射，不是 Python 循环逐次驱动设备。
`in_axes/out_axes` 决定哪些参数有 batch 维。

## scan：把时间循环变成数组程序

rollout 有“当前 state → action → next state”的递推。`lax.scan(body, carry, xs)`
将固定长度循环表达为：每步接收 carry 与输入，返回新 carry 与该步输出。结果通常
把时间维堆叠起来。它避免在 Python 中展开大量步骤，也方便 JIT 与自动微分。

## PyTree：批量搬运嵌套状态

环境 state 不只是一块数组，而可能是 dict/dataclass/tuple：observation、reward、
done、metrics、info。JAX 把容器结构视为 tree，数组视为 leaves；`tree.map`、jit、
vmap 可一致处理所有 leaves。两个 tree 的结构不一致会报错，即使每个数组 shape
分别合理。

概念 shape：

```text
single state leaves:          [...]
vmap over 512 envs:           [512, ...]
scan for 10 rollout steps:    [10, 512, ...]
```

## 暂停并预测

single observation 是 `{"pixels": [64,64,3], "state": [8]}`。先 vmap 32 个 env，
再 scan 10 步，收集输出 tree 的叶子 shape 为 `[10,32,64,64,3]` 与 `[10,32,8]`。
carry 的最终值则只保留 `[32,...]`，不会自动带 time 维。

## 主动实验

补全 `docs/labs/starter/04_jax_transforms_exercise.py`。先只实现 vmap，再实现 scan，
最后处理 key；每完成一段就运行，不要一次写完后才调试。

## 通过标准

能给单环境函数标出 batch/time 维；能解释 scan 的 carry 与 outputs；看到 tree
structure mismatch 时先比较容器键与类型，而非只查看数组数值。
