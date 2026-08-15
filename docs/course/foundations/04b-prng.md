# 04B｜显式随机数与可复现 key

建议 45–60 分钟。前置：04A。

## JAX 不隐藏随机状态

JAX 的随机函数显式接收 key。给相同 key 和 shape，结果完全相同：

```python
key = jax.random.key(7)
a = jax.random.uniform(key, (3,))
b = jax.random.uniform(key, (3,))  # 与 a 相同
```

要得到新随机流，先 split：

```python
key, sample_key = jax.random.split(key)
sample = jax.random.uniform(sample_key, (3,))
```

约定第一个 key 留给未来，第二个只消费一次。重复使用 key 不是“随机碰巧相同”，
而是程序主动请求同一伪随机样本。

## 批量环境中的 key 树

一个训练 seed 先生成根 key，再为 reset、policy sampling、domain randomization、
evaluation 等用途分裂。批量环境还会 split/vmap 成每个 world 的 key。仅记录一个
seed 不足以解释所有结果，代码中的 split 顺序和环境数也会改变随机流。

固定 seed 的用途是使协议可重复和配对比较，不是证明结果普遍。正式评估使用多个
固定 seed；训练改进最好也使用多个 train seed，避免一次幸运优化轨迹。

## 暂停并预测

若把 `num_envs` 从 512 改为 1024，即使根 seed 相同，前 512 个环境一定获得相同
随机序列吗？不能盲目保证；要看 split/fold_in 的实现和调用顺序。因此它是需要
记录的配置改变。

## 主动检查

运行 `docs/labs/04_jax_batching.py`，确认 `same_key_repeats=True` 与
`split_keys_differ=True`。然后在 starter 练习中为四个 env 构造独立 key。

## 通过标准

写出“保留 future key、消费 sample key”的模式；解释 fixed seed 与多 seed 推断
的区别；不再把 seed 当作跨 shape、跨版本的绝对随机轨迹保证。
