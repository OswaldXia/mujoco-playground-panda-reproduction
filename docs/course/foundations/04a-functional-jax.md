# 04A｜JAX 的函数式数组思维

建议 60–90 分钟。

## NumPy 相似，执行模型不同

`jax.numpy` 的表面 API 接近 NumPy，但 JAX 需要把函数转换、追踪和编译。最稳健
的心智模型是纯函数：输出只由显式输入决定，不偷偷读取或修改全局状态。数组更新
使用返回新值的 `.at[index].set/add(...)`，而不是原地修改。

```python
def integrate(position, velocity, dt):
  return position + velocity * dt
```

这样的函数容易被 `jit`、`vmap` 和自动微分组合。文件 I/O、print、Python list
append 或对象副作用应放在编译函数外层。

## Tracer 不是普通数值

JAX trace 函数时，参数可能是只携带 shape/dtype 的 tracer，而非具体 Python
float。`if x > 0:` 试图在 trace 阶段把数组值转为 Python bool，会失败。数据依赖
分支使用 `jax.lax.cond` 或数组运算；只依赖静态配置的 Python 分支可以在外层决定。

## Shape 是接口的一部分

在 NumPy 中换 batch size 通常只是再次运行；在 JIT 下，不同 shape/dtype/静态参数
可能触发新编译。对本项目而言，`nworld`、相机分辨率和 observation tree 结构都不
只是性能旋钮，也决定已编译程序和 buffer 布局。

## 暂停并判断

以下哪些适合放入 jitted step：数组加法；向日志文件写一行；依据数组值执行 Python
`if`；返回一个新 state？答案是数组加法与返回新 state。其余应移到外层或改用 JAX
控制流。

## 通过标准

能把一个原地数组更新改成函数式更新；能解释 tracer 错误为何发生；能预测改变
batch shape 可能重新编译。
