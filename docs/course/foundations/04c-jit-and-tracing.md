# 04C｜JIT、trace、缓存与计时

建议 60–90 分钟。前置：04A。

## 第一次调用发生了什么

`jax.jit(f)` 第一次遇到某个输入签名时大致经历：Python 函数被 trace → 生成数组
程序 → XLA 编译 → 设备执行。后续相同签名可复用编译结果，因此第一次明显慢。
训练终端在 0% 停留并伴随 Warp module load，可能仍在编译而非卡死。

输入 shape、dtype、PyTree 结构或标记为 static 的参数改变，可能建立新的缓存项。
函数对象生命周期、JAX/XLA 版本和编译选项也会影响缓存，不能假设另一个进程一定
瞬间复用。

## 异步 dispatch 与正确计时

JAX 可能在 Python 返回前只把工作提交给设备。错误计时：

```python
start = time.perf_counter()
y = compiled(x)
elapsed = time.perf_counter() - start
```

正确做法是在停止计时前 `jax.block_until_ready(y)`。冷启动时间与缓存执行时间应
分别报告，不能取第二次速度冒充首次用户体验，也不能用首次编译评估稳定吞吐。

## static 参数的代价

Python 字符串、分支模式或某些配置作为 static 参数，会在值变化时重编译；动态
数组值则在同一 shape/dtype 下通常复用。把频繁变化的量错误标静态会造成编译风暴，
把必须静态的结构当动态又可能 trace 失败。

## 暂停并预测

依次调用 shape `(8,3)`、`(8,3)`、`(16,3)`：预期第一和第三次需要新编译，第二次
复用。实际缓存还受函数与环境影响，但这是调试的第一预测。

## 通过标准

能区分 compile latency、dispatch latency、device execution；用
`block_until_ready()` 正确计时；看到第一次训练停顿时先检查心跳、进程、显存和
日志，而不是立即终止。
