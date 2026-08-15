# 故障排查

原则：先确认失败阶段和完整错误，再改一个因素；保留失败配置与日志。

## 安装解析冲突：JAX/JAXLIB 0.6.2

症状：`ResolutionImpossible`，或提示没有 matching distribution。

1. `python --version`：固定组合不支持 CPython 3.14，使用 3.12/3.11/3.13；
2. 确认正在目标 Linux checkout 自己创建的 `.venv`；
3. 让 `setup_gpu.sh` 安装 CUDA JAX，再按 constraints 安装项目；
4. 不创建 `.venv-gpu` 作为永久绕过；同一台 Linux checkout 只需一个正确 `.venv`。

## PyPI SSL EOF

这通常是网络/代理/TLS 传输问题，不是 `jax==0.6.2` 自己与同一约束冲突。

1. 检查系统时间、CA、代理/VPN；
2. 用浏览器或 `curl -I https://pypi.org/simple/jax/` 检查连通；
3. 重试官方索引；不要使用来源不明的 wheel 或永久关闭证书校验；
4. 日志同时出现 dependency conflict 时，先确认 pip 是否实际拿到了候选列表。

## JAX 只显示 CPU

1. `nvidia-smi` 确认驱动与 GPU；
2. `python -c "import jax; print(jax.devices())"`；
3. 确认激活的是 Linux `.venv` 且安装 CUDA plugin；
4. 避免本地 CUDA 库路径覆盖 pip wheel 自带库。

## 首次编译看似卡住

终端有心跳且 GPU/进程仍活动时等待第一轮 JIT。0% 只表示尚未完成首个 update，
不是编译百分比。长时间无心跳再查 `console.log`、进程和显存。

## GPU OOM（11 GiB）

官方 1024/128/256 在本项目曾请求额外 5.09 GiB 后失败。

1. 确认使用 `full` 而非 `official`；
2. `full` 应自动选 512/64/128；
3. 仍 OOM 时按 256/32/64 整组降低；
4. 保持总 timesteps 并在报告中记录 profile 差异；
5. 异步 allocator 可减轻碎片，但不能创造容量。

## Checkpoint 恢复 shape 错误

症状：vision network 将 `('shape','dtype')` 当 shape，报 concrete integer shape。

原因通常是用错误 API 从 checkpoint metadata 推断 observation spec。应先从实际
环境 observation 构造 shape/dtype，再调用与训练相同的 vision network factory。
运行仓库最新版 evaluator；不要手工把 checkpoint JSON 元组传给 `jnp.zeros`。

## 评估 OOM

用 `--num-envs 128` 减少同时评估回合；四个 seeds 仍顺序完成，总回合数不变。
记录并行度，但不要减少总 episodes 后沿用原置信度结论。

## 绘图失败但 raw report 已完成

三分布 robustness 使用 `--resume <run>`。脚本校验已有 report/schema 后重建分析，
只运行缺失分布。不要重跑并挑结果更好的一次。

## 画面像“成功后突然复位”

成功触发 done，autoreset 立即开始新回合；检查 terminal 前的最后 active state，
不要把 reset 后位置当成掉落。轨迹 schema 已保存 `last_active_box_position`。

## 成功率高但行为可疑

检查 guide probability、step-1 reached、deterministic policy、success 定义、初始
分布和 autoreset。先做完整性审计，再讨论 reward 或网络。

## Notebook 使用了错误 Python/kernel

症状：第一段代码提示 `Wrong kernel`，或系统 kernel 缺少 JAX/MuJoCo。

1. 关闭当前 Jupyter 服务，不要在页面内临时 `pip install`；
2. 从仓库根目录运行 `./reproduction/start_course_notebooks.sh --check`；
3. 检查显示的 `Panda Course (.venv)` kernel；
4. 再运行 `./reproduction/start_course_notebooks.sh`；
5. 页面中选择 `Panda Course (.venv)`，执行 Restart Kernel and Run All。

启动器只在 `reproduction/artifacts/jupyter/` 创建临时 kernelspec，不修改用户全局
kernel。若浏览器没有自动打开，复制终端给出的本地 `http://127.0.0.1...` 链接。

## Notebook 单独运行 cell 正常，Run All 失败

这是隐藏执行顺序依赖。不要通过重复点击“修好”发布源文件：先 Restart Kernel，
从第一格顺序执行；确认修改练习已恢复文档指定值。发布前运行：

```bash
.venv/bin/python reproduction/validate_course_notebooks.py
```

验证器在新 kernel 中顺序执行，但不把运行输出写回 `.ipynb`。
