# 06｜Mac 本地验证

建议时间：5 小时。硬件：本机 MacBook Air。

## 学习目标

- 创建与操作系统绑定的干净 `.venv`；
- 分层验证依赖、state 环境、原生渲染与一世界 RGB；
- 理解 Mac 验证能证明什么、不能证明什么；
- 生成 manifest、JSON 报告和图像证据。

## 本章知识清单

- **平台绑定虚拟环境**：macOS 与 Linux 必须分别创建 `.venv`；
- **JAX backend**：包已安装不等于正在使用 GPU，必须检查 devices/backend；
- **分层 smoke**：导入 → manifest → state step → 原生 render → 一世界 RGB；
- **故障半径**：每次只增加一层复杂度，让失败可定位；
- **能力边界**：Mac 可验证语义和图像路径，不能证明 CUDA 训练吞吐。

## 为什么先在 Mac 验证

源码、模型、状态、渲染或依赖问题若能在本机复现，就不应占用 GPU 服务器。
但 Apple Silicon 的 JAX 后端在本项目中是 CPU，MJX-Warp 正式并行视觉训练仍需
Linux NVIDIA；本地 RGB probe 通过不等于具备实用训练吞吐。

## 核心知识

虚拟环境包含解释器路径和平台二进制，不能从 macOS 复制到 Linux。约束文件
固定经过验证的版本组合；`jax` 是 Python 包，`jaxlib`/平台插件决定实际后端。
验证按故障半径递增：导入 → manifest → state reset/step → 原生 render →
MJWarp 一世界 RGB。每层失败时都能缩小原因。

本项目已验证 M1/16 GB：state smoke 通过；一世界 64×64 RGB reset/step 通过；
缓存后约 1.546 s / 2.025 s。它说明视觉数据路径可调试，不说明 10M PPO 可行。

## 最小实验

先运行 state smoke，并打开生成的 `panda_state_smoke.png`。确认机械臂、桌面、
方块画面合理，再运行 RGB probe 检查策略真正看到的 64×64 图像。

## 源码定位

- [`constraints-2026-08-02.txt`](../../reproduction/constraints-2026-08-02.txt)；
- [`smoke_test_macos.py`](../../reproduction/smoke_test_macos.py)；
- [`vision_backend_probe.py`](../../reproduction/vision_backend_probe.py)；
- [`collect_manifest.py`](../../reproduction/collect_manifest.py)。

## 运行前预测

写下预期 backend、observation 类型/shape 和输出路径。预测首次运行为什么比
第二次慢，以及如果只通过 state smoke 能否证明相机 observation 正常。

## 操作

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  -c reproduction/constraints-2026-08-02.txt \
  -e '.[notebooks]'

MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/collect_manifest.py \
  --output reproduction/artifacts/macos-manifest.json

MPLCONFIGDIR=reproduction/artifacts/matplotlib-cache \
  python reproduction/smoke_test_macos.py \
  --output-dir reproduction/artifacts/macos-smoke

python reproduction/vision_backend_probe.py \
  --require-ready \
  --image reproduction/artifacts/macos-vision-observation.png \
  --output reproduction/artifacts/macos-vision-probe.json
```

## 预期结果

- manifest 记录 Python、JAX、MuJoCo、Warp、平台与 Git 状态；
- state 报告中 observation/state 均 finite，图像为 640×480；
- RGB probe shape 为 `[1,64,64,3]`；
- JAX backend 为 CPU，而不是虚构的 CUDA/MPS 训练后端。

## 常见错误

- Python 3.14 没有固定版本的 `jaxlib` wheel：改用 3.12；
- 把旧 `.venv` 从另一台机器复用：删除失败环境后在目标机重建；
- SSL EOF：先确认服务器时间、代理和证书链，再重试，不能解释成依赖冲突；
- CoreGraphics 错误：从普通 Terminal 运行原生渲染；
- `ResolutionImpossible` 同时列同一版本：检查包索引是否实际取到候选 wheel。

## 修改练习

在 `notes/06-mac-boundary.md` 记录冷/热 reset、step 时间和 backend；用三句话
分别说明“已验证”“未验证”“下一台硬件要验证什么”。

## 自测

1. 为什么 macOS 与 Linux 都可命名 `.venv`，却必须分别创建？
2. state smoke、RGB probe、PPO smoke 各排除哪类故障？
3. manifest 为什么要包含 Git dirty 状态？

## 通过标准

三个报告存在且字段合理，能打开两幅图，能明确说出 Mac 的能力边界。完成后通过
Gate 3 的本地一半；GPU smoke 是另一半。

## Git 节点

运行产物默认忽略，只提交你的小型边界说明：

```bash
git add notes/06-mac-boundary.md
git commit -m "docs: record local Panda capability boundary"
```
