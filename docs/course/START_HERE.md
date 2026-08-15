# 从这里开始

这是一份给“会 Python 和 Git，但还不懂机器人学、强化学习与 JAX”的学习者的
入口。先花 10 分钟确认环境和路线，再开始第 00 章。不要先租 GPU，也不要先跑
10M 步训练。

## 1. 确认你在仓库根目录

下面两个路径都应存在：

```bash
test -f pyproject.toml && test -f docs/course/README.md && echo READY
```

若没有看到 `READY`，先进入
`mujoco-playground-panda-reproduction` 目录。课程中的所有命令都默认从这里运行。

## 2. 运行零依赖预检

```bash
python3 docs/labs/00_course_preflight.py
```

它会检查 Python、Git 分支、虚拟环境、核心包、搜索工具、本地产物和 GPU 痕迹，
最后给出可用路线。预检不会安装软件、修改环境或启动训练。

结果分为三类：

- `[PASS]`：本项已经满足；
- `[WARN]`：不阻止当前路线，但以后可能需要处理；
- `[NEXT]`：建议立刻执行的下一步。

## 3. 选择当前路线

| 路线 | 适用情况 | 现在能完成 | 暂时不能证明 |
| --- | --- | --- | --- |
| A：Mac 基础路线 | 当前这台 MacBook Air | 第 00–06 章、5 个 Notebook、基础实验、源码审计 | CUDA 吞吐与正式 PPO 训练 |
| B：离线分析路线 | 没有 GPU/checkpoint | 第 09–11 章的统计、失败分类、完整性练习（Gate 4 PRACTICED） | 自己的正式策略性能与 Gate 4 READY |
| C：Linux GPU 路线 | NVIDIA Linux 服务器 | 第 07–12 章的训练、正式评估与受控实验 | 无；仍需保留协议与证据 |

路线不是互斥的。推荐顺序为 A → B → C：先在 Mac 理解与验证，再用课程内的小型
数据学习评估，最后才把 GPU 时间用于正式实验。

## 4. 创建学习记录

先确认当前位于课程发布分支，再从它创建学习分支：

```bash
git switch docs/course-v0.10.1-knowledge-map
git switch -c learn/panda-course-v0.10.1
mkdir -p notes
cp docs/templates/reproduction-contract.md notes/00-reproduction-contract.md
```

若 `learn/panda-course-v0.10.1` 已存在，只需切回它，不要重复创建。旧的
`learn/panda-course-v0.10` 和 `learn/panda-course-v0.9.1` 分支会保留旧起点；
不要强制覆盖含有自己笔记的分支。
模板不是标准答案，而是防止遗漏预测、协议、证据和边界。

## 5. 先建立知识地图

打开 [`KNOWLEDGE_MAP.md`](KNOWLEDGE_MAP.md)，只看“全课程主线”和第 00–02 章。
把相关知识点标成红/黄/绿。进入每章先读“本章知识清单”，结束后遮住解释复述：

- 红色：没听过，先读对应微课；
- 黄色：能复述但不会计算、运行或定位源码；
- 绿色：能脱稿解释，并完成表中“掌握证据”。

不要用“读过一遍”标绿。遇到卡点只回看对应知识点，不必从整章开头重读。

## 6. 安装项目环境（第 06 章前需要）

Mac 与 Linux 必须各自在本机创建 `.venv`，不能复制虚拟环境。若当前已有通过
预检的 `.venv`，不要重复创建。Mac 的固定安装流程见
[`06 Mac 本地验证`](06-macos-validation.md)，Linux 见
[`07 GPU 训练`](07-gpu-training-and-resources.md)。

## 7. 第一学习单元

依次完成：

1. [`00 项目地图与复现契约`](00-project-map.md)；
2. 填写 `notes/00-reproduction-contract.md`；
3. 提交该文件；
4. 再进入 [`01 坐标、姿态与运动学`](01-coordinates-and-kinematics.md)。

进入第 01 章后，用下面的统一入口启动 Notebook：

```bash
./reproduction/start_course_notebooks.sh
```

它使用项目 `.venv`，不会把课程交给系统 Python 3.9。关闭服务时回到启动终端按
`Ctrl-C`。Notebook 不是必需的 GPU 服务，可完全在当前 Mac 上运行。

遇到错误时先查 [`故障排查`](../reference/troubleshooting.md)。若命令使用 `rg`
但预检提示未安装，可临时把 `rg -n "关键词" 目录` 换成
`grep -RIn "关键词" 目录`。

## 完成入口的标准

- 能说出自己当前选择的路线及其能力边界；
- 预检没有 Python/仓库根目录错误；
- `notes/00-reproduction-contract.md` 已创建；
- 没有为了“先跑起来”而跳过协议、复制跨平台 `.venv` 或提交大型产物。
