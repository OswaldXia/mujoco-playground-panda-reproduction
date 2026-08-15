# v0.10 Notebook 教学设计定稿

## 决策

采用混合教学，不进行全课程 Notebook 化。Notebook 负责概念可视化、逐步反馈与
低成本数据探索；Python/Shell 工具继续作为正式训练、评估、备份、测试和服务器
运行的唯一工程入口。

这个边界同时服务两个目标：新学习者能看见中间量和图形，求职项目仍能展示无交互、
可自动化、可审计的工程能力。

## 选入范围

| 主题 | 采用 Notebook 的理由 | Notebook 之后必须做 |
| --- | --- | --- |
| 01 坐标变换 | 空间关系与乘法顺序需要图形 | 三维 starter、Panda IK 源码审计 |
| 02 Return/GAE/PPO | 符号与正负优势容易产生直觉错误 | advantage starter、reward 源码审计 |
| 04 JAX | shape、时间维、PRNG 与冷/热调用适合即时观察 | JAX starter、计算栈图 |
| 09 评估统计 | 区间、分 seed 和门槛适合图表 | 命令行 offline audit 或正式 evaluator |
| 10–11 失败/完整性 | 轨迹时间线与分类分布需要可视化 | failure hypothesis、自己的全量审计 |

## 明确不转换

- 00：复现契约必须作为版本化 Markdown 证据；
- 03、05：MJCF、环境源码和调用链应在真实文件中审计；
- 06–08：安装、smoke、GPU 训练、checkpoint 与备份需要退出码、日志和后台运行；
- 12：control/treatment launcher、预注册和正式报告必须无交互、可批量执行。

Notebook 可以解释这些章节，但不能成为正式操作入口。

## 每个 Notebook 的固定学习循环

```text
学习目标 → 先预测 → 运行与观察 → 动手修改 → 自测 → 反思与记录
```

- “先预测”要求在运行前留下答案，防止只被动观看；
- “动手修改”只改变一个变量，随后恢复预设值；
- “自测”包含可执行断言，但断言通过不等于对应 Gate READY；
- “反思与记录”把结论写入 `notes/` 并连接到 starter 或正式工具。

## 工程约束

1. 统一使用项目 `.venv`；启动器创建仓库内临时 kernelspec，不修改用户全局 kernel；
2. 发布版不保存 outputs/execution count，每个 cell 有稳定 nbformat id；
3. 自动验证从空 kernel、仓库根目录、非交互绘图后端顺序执行全部 cell；
4. 共享纯函数放在 `docs/notebooks/course_utils.py`，正式数据来自已提交 compact JSON；
5. Notebook 不写 checkpoint、不修改策略、不覆盖正式报告；
6. 精选 8 条轨迹始终标注为非成功率样本；
7. 正式结果仍由 `reproduction/` 工具生成并接受现有单元测试。

## 发布验收

- 5/5 Notebook 结构检查通过；
- 5/5 从空 kernel 顺序执行通过；
- 所有本地链接存在，源文件无保存输出；
- `.venv` kernel 错配会在第一段代码给出操作性错误；
- Mac CPU 可完成全部 Notebook；JAX Notebook 不暗示 CUDA ready；
- 现有课程与 reproduction 完整测试继续通过。

这套设计改善交互性，但不宣称“仅完成 Notebook 即完全掌握”。能力等级仍按
[`GATE_RUBRIC.md`](GATE_RUBRIC.md) 的口头解释、starter、源码证据和正式产物综合
判断。
