# v0.11 Notebook 教学设计定稿

## 决策

Jupyter 是概念学习和离线分析的主要界面；Markdown 是可检索的系统参考；
Python/Shell 工具仍是正式训练、评估、备份、测试和服务器运行的工程入口。

三者不是重复版本：Notebook 让学习者观察中间量、犯可诊断的错误并获得即时反馈；
Markdown 解释完整背景、推导和故障边界；命令行工具提供无交互、可批量、可审计的
正式证据。Notebook 通过不等于对应能力门 READY。

## 选入范围

| Notebook | 教学任务 | 完成后迁移到 |
| --- | --- | --- |
| 00 dashboard | 路线、证据边界、红黄绿进度 | 复现契约与学习分支 |
| 01 坐标变换 | 观察坐标表达和变换顺序 | 三维 starter、Panda IK 源码 |
| 02 Return/GAE/PPO | 观察终止 mask、优势符号和裁剪 | advantage starter、PPO 配置 |
| 03 MuJoCo 状态/控制 | 连接 Model、Data、actuator 和物理步 | Panda MJCF 与控制器 |
| 04 JAX | 观察 PRNG、shape、时间维和冷/热调用 | JAX starter、GPU 计算栈 |
| 05 Panda 数据流 | 串起 action→IK→physics→reward→RGB | 真实环境 reset/step/reward 源码 |
| 09 评估统计 | 区分点估计、区间、per-seed 与门槛 | offline audit 或正式 evaluator |
| 10–11 失败/完整性 | 从分类和时间线定位机制、审计协议 | failure hypothesis、正式重评估 |

第 06–08 章的安装、smoke、GPU 训练、checkpoint 与备份，以及第 12 章的
control/treatment 正式实验，不转换成 Notebook 操作。它们需要稳定退出码、日志、
后台运行、资源监控和可重复调用。

## 每个 Notebook 的固定学习循环

```text
开始前诊断 → 知识地图 → Worked example → 先预测 → 运行与观察
             → 故意出错 → 动手修改 → 自测 → 项目源码连接
             → Exit ticket → 记忆与反思
```

- **开始前诊断**：暴露已有概念模型，不计分；答错时给具体线索。
- **Worked example**：完整演示一个最小问题，解释每个中间量。
- **先预测**：在看结果前写下方向、shape 或门槛判断。
- **故意出错**：默认包含一个不会中止 Run All 的错误答案，由反馈函数指出原因；
  学习者修改后再次运行。
- **动手修改**：使用控件或单一变量做小型反事实实验，并保留可编辑变量作为回退。
- **自测**：断言验证计算和数据边界，不以“代码跑通”替代口头解释。
- **项目源码连接**：明确 Notebook 中的概念落在真实文件、配置或报告的哪里。
- **Exit ticket**：检查能否迁移判断；通过后才建议进入下一单元。
- **记忆与反思**：脱稿复述，并把需要长期保留的结论写进个人 `notes/`。

## 反馈与进度规则

`course_feedback.py` 提供数值和选择题的针对性反馈。错误提示必须说明应该检查什么，
不能只返回“Wrong”。进度只采用红、黄、绿三级：

- 红：尚不能解释，返回对应 worked example；
- 黄：能跟做，但不能独立计算或定位源码；
- 绿：Exit ticket 通过，且能脱稿解释项目落点。

发布版默认 `SAVE_PROGRESS = False`。学习者主动打开后，状态只写入
`reproduction/artifacts/course-progress/progress.json`；该目录不进入 Git，
也不修改正式实验结果。

## 工程约束

1. 统一使用项目 `.venv`；启动器直接打开 dashboard，并创建仓库内临时 kernelspec；
2. 发布版不保存 outputs/execution count，每个 cell 有稳定 nbformat id；
3. 全部 Notebook 必须从空 kernel、仓库根目录、非交互绘图后端顺序执行；
4. 每个 Notebook 预计不超过 90 分钟，并能在 Mac CPU 上完成；
5. 共享纯函数放在 `docs/notebooks/course_utils.py`，教学反馈放在
   `docs/notebooks/course_feedback.py`；
6. Notebook 不写 checkpoint、不修改策略、不覆盖正式报告；
7. 精选 8 条轨迹始终标注为非成功率样本；
8. 控件必须有确定的默认值，`Restart Kernel and Run All` 不需要人工点击；
9. 正式结果仍由 `reproduction/` 工具生成，并接受现有单元测试与证据校验。

## 发布验收

- 8/8 Notebook 主动学习结构检查通过；
- 8/8 从空 kernel 顺序执行通过，单本不超过 180 秒；
- 所有本地链接存在，源文件无保存输出和 execution count；
- 反馈函数、进度路径与关键数值有自动测试；
- 启动器默认打开 dashboard，kernel 错配能给出操作性错误；
- Mac CPU 可完成全部 Notebook；JAX Notebook 不暗示 CUDA ready；
- Markdown 公式审计、课程测试与 reproduction 回归测试继续通过。

能力等级仍按 [`GATE_RUBRIC.md`](GATE_RUBRIC.md) 的口头解释、starter、源码证据和
正式产物综合判断。Notebook 的作用是缩短反馈回路，不是把“完全掌握”简化成八次
Run All。
