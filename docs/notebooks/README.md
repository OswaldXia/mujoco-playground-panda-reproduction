# Panda 课程交互式 Notebook

Notebook 是主要学习界面，不是正式训练入口。先打开
[`00_course_dashboard.ipynb`](00_course_dashboard.ipynb) 选择路线和查看本地进度；
再按“诊断—知识地图—worked example—预测—运行—故意出错—迁移—Exit ticket”
完成单元。可复现训练、评估、备份和 Git 证据仍以 `reproduction/` 下的脚本为准。

## 推荐顺序

| Notebook | 对应章节 | 预计时间 | 重点 |
| --- | --- | ---: | --- |
| `00_course_dashboard.ipynb` | 00 | 30 分钟 | 路线、证据边界、红黄绿进度 |
| `01_frames_and_transforms.ipynb` | 01 | 60–90 分钟 | 坐标系、组合顺序、三维可视化 |
| `02_returns_gae_and_ppo.ipynb` | 02 | 90 分钟 | return、GAE、概率比、裁剪 |
| `04_jax_execution_model.ipynb` | 04 | 90 分钟 | PRNG、vmap、scan、JIT 冷热调用 |
| `09_evaluation_statistics.ipynb` | 09 | 60–90 分钟 | Wilson、分 seed、固定门槛 |
| `10_failure_analysis.ipynb` | 10–11 | 90 分钟 | 轨迹阶段、失败类别、协议完整性 |

## 正确启动

从仓库根目录运行：

```bash
./reproduction/start_course_notebooks.sh
```

启动器会把当前项目 `.venv` 注册为仓库内临时 kernel，并将缓存写到已忽略的
`reproduction/artifacts/jupyter/`。不要在系统 Python 3.9 kernel 中运行课程。

只检查 kernel、依赖和路径但不启动服务：

```bash
./reproduction/start_course_notebooks.sh --check
```

如果只想验证而不打开浏览器：

```bash
.venv/bin/python reproduction/validate_course_notebooks.py
```

## 学习规则

1. 第一次先运行 00；以后每次只打开 dashboard 推荐的一个 Notebook；
2. 先完成开始诊断、知识地图和 worked example，再写“先预测”答案；
3. 使用 `Restart Kernel and Run All`，不要依赖乱序执行留下的变量；
4. 完成“动手修改”后恢复原始变量，再运行自测；
5. 将反思写入 `notes/`，不要把个人运行输出提交到课程发布分支；
6. 继续完成对应 `.py` starter，因为能运行现成 cell 不等于能独立实现；
7. 关闭页面后复述“学完请记住”，并完成 Exit ticket；
8. 正式数字必须由命令行 evaluator 产生，Notebook 只复核已有证据。

学习状态仅在学习者把 `SAVE_PROGRESS` 改为 `True` 后写入
`reproduction/artifacts/course-progress/`，不会进入 Git，也不会修改正式结果。

## 仓库输出策略

发布版 `.ipynb` 的 `execution_count` 必须为 `null`，`outputs` 必须为空。预期结果
写在 Markdown 中，图片在运行时生成。这样可以审查源代码、避免隐藏旧输出，也
不会让 Git 因二进制图像产生噪声。自动验证器会强制执行这条规则。
