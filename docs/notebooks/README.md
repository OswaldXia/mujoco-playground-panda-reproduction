# Panda 课程交互式 Notebook

Notebook 是概念教学层，不是正式训练入口。它们把“先预测—运行—修改—自测—记录”
放在同一页面；可复现训练、评估、备份和 Git 证据仍以 `reproduction/` 下的脚本为准。

## 推荐顺序

| Notebook | 对应章节 | 预计时间 | 重点 |
| --- | --- | ---: | --- |
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

如果只想验证而不打开浏览器：

```bash
.venv/bin/python reproduction/validate_course_notebooks.py
```

## 学习规则

1. 每次只运行一个 Notebook，先写“先预测”答案；
2. 使用 `Restart Kernel and Run All`，不要依赖乱序执行留下的变量；
3. 完成“动手修改”后恢复原始变量，再运行自测；
4. 将反思写入 `notes/`，不要把个人运行输出提交到课程发布分支；
5. 继续完成对应 `.py` starter，因为能运行现成 cell 不等于能独立实现；
6. 正式数字必须由命令行 evaluator 产生，Notebook 只复核已有证据。

## 仓库输出策略

发布版 `.ipynb` 的 `execution_count` 必须为 `null`，`outputs` 必须为空。预期结果
写在 Markdown 中，图片在运行时生成。这样可以审查源代码、避免隐藏旧输出，也
不会让 Git 因二进制图像产生噪声。自动验证器会强制执行这条规则。
