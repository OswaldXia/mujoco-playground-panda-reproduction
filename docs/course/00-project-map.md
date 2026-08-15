# 00｜项目地图与复现契约

建议时间：2 小时。硬件：Mac 即可。

## 学习目标

- 区分“官方基线复现”“工程适配”和“新实验”；
- 能指出源码、执行脚本、运行产物、小型证据各自的位置；
- 写出在运行前固定的复现契约。

## 为什么先学这一章

机器人学习最容易出现的错误不是代码报错，而是实验对象在过程中悄悄改变：
换了 checkpoint、种子、采样范围或成功阈值，最后却仍称为同一次实验。复现契约
让“运行成功”和“结论成立”成为两件可分别审计的事。

## 核心知识

仓库有四类内容：上游环境源码、`reproduction/` 工程脚本、本机忽略的
`artifacts/`、以及提交到 Git 的 `results/`。本项目固定上游提交与依赖；Mac
用于理解、渲染和 smoke，Linux NVIDIA 用于大规模像素 PPO。一次实验应固定：

- 代码提交与分支；
- 环境/算法配置；
- 起始 checkpoint；
- 训练步数与硬件；
- 评估种子、分布、回合数和验收线；
- 保留哪些原始产物与小型证据。

## 最小实验

阅读根目录 README、`reproduction/STATUS.md` 和最近六次提交。把每个结论映射
到一个代码提交和一个 JSON 证据。特别标出历史 guide-assisted 结果与当前
guide-free 结果，二者不可合并。

## 源码定位

- [`README.md`](../../README.md)：项目边界与已验证结果；
- [`reproduction/STATUS.md`](../../reproduction/STATUS.md)：按阶段的事实记录；
- [`reproduction/results/`](../../reproduction/results/)：可版本化证据；
- [`reproduction/artifacts/.gitignore`](../../reproduction/artifacts/.gitignore)：大文件策略。

## 运行前预测

在查看 Git 输出前写下：当前分支属于 baseline、analysis、experiment 还是 docs？
最新 guide-free 左侧评估是否通过 95% 门槛？需要多少额外成功？

## 操作

```bash
git status --short --branch
git log -6 --oneline --decorate
python3 -m json.tool \
  reproduction/results/linux-guide-free-left-trajectory-analysis.json
```

## 预期结果

JSON 显示 964/1,024、`guide_swap_probability: 0.0`、总体未通过，且距离 95%
门槛还差 9 次成功。结论是“部分改善、门槛未通过”，不是“约等于通过”。

## 常见错误

- 把 Git 中的小型摘要当作完整 checkpoint；
- 只记录成功率，不记录分母、种子与采样范围；
- 在看到结果后修改门槛；
- 在分析分支直接训练第二个不相关变量。

## 修改练习

在自己的 `notes/reproduction-contract.md` 写一个表格，固定第 12 章的假设、
唯一变量、控制项、接受标准和产物位置。

## 自测

1. smoke 成功能否证明策略学会抓取？为什么？
2. 为什么 checkpoint 和视频不进入普通 Git 历史？
3. 为什么 guide-assisted 与 guide-free 结果不能直接比较？

## 通过标准

能在两分钟内说明“代码—配置—产物—证据—结论”的关系，并提交一份无事后
改门槛的复现契约。

## Git 节点

```bash
git add notes/reproduction-contract.md
git commit -m "docs: define Panda reproduction contract"
```
