# 课程离线数据

这里的数据让没有 NVIDIA GPU 或 checkpoint 的学习者练习统计、失败分类和评估
完整性。它们不是可用于重新训练的 dataset，也不替代正式 1,024 回合评估。

## `guide-free-left-episodes-fixture.json`

- `formal_reference` 保存真实 guide-free 左侧正式评估的完整聚合、分 seed 和失败
  分类计数；与
  [`linux-guide-free-left-trajectory-analysis.json`](../../reproduction/results/linux-guide-free-left-trajectory-analysis.json)
  一致。
- `curated_episode_examples` 是从真实 1,024 回合报告中按每 seed 一成功一失败抽取的
  8 条教学样例，只保留分析所需字段。
- 样例是有意平衡的，**绝对不能**用 4/8 估计策略成功率。成功率必须使用
  `formal_reference` 的 964/1,024。
- 原始 `left-trajectory.json` SHA-256 为
  `13a98c7f6b195fe19b1bdc8319104afc83a970366d03284a0936d40e15305f5a`；因包含
  全量轨迹且体积较大，不随教材提交。

这一区分刻意训练一个工程习惯：用于“看懂一条轨迹”的精选样例，与用于“估计
成功率”的预定义随机样本不是同一种证据。
