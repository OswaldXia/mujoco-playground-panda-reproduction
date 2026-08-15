# 02B｜回报、价值、优势与 GAE

建议 90–120 分钟。前置：02A。

## 回报回答“以后一共得到多少”

从时刻 t 开始的折扣回报为：

\[
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
\]

若奖励 `[0,0,1]`、`gamma=0.9`，则从头的回报是 `0.81`，最后一步是 `1`。
折扣既表达对近期结果的偏好，也让长序列数值更稳定。

## 价值与优势分工

`V(o_t)` 预测从当前 observation 出发的平均未来回报，是 baseline。优势
`A_t` 比较“这次实际选择”与 baseline：正值表示比预期好，负值表示更差。policy
根据优势更新；value network 通过回归 return/target 学会更好的 baseline。

最简单的一步 TD residual：

\[
\delta_t=r_t+\gamma(1-d_t)V(o_{t+1})-V(o_t).
\]

`done` 时不能把下一回合的价值 bootstrap 进来，所以乘 `(1-d_t)`。

## GAE 为什么存在

只看一步 TD 偏差较大、方差较小；完整 Monte Carlo 回报偏差较小、方差较大。
GAE 用 `lambda` 将不同长度的 TD residual 混合：

\[
A_t^{GAE}=\delta_t+\gamma\lambda(1-d_t)A_{t+1}^{GAE}.
\]

实现通常从轨迹末端向前扫描。`lambda` 越接近 1，使用更长期信息；并非越大一定
越好。训练前常对 batch 内 advantage 标准化，这改变尺度但保留相对信号。

## Worked example

给定 `r=[0,1]`、`V=[0.4,0.6]`、末端 bootstrap `0`、`gamma=0.9`、第二步 done：

```text
delta_1 = 1 + 0 - 0.6 = 0.4
delta_0 = 0 + 0.9*0.6 - 0.4 = 0.14
A_1 = 0.4
A_0 = 0.14 + 0.9*lambda*0.4
```

取 `lambda=0.95` 得 `A_0=0.482`。

## 主动实验

补全 `docs/labs/starter/02_advantage_exercise.py` 中 return 与 GAE。先手算，再运行
断言；若失败，先检查 done mask 和逆序，再看提示或 solution。

## 通过标准

能区分 reward、return、value、advantage；能解释 done mask；能手算两步 GAE，
并说明 value network 在训练时有用但部署 deterministic policy 时通常不需要。
