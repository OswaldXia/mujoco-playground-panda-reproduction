# 02D｜PPO 裁剪目标逐项推导

建议 90–120 分钟。前置：02B–02C。

## 未裁剪的更新

importance ratio 乘 advantage 得样本目标 `ratio * A`。若 `A>0`，优化倾向提高
该动作概率；若 `A<0`，倾向降低。问题是同一批 on-policy 数据上更新多轮后，
新旧策略可能偏离太远，而旧数据不再能可靠代表新策略。

## conservative minimum

PPO-Clip 使用：

\[
\min(rA,\;clip(r,1-\epsilon,1+\epsilon)A).
\]

这里的 `min` 在负优势时容易直觉出错：

| advantage | 危险方向 | 被限制的 ratio |
| --- | --- | --- |
| 正 | 把好动作概率抬得过高 | `r > 1+epsilon` |
| 负 | 把坏动作概率降得过低 | `r < 1-epsilon` |

另两个方向不会被裁剪掉“变坏”的结果，因为 conservative minimum 会保留更差项。
裁剪是对 surrogate improvement 的限制，不是把全部 ratio 强制夹在区间内。

## 完整训练 loss

实现通常最小化符号相反的 policy loss，再加 value regression、entropy bonus 和
可能的 gradient clipping：

```text
loss = -policy_objective + value_cost * value_loss
       - entropy_cost * entropy
```

这些系数影响优化尺度，却不能修复 reward 定义、reset 分布或评估泄漏。PPO 也不
保证每轮真实任务成功率单调上升，所以必须保存多个 checkpoint 并预先定义选择规则。

## 四格手算

取 `epsilon=0.2`：

- `A=+1, r=1.5`：raw=1.5，clipped=1.2，取 1.2；
- `A=+1, r=0.5`：raw=0.5，clipped=0.8，取 0.5；
- `A=-1, r=0.5`：raw=-0.5，clipped=-0.8，取 -0.8；
- `A=-1, r=1.5`：raw=-1.5，clipped=-1.2，取 -1.5。

运行 `docs/labs/02_ppo_clipping.py` 验证，然后把 epsilon 改成项目值 0.3，解释
哪些边界改变、哪些样本符号不变。

## 通过标准

不用记表也能从 `min` 推导四格；能说出 PPO 裁剪限制什么、不保证什么；能解释
为什么训练日志的 reward 上升仍需独立 success 评估。
