# 课程练习

先独立回答，再查看 [`solutions/README.md`](../solutions/README.md)。能背答案但不能
从源码或输出举证，不算完成。

## Gate 1：理论与环境

1. 画出 world→base→tip→cube→camera 的关系，说明图像像素为何不是世界坐标。
2. 将视觉 Panda 写成 MDP，并指出 observation 对 Markov 性的近似。
3. 手算 advantage 为 ±1、ratio 为 0.5/1.5、epsilon 为 0.2 的 PPO 目标。
4. 从源码列出每项 raw reward、scale、metric 与 done。
5. 解释 guide-state 为什么属于训练探索辅助而不是 imitation dataset。

## Gate 2：计算栈

1. 给定 batch `[512,64,64,3]`，指出 batch/height/width/channel 维。
2. 解释为什么改变 `nworld` 会触发新 context/编译，并影响显存。
3. 写一个 `vmap` + `jit` 例子，并用 `block_until_ready()` 正确计时。
4. 画出 XML→MjModel→MJX model/data→Warp renderer→CNN→PPO 的数据流。

## Gate 3：复现

1. 比较 state smoke、RGB probe、GPU smoke 与 full 的验收条件。
2. 针对 SSL EOF、Python 3.14 wheel 缺失、11 GiB OOM 各写一个诊断顺序。
3. 从 `manifest.json` 说明一次结果可复现所需的最小版本信息。
4. 检查一次运行产物是否满足 artifacts checklist。

## Gate 4：评估

1. 重算 964/1,024 的成功率和 Wilson 区间。
2. 解释 point estimate、置信区间、worst seed、acceptance 各回答什么问题。
3. 从 episode records 验证 per-seed 之和等于 aggregate。
4. 比较历史 guide-assisted 与 schema-4 guide-free，列出不可直接比较项。
5. 为一个失败轨迹分配互斥类别，并指出所需观测字段。

## Gate 5：毕业实验

1. 说明为什么当前证据支持 post-contact intervention，而非更多 position sampling。
2. 写出 contact-gated lift progress 的三个边界测试。
3. 预注册唯一变量、控制项、三分布验收线和否定条件。
4. 用 [`capstone-report-template.md`](capstone-report-template.md) 完成报告。
