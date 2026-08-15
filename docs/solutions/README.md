# 练习参考答案

答案用于校对关键判断，不替代完整推导。

主动编程练习的可运行参考实现位于 [`labs/`](labs/)。请先独立实现并用断言定位
错误，至少尝试 20 分钟后再查看。

## Gate 1

1. 像素是三维点经相机外参与内参投影后的二维采样；图像左/右不直接定义世界 y。
2. `S` 为像素及隐含历史，`A` 为 y/z/夹爪，`P` 为物理，`R` 为 shaped progress
   加 sparse bonus，`gamma=0.97`。单帧不含速度，严格说是部分可观测近似。
3. 正优势 ratio=1.5 被上界裁剪；负优势 ratio=0.5 取下界裁剪后更负的值。
   正优势 ratio=0.5 与负优势 ratio=1.5 则保留更差的未裁剪值，运行 lab 验证。
4. 以 `pick_cartesian.py::step` 为准；注意 `no_box_collision` 的计算顺序需审计，
   不要仅按配置名推断是否进入已构造的 scaled dict。
5. guide-state 没有提供专家动作标签，只改变少数起始状态；但评估中它会让结果
   不再代表标准 reset 下的策略能力。

## Gate 2

1. 512 是 world batch，其余依次为 64 高、64 宽、3 RGB channel。
2. renderer 在创建时固定 `nworld` 并分配外部 buffer；shape/静态结构变化还会
   触发 JAX 重新编译。
3. 正确计时必须在结果上 `block_until_ready()`，否则只量到异步 dispatch。
4. MuJoCo 编译模型，MJX/Warp 批量推进和渲染，CNN 编码像素，Brax PPO 更新参数。

## Gate 3

1. state smoke 验证模型/状态；RGB probe 验证像素路径；GPU smoke 验证训练/保存
   管线；full 才测试完整预算，仍不等于最终独立验收。
2. SSL 先查网络/证书/时间；wheel 先查 Python 与平台候选；OOM 先查实际 profile、
   峰值阶段，再有记录地降低 env/eval/batch。
3. 至少包含 commit/dirty、OS/架构、Python、JAX/JAXLIB、MuJoCo/MJX、Warp、Brax、
   Playground 和设备。
4. manifest、console、summary、checkpoint、events、videos、正式 evaluation 缺一项
   都要说明，而非默认为存在。

## Gate 4

1. 964/1,024=94.140625%，Wilson 95% 约 92.53%–95.42%。
2. point estimate 描述样本；区间描述抽样不确定性；worst seed 看批次下界；门槛
   执行事先约定的工程决策。
3. 四个 seed 的 success 为 235+242+245+242=964。
4. guide probability/schema/轨迹字段/正式状态不同；旧报告保留为开发证据，不能
   与新报告组成严格同协议提升量。
5. 类别必须按固定优先级从非法、接近、reached、lift、drop、success/timeout
   判定；所需字段见 schema-4 trajectory diagnostics。

## Gate 5

1. 所有 outcome 都已接近并接触，55/60 卡在 lift 之前；再改位置采样没有直接
   对准 dominant mechanism。
2. 无双指接触恒为 0；接触时随高度非递减；达到 lift threshold 后 clip 封顶。
3. 对照表见第 12 章。验收线必须在训练前冻结，并保留 original regression。
4. 报告即使 FAIL 也应完整；可解释的负结果比挑选成功视频更有工程价值。
