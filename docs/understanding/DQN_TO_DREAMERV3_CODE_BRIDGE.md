# 从 DQN 到 DreamerV3：代码级机制桥

本文绑定两个本地实现：DQN Nature-style `debfa3e`（`dqn-reproduction`）与 DreamerV3
`ad49802` 加 EXP-0007 推理 commit `cdb3d00`。它解释机制对应，不声称两者组件一一等价。

## 一条最短主线

```text
DQN:
transition -> replay sample -> one-step TD target -> Q loss -> epsilon-greedy action

DreamerV3:
sequence -> encoder/RSSM posterior -> model losses
         -> latent prior rollout -> lambda-return -> actor/value losses -> stochastic action
```

DQN 直接在真实 transition 上学习动作价值；DreamerV3 先用真实序列学习一个 latent dynamics，
再把 actor/critic 的大部分训练放到该 dynamics 生成的想象轨迹里。

## 代码对照

| 问题 | DQN Nature-style | DreamerV3 | 关键区别 |
|---|---|---|---|
| 经验存储 | `src/dqn2015_nature_breakout.py:507-515` 建 transition replay，`568-573` 抽样 | `dreamerv3/main.py:182-213` 的 replay/stream 抽连续 sequence | DQN 样本是单步；Dreamer 的 RSSM 需要时间上下文 |
| 目标 | `571-575`：`r + gamma max_a Q_target(s',a)` | `agent.py:188-214,459-523`：latent rollout 后算 lambda-return | 单步 bootstrap 变成多步模型预测加 value bootstrap |
| 预测网络 | `573` 从 online Q 取已执行 action 的值 | `agent.py:65-72` value/slowvalue；`494-499` value loss | Dreamer 还要训练 actor，不从 Q 的 argmax 直接出动作 |
| 稳定目标 | `497-499,584-586`：target network 每 10K decision 硬同步 | `agent.py:65-68,474-499`：当前 value 默认生成 return，slow value 作正则 | 当前配置 `slowtar=False`，所以 slow value 不是 DQN target network |
| 探索 | `169-172,533-540`：epsilon-greedy | `agent.py:192-214,488-492`：stochastic actor 加 entropy；连续动作另有 minstd | entropy/minstd 不等于随机替换整个动作的 epsilon-greedy |
| 表征 | TD loss 通过 Q 网络反传，像素特征只为价值服务 | `agent.py:163-182`：观测、reward、continuation、KL 联合训练表征 | Dreamer 显式要求 latent 可预测世界，而不只拟合当前 Q target |
| 动态 | 无显式环境模型 | `rssm.py:61-118`：真实观测 posterior 与 action-conditioned prior | Dreamer 可在无新环境交互时推进 latent state |
| 模型约束 | 主要靠 replay、target network、reward clipping 等 | `rssm.py:120-138,179-182`：KL balance、free bits、unimix | Dreamer 新增 prior/posterior 匹配与表征容量的矛盾 |

## Target Network 最容易讲错的地方

DQN 的 target network 明确参与 TD 标签：`target_network(next_obs).max()`，并每 10K decision 从
online network 硬复制。它直接减缓“预测网络追逐自己刚改出的标签”。

当前 DreamerV3 配置中 `imag_loss.slowtar=False`。`agent.py:474-482` 因而用当前 value 计算
lambda-return；slow value 出现在 `497-499` 的一致性正则。它确实也提供慢时间尺度，但不应命名成
“Dreamer 版 DQN Target Network”。两者最稳妥的共同点只是：都在控制 bootstrap 学习的移动目标。

## 从 DQN 的估计偏差到 Dreamer 的模型偏差

DQN 的 `max` 同时选择动作和估值。各动作 Q 估计有噪声时，`max` 更容易选中正误差，这就是
overestimation 的入口。target network 减少目标漂移，但不消除同一估计器上的 maximization bias；
Double DQN 才把 action selection 与 evaluation 拆开。

DreamerV3 不再用 `max_a Q` 生成每一步训练标签，但引入另一类误差链：

1. posterior 看到了真实 observation；prior 只看过去 latent 和 action；
2. actor/critic 在 prior rollout 上训练；
3. 单步 prior 小偏差会改变下一步 latent 分布并随 horizon 累积；
4. 因此需要同时看 teacher-forced reconstruction、prior prediction 和 posterior-prior KL。

低 KL 只说明两个 latent 分布接近，不保证二者保留了足够任务信息。`EXP-0006` 的 P4 正好给出
局部反例候选：KL 很低，但重构和控制更差。`EXP-0007` 用共同 replay panel 检查它是否也对应
功能性的多步预测恶化。

## 为什么共同离线 panel 是自然延伸

DQN 已经可以在固定 held-out replay states 上比较不同 checkpoint 的 Q、action margin 与扰动
敏感性。DreamerV3 的对应分析不能只固定单帧，因为研究对象是 dynamics：必须固定一段 observation
context 和后续真实 actions，再比较 horizon 1/3/5/10/15 的预测。

共同 panel 解决“不同策略访问了不同状态”的混杂，但保留三个边界：它不是各策略的 on-policy
分布，不是动作反事实，也不是闭环分数。它回答的是更窄的问题：在完全相同的已记录状态序列上，
哪个冻结世界模型能更好地重构和向前预测？
