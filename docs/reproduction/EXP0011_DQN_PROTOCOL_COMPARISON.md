# Breakout：DreamerV3 EXP-0011 与 DQN EXP-0004 协议差异

这张表用于连接两条学习路线，不用于算法排行榜。两者预算、输入、环境封装和评测均不等价。

| 项目 | DreamerV3 EXP-0011 | Nature DQN EXP-0004 |
|---|---|---|
| 代码定位 | 2026 作者重实现，50M/ratio256 降规模 | CleanRL 派生的 Nature 2015 独立重实现 |
| 交互预算 | 100K agent decisions = 400K nominal emulator frames | 10M decisions = 40M nominal emulator frames |
| 输入 | 64x64 RGB 单帧，RSSM latent state | 84x84 灰度，4 帧堆叠 |
| 动作 | minimal action set，stochastic categorical actor | minimal action set，epsilon-greedy Q argmax |
| 动作重复 | 4 | 4 |
| Sticky action | false | modern ALE 环境协议另行冻结 |
| 状态/记忆 | 学习世界模型与 latent dynamics | frame stack，无显式环境模型 |
| Replay | 在线 replay，ratio256 | 1M replay buffer，固定更新频率 |
| 稳定机制 | symlog/twohot、KL/free bits、return normalization 等 | target network、reward clipping、Huber loss 等 |
| 训练奖励 | raw reward | clipped reward |
| 报告分数 | raw training score + 独立 raw-score eval | 独立 raw-score eval |
| 主要裁决 | 单 seed 低数据可行性 | 单 seed modern-ALE Table 3 量级部分复现 |

因此，即使两者都在 Breakout 上得到一个分数，也不能直接把数值差异解释为算法优劣。EXP-0011 的
价值是用同一游戏观察“显式世界模型 + latent policy”能否在少两个数量级的交互预算下形成学习证据。
