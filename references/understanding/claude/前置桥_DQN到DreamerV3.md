# 前置桥 · DQN（1312.5602）→ DreamerV3（2301.04104）

> 2026-07-16。导师建议：若过渡跨度大，先看 DQN。本文 = 精读指引 + 两篇之间的概念血脉映射（事实均对 1312.5602 原文核过，见 `_parsed/1312.5602.txt`）。

## 一、这篇是什么

**"Playing Atari with Deep Reinforcement Learning"**（Mnih et al., DeepMind, 2013，NIPS Deep Learning Workshop，9 页）——深度强化学习的开山之作：第一次证明**一个 CNN 能从原始像素端到端学出控制策略**。7 个 Atari 游戏、同一架构同一超参，6/7 超过既往 RL 方法、3 个超过人类高手。今天所有深度 RL（包括 DreamerV3）都站在它的地基上。

## 二、精读指引（9 页，2-3 小时够）

| 重点 | 内容 | 为什么 |
|---|---|---|
| §4 + **Algorithm 1** | Q-learning + 经验回放的 20 行伪码 | 深度 RL 的"Hello World"，值得手抄一遍 |
| §4.1 预处理 | $110\times84$ 下采样裁 $84\times84$、4 帧堆叠、灰度化 | "把 POMDP 近似成 MDP"的最朴素做法——对照 DreamerV3 用 RSSM 隐状态解决同一问题 |
| §5 工程细节 | reward clip 到 $\pm1$ / frame skip $k=4$（Space Invaders 例外 $k=3$，否则激光因闪烁周期不可见）/ RMSProp batch 32 / $\epsilon$ 从 1 线性退火到 0.1（首 1M 帧）/ replay 容量 1M、共训 10M 帧 | 这些"土办法"后来全成标配，且每一个都对应 DreamerV3 的一次升级（见下表） |

⚠️ **版本注意**：2013 版**还没有独立 target network**（目标用上一迭代参数 $\theta_{i-1}$ 算，弱化形式）；冻结 target 网络是 **Nature 2015 版**（"Human-level control…"）才引入的。若想补 target network，配读 Nature 版即可（不必精读）。

## 三、概念血脉：DQN 的每个部件都活在 DreamerV3 里

| DQN（2013） | 病灶 | DreamerV3（2023）的升级版 |
|---|---|---|
| **经验回放** replay memory（1M） | 打破样本相关性、重用数据 | replay buffer 成为**世界模型的训练数据源**；critic 还额外在 replay 轨迹上以 0.3 权重训练 |
| 目标用 $\theta_{i-1}$（Nature 版进化为冻结 target 网） | bootstrap 目标不稳 | **EMA target critic**（每步 2% 混合）——同一思想的连续化版本，且只作正则不接管 $\lambda$-return |
| **reward clip 到 $\pm1$** | 跨游戏奖励尺度差异大 | **symlog + twohot 分类损失**——同一个病（尺度），十年后的新药：clip 丢掉量级信息（吃 1 分和 100 分一样），symlog 压缩但保序保量级。这是理解 v3"尺度无关化"动机的最佳入口 |
| 端到端 CNN 从像素学 Q | 免手工特征 | CNN encoder（$64\times64$）进 RSSM——从"像素→动作值"变成"像素→世界模型→想象中学策略" |
| 4 帧堆叠近似状态 | 部分可观测 | RSSM 循环隐状态（确定态 GRU + 随机离散态）——真正的记忆 |
| Atari 基准（7 游戏） | — | Atari100k 仍是 v3 的 8 域之一 |
| $\epsilon$-greedy 探索 | 探索靠随机 | 熵正则（固定 $\eta=3\times10^{-4}$）+ 百分位回报归一化，探索利用平衡不随尺度漂 |

**根本分野**：DQN 是 **model-free**（直接学"哪个动作值多少分"，不理解世界怎么运转）；Dreamer 系是 **model-based**（先学一个能做梦的世界模型，再在梦里训练策略）。读完 DQN 再读 DreamerV3，"哪些是继承、哪个是革命"一目了然。

## 四、若确认"跨度大"：热身复现选项

DQN 是绝佳的复现热身：**CartPole 上 ~150 行 PyTorch、CPU 半小时收敛；Pong 上单卡数小时**。若导师认可，可作为 DreamerV3 复现前的 1-2 天热身（顺手把 replay/target/$\epsilon$-greedy 亲手写一遍，DreamerV3 的对应组件就全懂了）。是否执行，随研读报告一并问导师。

## 五、推荐阅读顺序（更新版）

1. DQN 原文（本篇，2-3 小时）→ 2. WM1_f1 §WM1-1.2-1.4（World Models→Dreamer v1→v2，谱系怎么从 model-free 走向 model-based）→ 3. WM1_f1 §WM1-1.5 + 研读报告（DreamerV3 本体）→ 4. 代码走读（上手前）。
