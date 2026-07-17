# DQN 社区实现谱系（provenance · Claude 凭记忆整理，逐条标确信度）

> 2026-07-17 · 应 codex 请求。**本文为记忆性调研，非已核事实**：每条附确信度与取证动作；lineage 分类按 kit `paper-reproduction.md` 四级（original / author-reimpl / third-party / independent）。服务 DQN 铺垫复现（1312.5602 阅读 + 可能的热身复现）。

## 一、原始与"官方"侧

| 实现 | lineage | 说明 | 确信度 / 取证 |
|---|---|---|---|
| 2013 workshop 版（1312.5602） | original | **无公开官方码** | 高 |
| Nature 2015 版官方码 | original | **Lua + Torch7**，DeepMind 随 Nature 论文发布（tarball 形式；GitHub 上流传的 `kuz/DeepMind-Atari-Deep-Q-Learner` 为社区镜像）。考古级，不建议实际运行 | 中高 / 需要时核 GitHub 与 Nature 补充材料链接 |

## 二、现代第三方实现（复现实际会用的）

| 实现 | 框架 | 忠实度印象 | 参照曲线 | 确信度 / 取证 |
|---|---|---|---|---|
| **CleanRL `dqn_atari.py`**（vwxyzjn/cleanrl） | PyTorch + gymnasium + ale-py | 单文件、对齐 Nature 版超参，教学与复现首选 | **openrlbenchmark 有公开基准曲线（W&B）可直接对分** | 高 / clone 后对 Nature Extended Data 表核超参 |
| Dopamine（google/dopamine） | JAX/TF | Google 官方 RL 库，DQN/Rainbow 忠实度高，学术界常用基线 | 自带公开 baseline 数据（JSON） | 高 |
| Stable-Baselines3 DQN（DLR-RM） | PyTorch | 工程化封装深；**默认超参与 Nature 版有出入**（buffer/探索调度需按 rl-baselines3-zoo 的 atari 配置） | RL Zoo 有调优结果 | 高 |
| tianshou（thu-ml） | PyTorch | 忠实、模块化，中文社区友好 | 有 benchmark 页 | 中高 |
| PFRL（pfnet/pfrl，前 ChainerRL） | PyTorch | 带 reproduction 表格（对论文分数逐项对照）——谱系意识最好的一家 | reproduction 表 | 中高 / 核其 DQN 复现表是否含 Nature 口径 |
| RLlib（ray-project） | 多后端 | 分布式工程向，教学/轻量复现不推荐 | — | 高 |

## 三、⚠️ 复现口径的最大坑：评估协议与环境版本（比选哪家实现更重要）

1. **2013 vs Nature 2015 协议不同**：2013 版仅 7 游戏、评估口径古老且**无 target network**；Nature 版 49 游戏、30 no-op starts、评估 $\epsilon=0.05$、human-normalized 汇总。**对分数应对 Nature 口径或现代基准**（CleanRL/Dopamine 曲线），不要对 2013 论文表逐分复刻。确信度高。
2. **ALE 环境版本坑**：现代 gymnasium 的 `ALE/*-v5` 默认 **sticky actions = 0.25**（Machado et al. 2018 建议），与 Nature 设置（无 sticky、v4 风格）不同——**同一实现在 v4/v5 上分数不可直接互比**。热身复现时先固定环境版本并写进 freeze。确信度高 / 取证：ale-py 文档确认默认值。
3. frame skip、reward clip、生命损失终止（terminal-on-life-loss）等 wrapper 细节各家默认不一——对分前逐项核对 wrapper 栈。确信度高。

## 四、给热身复现的建议路径（若导师确认做）

1. CartPole 手写 DQN（~150 行，CPU 半小时）——练 replay/target/$\epsilon$-greedy 三大件；
2. Atari 选 Pong/Breakout：**手写 + 以 CleanRL 为对拍参照**，分数对 openrlbenchmark 曲线（同 wrapper 栈前提下）；
3. 裁决按 kit 五档：单 seed 趋势一致 = `promising_unresolved` 即达热身目的，不必追数值三闭环。
