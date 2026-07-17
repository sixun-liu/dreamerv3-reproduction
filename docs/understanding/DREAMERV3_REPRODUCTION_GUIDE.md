# DreamerV3 理解与复现导读

## 1. 论文要解决什么

DreamerV3 的核心主张不是“在某一个任务上做到最高分”，而是同一套超参数和训练框架能跨越
离散/连续动作、图像/向量观测、稠密/稀疏奖励和不同奖励尺度。算法先从真实经验学习世界模型，
再主要在模型想象出的轨迹中训练 actor 和 critic，从而减少昂贵的真实环境交互。

本次选择 DMC proprio `walker_walk`，是因为它有五条作者提供的逐 seed 曲线，500K
environment-step 预算在单卡可承受范围内，也能覆盖世界模型、想象行为学习和连续控制整条路径。

## 2. 一条训练数据如何流动

```text
真实观测/动作
  -> encoder
  -> RSSM posterior（看到了当前观测）
  -> RSSM prior（只靠上一步状态和动作预测）
  -> decoder / reward / continue heads
  -> 从后验状态出发想象 15 步
  -> actor 产生动作，critic 估计回报
  -> 更新世界模型、actor、critic
```

- `dreamerv3/agent.py` 组织 policy、train、report 和 imagination。
- `dreamerv3/rssm.py` 实现确定性循环状态与离散随机状态。
- `dreamerv3/nets.py`、`dreamerv3/outs.py` 提供 encoder/decoder、分布输出和 two-hot 等组件。
- 本次 `size12m` 展开后使用 deterministic state 2048、32 个 stochastic latent、每个 16 类，
  MLP hidden 256。完整值见冻结 YAML，不从模型昵称反推。

## 3. 五个最值得理解的稳定化设计

1. **Symlog/symexp**：压缩大数值又保留符号，降低不同奖励和观测尺度带来的优化差异。
2. **Two-hot categorical prediction**：reward 和 value 不直接做单点回归，而是在相邻离散桶上
   分配概率，配合 symlog 改善大范围标量预测。
3. **Free bits**：KL 小于阈值的部分不继续施压，避免表征被过度正则化。
4. **Unimix**：在离散 latent 和离散策略概率中混入少量均匀分布，避免概率退化为精确 0/1。
5. **Percentile return normalization**：用回报的 5%--95% 范围缩放 actor 优势，跨域保持相似
   的梯度尺度；critic 仍学习原始回报语义。

当前代码还包含 replay critic loss（`repval`，scale 0.3），让 critic 同时利用 replay 中的真实
状态，减少只沿当前 imagination 分布学习造成的漂移。这些设计是后续消融的候选变量，但复现
阶段全部保持默认，不能一边跑 baseline 一边改机制。

## 4. 本次冻结协议

| 项目 | 值 |
|---|---|
| 论文结果 | DMC proprio `walker_walk`，500K environment steps，论文表格分数 936 |
| 代码 | 作者公开重实现 commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe` |
| 模型 | `size12m` |
| 环境 | dm-control 1.0.43 / MuJoCo 3.10.0，向量观测 |
| 并行环境 | 16 |
| Action repeat | 2 |
| Replay ratio | 512 |
| 本地 counter | 250K agent decisions；logger 乘 repeat 后为 500K environment steps |
| Seed | 0 |
| 参考 | `scores/dmc_proprio-dreamerv3.json.gz`，五个 seed |

## 5. 必须显式报告的漂移

- 当前仓库 README 将其定位为基于 DreamerV2 的作者公开重实现，并非论文原始 Google/DeepMind
  代码；因此证据等级是 `author_reimplementation`，不是 exact artifact。
- 当前 `dmc_proprio` 默认配置是 1M 模型、repeat 1、ratio 1024、1.1M decisions；本次依据论文
  Table 4 显式覆盖为 12M、repeat 2、ratio 512 和 500K environment steps。
- 论文模型表的 12M recurrent units 写 1024，但同表的 `8d` 规则和当前代码均对应 2048；本次
  采用展开代码值并把论文单元格视为未获作者确认的表内不一致。
- 当前 `script=train` 记录训练 episode return，不是单独 deterministic evaluation。对照图因此
  是作者参考曲线与本地训练回报的可比性证据，不自动等于完全相同的评估协议。

## 6. 如何读最终结果

- **跑通**：环境、训练、日志和 checkpoint 正常，只说明工程链路成功。
- **趋势部分复现**：本地曲线持续上升，并进入作者五 seed 的范围。
- **数值复现**：还需要多 seed、完全一致的代码/环境/评估协议和预注册容差；本次单 seed 不具备
  这个证据权限。
- 若曲线偏低，应先检查代码代际、dm-control/MuJoCo、step 口径和训练/评估差异，再讨论算法。

