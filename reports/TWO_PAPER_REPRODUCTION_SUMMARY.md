# DQN 与 DreamerV3 阶段性复现总结

> Updated: 2026-07-21T02:40:00Z
> Maintainer: codex
> Source of truth: DreamerV3 `EXP-0001` and DQN `EXP-0004` registries and artifacts

## 结论摘要

两条代码路线均已跑通，并各自在一个任务、一个训练 seed 上复现了论文结果的后期分数量级：

| 项目 | 冻结目标 | 本地结果 | 论文/官方参考 | 当前裁决 |
|---|---|---:|---:|---|
| DreamerV3 | DMC proprio `walker_walk`，500K environment steps | 490K 同坐标 bin 中位数 914.2 | 官方五 seed 范围 735.6--955.0，中位数 897.9；论文表格 936 | `promising_unresolved` |
| Nature 2015 DQN | Breakout，10M agent decisions，Table 3 replay+target | 40 次评估 peak/final mean 350.18 | 316.8 | `promising_unresolved` |

这可以表述为“完成两套代码的单任务运行，并得到单 seed 部分数值复现”。不能表述为“完整复现两篇
论文”或“与原论文协议严格等价”。

## DreamerV3：世界模型主线

### 做了什么

- 使用作者维护的公开重实现 `danijar/dreamerv3@e3f0224`，运行 `size12m` 的 DMC
  `walker_walk` proprio 配置。
- 冻结 repeat 2、16 environments、replay ratio 512、seed 0 和 500K environment-step 预算。
- 将本地训练 episode score 按 10K steps 分箱，与仓库提供的五条官方 seed 曲线放在同一坐标比较。

### 观察到什么

- 运行约 1.1 GPU 小时后自然完成，无 traceback、NaN 或 OOM；496 个本地 episode 可用于分析。
- 本地前半程明显慢于官方曲线：250K bin 中位数 476.1，低于官方 826.9--945.8。
- 约 400K 后本地曲线进入官方 seed 包络；490K bin 中位数 914.2，官方中位数 897.9。
- `(490K, 500K]` 的额外尾窗中位数为 861.9，说明末段结果仍有 episode 波动。

### 能说明什么

当前作者公开实现能在冻结配置下学会该任务，并在后期达到论文/官方曲线的性能量级。它不能说明前半程
样本效率得到复现，也不能区分代码代际、dm-control/MuJoCo 版本以及训练回报与官方评估语义的影响。

## DQN：强化学习铺路

### 做了什么

- 以 DeepMind DQN 3.0 作者代码作为协议锚点，以 CleanRL 的 MIT 工程结构作为参考，运行独立
  PyTorch 重实现；没有把受限许可的 Lua/Torch7 代码复制进公开仓。
- 冻结 Nature 网络、1M replay、target network、centered RMSProp、reward clipping、探索计划和
  10M agent-decision 预算；现代 ALE 明确作为协议漂移记录。
- 每 250K 训练 decisions 做一次 135K-decision 完整评估，共 40 次；另用最终 checkpoint 独立复评。

### 观察到什么

- 正式运行 `7.934h` 自然完成，共 2,487,500 次 optimizer updates 和 5,498 个完整评估 games。
- peak/final mean 均为 350.1833，final median 373.5，相对论文 316.8 高 10.54%。
- 最后五次均值为 327.55、141.00、342.54、296.32、350.18，高分量级成立但周期波动显著。
- 固定 held-out 状态的 Q 值全程有限；最终 checkpoint 独立复评的 60 个 games 与原 evaluator
  逐 episode return 完全一致。

### 能说明什么

Nature 2015 replay+target 的 Breakout 分数量级在现代独立重实现中得到单 seed 部分复现。由于只有
一个 seed、使用现代 ALE、只固定一个学习率，350.18 不能被解释为与论文 316.8 统计等价。历史
2013 路线也不是配对的 no-target 对照，因此本地结果不能单独证明 target network 的因果贡献。

## 两篇论文如何衔接

DQN 提供最基础的 model-free value learning 框架：从像素直接学习动作价值，用 replay 打破样本相关性，
用 target network 稳定 bootstrap 目标。DreamerV3 则把“从经验中学什么”推进了一层：先在潜空间学习
世界模型，再在模型想象的轨迹中训练 actor 和 critic。研读时可沿以下顺序建立联系：

1. 先理解 DQN 的 Bellman target、experience replay、target network 和 epsilon-greedy。
2. 再理解 DreamerV3 的 encoder、离散 RSSM、decoder/reward/continuation heads。
3. 对照真实环境 replay 与 latent imagination，理解 actor-critic 为何可以主要在想象轨迹中更新。
4. 最后理解 DreamerV3 的 symlog/twohot、free bits、unimix、回报归一化如何处理跨域尺度和稳定性。

## 当前边界与下一步

- 当前不追加 DQN seed、50M 训练、全 Atari 或 DreamerV3 Crafter 长跑；这些成本不能解决当前最急迫
  的论文理解任务。
- 人工复核两张主图后，优先做论文与代码研读：把算法模块、损失函数、数据流和本次曲线现象逐项对应。
- 若导师以后要求提高数值证据，DQN 的下一判别实验是第二固定协议 seed；DreamerV3 则应先恢复官方
  DMC JSON 的代码与评估生成谱系，再决定是否补 seed。

## 证据入口

- DreamerV3 图：`/root/autodl-tmp/artifacts/dreamerv3/EXP-0001/curve_comparison.png`
- DreamerV3 结论：`/root/autodl-tmp/artifacts/dreamerv3/EXP-0001/RESULT.md`
- DQN 图：`/root/autodl-tmp/artifacts/dqn2013/EXP-0004/nature2015_breakout_replication.png`
- DQN 结论：`/root/autodl-tmp/artifacts/dqn2013/EXP-0004/RESULT.md`
- DQN 协议差异：`/root/autodl-tmp/artifacts/dqn2013/EXP-0004/PROTOCOL_DIFFERENCES.md`
- 所有运行状态：`/root/autodl-tmp/runs/STATUS.md`

