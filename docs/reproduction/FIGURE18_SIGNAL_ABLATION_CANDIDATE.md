# Figure 18 Learning-Signal 消融候选

> Updated: 2026-08-05T22:08:00Z
> Maintainer: codex
> Source of truth: arXiv:2301.04104v2 main text and supplementary Figure 18; runtime `6642b94`

## 论文问题

论文比较三臂：Dreamer、不让 reward/value prediction gradients 塑形表征、不让
reconstruction gradients 塑形表征。主文结论是世界模型的任务无关重建目标贡献主要性能，
reward/value 任务信号在部分任务上进一步提升。

补充图覆盖 14 任务：Atari Atlantis/Breakout/Montezuma，Crafter Reward，DMLab Goals Small/
Nonmatching，PinPad Five/Six，ProcGen Bossfight/Caveflyer，Proprio Dog Run/Reacher Hard，
Visual Acrobot Sparse/Humanoid Run。完整复现是 `3 arms x 14 tasks x repetitions`，不适合直接开跑。

## 2411f7d 代码映射

- `reward_grad=False`：用 stop-gradient 阻断 reward head 对 replay representation 的梯度。
- `replay_critic_grad=False`：阻断 replay critic 对 replay representation 的梯度。默认
  `ac_grads=none` 已阻断 imagination actor/critic 经过 latent 反向塑形世界模型。
- 因此“no reward & value gradients”可以由现有开关接近原义实现，但仍需参数组梯度测试
  确认 encoder/RSSM 收到的任务信号为零，reward/critic head 自身梯度不为零。
- 当前没有等价的“no reconstruction gradients”开关。把 `dec_cnn/dec_mlp` loss scale 设为零会同时
  停止 decoder 学习，不等价于“decoder 继续预测，但重建 loss 不塑形 representation”。
- 候选精确实现是只对 decoder 输入的 latent tree 做 stop-gradient，保留 decoder 参数梯度；
  此语义必须用 known-answer test 证明，不能只看 loss 是否下降。

## 最低成本证据阶梯

1. 单 batch 梯度路由测试：分别报告 encoder、RSSM、reward/value heads 和 decoder 参数组 grad norm。
2. 三臂 2K--16K decision smoke：只验证有限 loss、checkpoint、梯度信标和 ETA，不作性能裁决。
3. 优先从论文列出的 proprio 任务 `reacher_hard` 中选一个做三臂 pilot；任务、预算、
   seed 和聚合口径需在结果盲的新 EXP 中预注册。
4. 单任务 pilot 的证据权限仅为 `conceptual`；不能称为 Figure 18 数值复现。

## 开跑前必须回答

- 论文原始实现中“no reconstruction gradients”是否精确采用 decoder-input stop-gradient？
- reward/value 阻断是否还包含 continuation head 或其他 task-specific head？
- Figure 18 每任务 seed 数、预算与归一化/聚合口径是否可从原始 config 或 score artifact 恢复？
