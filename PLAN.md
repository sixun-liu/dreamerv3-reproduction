# PLAN

> Updated: 2026-07-21T04:53:18Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `reproduction`
- 北极星：理解 DreamerV3 的关键机制，并在冻结协议下复现至少一个论文结果。
- 当前主要矛盾：用三个 clean seeds 区分 `EXP-0001` 的延迟学习是 seed 偶然还是 2026 作者重实现
  的系统性样本效率差异，并补齐终点 checkpoint 与独立评估。

## 阶段退出门

- [x] 目标论文版本、代码谱系、首个 target 和参考产物已固定。
- [x] `EXP-0001` 完成 500K environment steps 并形成同坐标对照。
- [ ] 用户完成 `EXP-0001` 曲线人工复核。
- [x] 官方 DMC JSON 的代码版本、training-return 语义和 Table 4 最后三点聚合完成取证。
- [x] 用户批准当前作者实现的三个 seeds replication，预计 3--4 GPU h。
- [x] Nature DQN Breakout 完成 10M-decision 单 seed 部分数值复现，并与 DreamerV3 形成中文总结。

## 活动路线

1. 对 `EXP-0001` 的 462K checkpoint 做固定 `eval_only`，验证可用模型与评估语义。
2. 增加并 smoke 自然结束 final checkpoint 与固定评估管道。
3. 运行三个 clean seeds，按最后 30K training-return mean 和独立评估双口径关闭基线。

## Parked Lanes

- Crafter 完整 1.1M run 与 scaling 矩阵。
- DMC visual、完整 DMC suite 和五 seed 扩展。
- Minecraft diamond。
- 复现闭环前的消融矩阵和 PyTorch 独立重写。
- Nature DQN 第二 seed、50M 训练和全 Atari 扩展。
