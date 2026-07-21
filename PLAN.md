# PLAN

> Updated: 2026-07-21T02:40:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `reproduction`
- 北极星：理解 DreamerV3 的关键机制，并在冻结协议下复现至少一个论文结果。
- 当前主要矛盾：在不夸大单 seed 证据的前提下，把已跑通的 DQN 基础与 DreamerV3 世界模型机制
  连接起来，完成导师要求的理解与复现叙事。

## 阶段退出门

- [x] 目标论文版本、代码谱系、首个 target 和参考产物已固定。
- [x] `EXP-0001` 完成 500K environment steps 并形成同坐标对照。
- [ ] 用户完成 `EXP-0001` 曲线人工复核。
- [ ] 官方 DMC JSON 的代码版本和评估生成协议完成取证。
- [ ] 根据协议取证决定补 seed 或在当前部分复现处停止。
- [x] Nature DQN Breakout 完成 10M-decision 单 seed 部分数值复现，并与 DreamerV3 形成中文总结。

## 活动路线

1. 用户复核 DreamerV3 `EXP-0001` 与 DQN `EXP-0004` 证据图，分别登记人工判断。
2. 结合 `docs/understanding/DREAMERV3_REPRODUCTION_GUIDE.md` 和双论文总结进入论文研读。
3. 离线核对官方 DMC JSON 的 lineage、环境和训练/评估语义；当前不启动补 seed 或消融。

## Parked Lanes

- Crafter 完整 1.1M run 与 scaling 矩阵。
- DMC visual、完整 DMC suite 和五 seed 扩展。
- Minecraft diamond。
- 复现闭环前的消融矩阵和 PyTorch 独立重写。
- Nature DQN 第二 seed、50M 训练和全 Atari 扩展。
