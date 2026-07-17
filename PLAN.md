# PLAN

> Updated: 2026-07-17T05:26:23Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `reproduction`
- 北极星：理解 DreamerV3 的关键机制，并在冻结协议下复现至少一个论文结果。
- 当前主要矛盾：解释 walker_walk 前半程落后而约 400K 后进入官方包络的证据缺口。

## 阶段退出门

- [x] 目标论文版本、代码谱系、首个 target 和参考产物已固定。
- [x] `EXP-0001` 完成 500K environment steps 并形成同坐标对照。
- [ ] 用户完成 `EXP-0001` 曲线人工复核。
- [ ] 官方 DMC JSON 的代码版本和评估生成协议完成取证。
- [ ] 根据协议取证决定补 seed 或在当前部分复现处停止。

## 活动路线

1. 用户复核 `EXP-0001` 证据图，登记 `confirmed` 或 `disagreed`。
2. 离线核对官方 DMC JSON 的 lineage、环境和训练/评估语义。
3. 若协议可比且 seed 方差仍是主不确定性，预注册 seed 1、2；否则明确限制并停止加算力。

## Parked Lanes

- Crafter 完整 1.1M run 与 scaling 矩阵。
- DMC visual、完整 DMC suite 和五 seed 扩展。
- Minecraft diamond。
- 复现闭环前的消融矩阵和 PyTorch 独立重写。
