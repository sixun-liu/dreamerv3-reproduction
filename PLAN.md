# PLAN

- 阶段：`reproduction`
- 北极星：理解 DreamerV3 的关键机制，并在冻结协议下复现至少一个论文结果。
- 当前主要矛盾：固定目标论文与代码版本，选择一个有官方参考产物且算力可承受的首个复现结果。

## 阶段退出门

- [x] 目标论文版本与代码谱系已固定
- [x] 首个目标的 claim-protocol matrix 已完整
- [x] 参考产物、验收包络和 12 小时成本边界已获用户确认

## 活动路线

1. `EXP-0001`：DMC proprio `walker_walk` seed 0，500K environment-step 目标，10K pilot gate。
2. Dreamer 关闭或完成后，运行独立 DQN 2013-style Breakout，目标最多 10M emulator frames。
3. 为两条曲线生成论文同坐标图，分开记录工程成功、趋势复现和数值复现权限。

## Parked Lanes

- Crafter 完整 1.1M run 与 scaling 矩阵。
- DMC visual 和完整 DMC suite/多 seed。
- Minecraft diamond。
- 消融矩阵和 PyTorch 独立重写。
