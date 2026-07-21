# PLAN

> Updated: 2026-07-21T14:23:46Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在受控实验中理解 DreamerV3 的关键机制，并区分论文原消融与当前 runtime 的机制扩展。
- 当前问题：区分 free bits 本身与 KL balance 对 walker 表征和性能的作用。

## 阶段退出门

- [ ] baseline、E1、P4 各两个配对 seed 自然完成 500K environment steps，完整性门通过。
- [ ] raw KL、低 KL 比例、prior/posterior entropy、reconstruction loss、score AUC 和 final-window
  形成同坐标对照。
- [ ] 结论明确限制在当前 runtime 与任务，不外推为 Figure 6/17 数值复现。

## 活动路线

1. 冻结默认关闭的 DMC seed 与 raw-KL instrumentation runtime。
2. 顺序运行 baseline/E1/P4 的 seeds 0、1，固定 500K environment steps。
3. 完成逐 seed 配对分析；仅在主矩阵闭环且预算允许时补最关键对比的第三 seed。

## Parked Lanes

- Figure 17 完整 14 任务矩阵及原论文多 seed 聚合。
- Crafter scaling、DMC visual/full suite 和旧 runtime 曲线谱系归因。
- EMA critic、entropy、unimix、replay critic 等 E2--E5 横向扩展。
