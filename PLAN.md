# PLAN

> Updated: 2026-08-05T16:06:22Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在受控实验中理解 DreamerV3 的关键机制，并区分论文原消融与当前 runtime 的机制扩展。
- 当前问题：用第二个 DMC 任务的五 seed 冻结复现，检验 walker 单任务证据能否扩展为
  DreamerV3 Figure 14 / Table 11 的独立数值证据。

## 阶段退出门

- [ ] `EXP-0008` 五个 Cheetah Run seeds 自然完成 500K environment steps，完整性门通过。
- [ ] 本地 final-window 五 seed 聚合、官方末三点五 seed 聚合和完整学习曲线形成同坐标对照，
  并完成不共享主分析实现的独立复算。
- [ ] 裁决明确限制在 `author_reimplementation`、当前环境版本和未受控 DMC env RNG，
  不外推为 exact artifact 或整篇论文复现。

## 活动路线

1. 冻结论文锚点、官方 score hash、2411f7d runtime、五 seed expanded config 和成本包络。
2. 独立 smoke 验证 Cheetah 环境、训练、自然 checkpoint、ETA、显存与磁盘增长。
3. detached 顺序运行 seeds 0--4；健康运行不按中途分数或曲线终止。
4. 完成官方同图、独立复算、受限裁决、artifact 登记和一键复算入口。

## Parked Lanes

- EXP-0006/0007 之后的访问分布匹配机制归因。
- Figure 17 完整 14 任务矩阵及原论文多 seed 聚合。
- Crafter scaling、DMC visual/full suite 和旧 runtime 曲线谱系归因。
- EMA critic、entropy、unimix、replay critic 等 E2--E5 横向扩展。
