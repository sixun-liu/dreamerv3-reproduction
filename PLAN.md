# PLAN

> Updated: 2026-08-06T09:54:43Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在受控实验中理解 DreamerV3 的关键机制，并区分论文原消融与当前 runtime 的机制扩展。
- 当前问题：EXP-0009 单 seed pilot 已恢复并验证 Figure 18 梯度语义且通过定性方向门；下一步判断
  该排序能否在同预算的 paired seeds 1、2 上保持，而不是直接扩大到论文完整矩阵。

## 阶段退出门

- [x] 把 Figure 18 两种梯度阻断分别映射到论文原文和 2411f7d 代码，标出等价项与非等价近似。
- [x] 在任何训练前，用一次小 batch 参数组梯度测试证明：被阻断信号不更新 encoder/RSSM，
  但对应 prediction head/decoder 仍能学习。
- [x] 候选任务、预算、seed 政策与证据权限经用户确认后，再新建独立 EXP 和 runtime 分支。
- [ ] 用户人工审查 EXP-0009 图与结论权限，再决定是否把同一 1M 协议扩展到 paired seeds 1、2。

## 活动路线

1. 用户审查 EXP-0009 fixed-bin 曲线、配对行为展示和 scoped claim。
2. 若用户批准，冻结相同 Reacher Hard 1M 三臂协议的 paired seeds 1、2；不按 seed 0 结果调参。
3. 跨 seed 顺序稳定后，再在“延长到 10M”与“增加第二代表任务”之间预注册一个选择。
4. Figure 18 完整 3×14 矩阵继续 parked。

## Parked Lanes

- EXP-0006/0007 之后的访问分布匹配机制归因。
- Figure 18 完整 3×14 learning-signal 矩阵及原论文多 seed 聚合。
- Crafter scaling、DMC visual/full suite 和旧 runtime 曲线谱系归因。
- EMA critic、entropy、unimix、replay critic 等 E2--E5 横向扩展。
