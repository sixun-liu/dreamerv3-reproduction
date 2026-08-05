# PLAN

> Updated: 2026-08-05T22:08:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在受控实验中理解 DreamerV3 的关键机制，并区分论文原消融与当前 runtime 的机制扩展。
- 当前问题：在 EXP-0008 区分了“持续学习”与“论文数值对齐”后，恢复 Figure 18
  learning-signal 消融的精确梯度语义，为下一个低成本机制实验做准备。

## 阶段退出门

- [ ] 把 Figure 18 两种梯度阻断分别映射到论文原文和 2411f7d 代码，标出等价项与非等价近似。
- [ ] 在任何训练前，用一次小 batch 参数组梯度测试证明：被阻断信号不更新 encoder/RSSM，
  但对应 prediction head/decoder 仍能学习。
- [ ] 候选任务、预算、seed 政策与证据权限经用户确认后，再新建独立 EXP 和 runtime 分支。

## 活动路线

1. 用户审查 EXP-0008 官方对照图和受限负复现表述。
2. 以 arXiv v2 主文/补充 Figure 18 为主真源，恢复三臂协议和 14 任务范围。
3. 先做梯度路由 known-answer test 和最小 smoke，再评估单任务三臂 pilot 是否能区分主要解释。
4. 只有在一个代表任务上出现稳定信号，才考虑增加 seed 或任务；不直接复制 3×14 全矩阵。

## Parked Lanes

- EXP-0006/0007 之后的访问分布匹配机制归因。
- Figure 18 完整 3×14 learning-signal 矩阵及原论文多 seed 聚合。
- Crafter scaling、DMC visual/full suite 和旧 runtime 曲线谱系归因。
- EMA critic、entropy、unimix、replay critic 等 E2--E5 横向扩展。
