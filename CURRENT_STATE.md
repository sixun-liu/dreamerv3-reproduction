# CURRENT_STATE

> Updated: 2026-07-21T02:40:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 `walker_walk` 与 Nature DQN Breakout 均已完成当前单任务、单 seed 预算，并在后期达到
论文/官方曲线的分数量级。两项裁决均为 `promising_unresolved`：工程链路和部分数值证据成立，
但不宣称多 seed、严格历史协议或整篇论文复现。

## 当前主要矛盾

DreamerV3 本地曲线约 400K 后才进入官方包络；当前证据不能区分代码代际、dm-control/MuJoCo
版本、训练回报与官方评估生成口径对早期样本效率差异的贡献。新增计算的边际价值低于先完成论文
理解和人工图审，因此两条实验路线均停止自动续跑。

## 下一项决策

用户先复核 DreamerV3 与 DQN 两张主图，再按 DQN 的 Bellman/replay/target network 到 DreamerV3
的 RSSM/latent imagination/actor-critic 顺序进入研读。完整阶段性口径见
`reports/TWO_PAPER_REPRODUCTION_SUMMARY.md`；新 seed 与消融保持 parked。
