# CURRENT_STATE

> Updated: 2026-07-21T05:51:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 `EXP-0001` 的 462K checkpoint 已通过独立固定 seed stochastic 评估：64 episode mean
893.5，确认参数可加载且策略可用。论文表值仍是五 seed training-return 聚合，不与该 probe 混用；
下一门是自然结束 final checkpoint 保存和三个 clean seed replication。

## 当前主要矛盾

官方 score 可追溯到 2023，而本地 runtime 是 2026 post-Nature 作者重实现。训练回报语义、表格
最后三点聚合和旧 checkpoint 消费路径已经闭合；剩余主要不确定性是代码/环境代际、跨 seed 方差、
前期样本效率落后以及终点 checkpoint 缺失。

## 下一项决策

先补自然结束 final checkpoint 保存并通过短 smoke；随后冻结三个 seeds 的正式 replication，同时保存
训练末段统计和 final-checkpoint 独立评估。三 seed 基线闭环前不启动消融。
