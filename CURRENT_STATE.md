# CURRENT_STATE

> Updated: 2026-07-21T06:11:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 旧 checkpoint 独立评估和自然结束 final-checkpoint smoke 均已通过。论文表值仍是五 seed
training-return 聚合，不与固定 seed checkpoint eval 混用；现在进入 seeds 0、1、2 的 clean
500K environment-step replication。

## 当前主要矛盾

官方 score 可追溯到 2023，而本地 runtime 是 2026 post-Nature 作者重实现。训练回报语义、表格
最后三点聚合、checkpoint保存和消费路径已经闭合；剩余主要不确定性是代码/环境代际、跨 seed
方差和前期样本效率落后。

## 下一项决策

冻结并顺序运行 seeds 0、1、2 的正式 replication，同时保存训练末段统计和 final-checkpoint 独立
评估。三 seed 基线闭环前不启动消融。
