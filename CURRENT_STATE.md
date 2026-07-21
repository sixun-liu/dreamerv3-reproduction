# CURRENT_STATE

> Updated: 2026-07-21T04:53:18Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 `EXP-0001` 已确认完成训练并在后期进入官方包络，但按恢复的论文统计口径，本地最后
30K episode mean 为 891.7，论文五 seed 最后三点均值为 935.8。用户已批准先做 checkpoint
独立评估，再运行三个 clean seeds；`EXP-0001` 保留为 `promising_unresolved` 历史证据。

## 当前主要矛盾

官方 score 可追溯到 2023，而本地 runtime 是 2026 post-Nature 作者重实现。训练回报语义和表格
最后三点聚合已经闭合；剩余主要不确定性是代码/环境代际、单 seed 方差、前期样本效率落后以及
终点 checkpoint 缺失。

## 下一项决策

先用 462K checkpoint 跑固定 `eval_only` 并验证保存/加载；随后冻结三个 seeds 的正式 replication，
同时保存训练末段统计和 final-checkpoint 独立评估。三 seed 基线闭环前不启动消融。
