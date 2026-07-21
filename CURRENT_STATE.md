# CURRENT_STATE

> Updated: 2026-07-21T14:23:46Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 已完成代表性结果复现，当前经用户批准进入机制探索：在显式控制 DMC 环境随机源的
`walker_walk` 上比较 baseline、只关 free bits 的 E1，以及同时去掉 KL balance 与 free bits 的
重构 P4，回答两种 KL 手术分别如何影响表征和性能。

## 当前主要矛盾

论文 Figure 6/17 只给出组合臂与跨任务聚合，公开仓库没有完整 ablation config。当前实验因此
追求受控机制证据，不声称复现原图数值；P4 使用代码语义恢复，最终结论必须限定在 2026 runtime、
seeded DMC 和 walker 任务。

## 下一项决策

先完成三臂各两个配对 seed 的 500K environment-step 主矩阵。只有六条 run、raw-KL 仪器、终点
checkpoint 和同坐标分析全部通过后，才根据剩余预算决定是否补第三个配对 seed。
