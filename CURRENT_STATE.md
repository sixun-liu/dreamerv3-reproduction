# CURRENT_STATE

> Updated: 2026-07-21T21:45:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 已完成首个两seed受控机制实验：只关 free bits 的 E1 未在两个seed一致降低 raw KL，
但性能均下降；在 E1 上把 representation KL 权重恢复为1的 P4-reconstructed 在两个seed都把
late KL 压低到 baseline/E1 的约1.6%--4.0%，同时重构损失和性能明显恶化。

## 当前主要矛盾

P4 的强效应在两个seed方向一致，但 posterior entropy 绝对值有明显seed差异；E1 raw-KL方向也
不一致。论文 Figure 6/17 的完整 ablation config 仍未恢复，因此当前证据只支持 2026 runtime、
seeded DMC、walker_walk 下的代码语义机制结论，不支持原图数值或跨任务主张。

## 下一项决策

计算已停止。先由用户审查 `EXP-0006` 两张主图；下一判别问题是：在恢复论文原 P4 配置或选择
第二个代表任务后，E1→P4 的低 KL、重构恶化与性能下降是否仍能跨任务复现。不得自动补第三seed。
