# CURRENT_STATE

> Updated: 2026-08-12T02:57:14Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` 已完成工程、训练、独立评测和材料闭环：100K 内完整观察到 L1，固定终点评测也在
3/3 回合取得 log/planks、2/3 取得 crafting table。L2 链未闭合，论文约 100M 的 Diamond
结果未复现；GPU 空闲，无自动追加预算。

## 当前主要矛盾

DMC Vision、Atari100K 与 Minecraft Reduced 三条跨域最小路线均已受限闭环。Minecraft 训练中虽
观察到一次 cobblestone，但 wooden_pickaxe 从未出现，不能据此宣称 L2；当前主要矛盾从工程可行性
转为用户审查结果并选择下一项有界学习问题，而不是自动扩大计算。

## 下一项决策

用户审查 `EXP-0012` 曲线、固定 episode0 视频及受限结论；之后只在“用已有 replay 分析
wooden-pickaxe 瓶颈”和“新授权下扩大预算”之间选择一个预注册问题，当前不启动新实验。
