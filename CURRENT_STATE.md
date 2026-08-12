# CURRENT_STATE

> Updated: 2026-08-12T14:36:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0023 已从精确 200K 有界推进到精确 500K；新增 replay 与固定三局评测均出现木镐和圆石，
预注册 L2 推进假说通过，训练、评测、视频、资源和清场完整性全部通过。

## 当前主要矛盾

EXP-0023 精确新增 `300000` 个唯一 transition；新增 replay 的 `185/304` 条轨迹出现 wooden_pickaxe、
`120/304` 条出现 cobblestone，固定三局中两局达到两项里程碑。另有 `5/304` 条新增轨迹出现
iron_ore 和 iron_ingot，但没有 iron_pickaxe 或 diamond，因此只作为 L3 前置信号，不晋级为稳定能力。
500K 仍远低于论文 Diamond 预算，且 Minecraft 世界 seed 不可控，200K/500K 三局比较只作描述。

当前 `size50m` 实际 optimizer 参数为 `46,812,213`。约 `24.6 GiB` GPU 分配主要受 JAX 预分配
影响；本轮新增 300K 用时 `87.38` 分钟，峰值 RAM/VRAM 为 `53.61/24.07 GiB`。这建立了 50M
时间与资源基线，但不能证明 100M 可行、训练更快或更省总墙钟。

## 下一项决策

下一 cycle 只比较 `size50m/size100m` 的初始化、活跃显存、编译峰值和约 5K 同预算吞吐。100M
通过资源安全和成本门后，才另行预注册从零、等交互预算学习对照，分别报告每步墙钟、交互到里程碑
和墙钟到里程碑；不继续搜索环境数，也不自动追加到 1M 或测试 200M/400M。
