# CURRENT_STATE

> Updated: 2026-08-12T07:11:12Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0015` 已证明 EXP-0012 checkpoint、replay 与 step 可在隔离 CoW 输出中完整恢复到
`envs=2`。当前可从原始 100K 基线另开正式增量训练，而不混入 5K 诊断结果。

## 当前主要矛盾

恢复后恰好保留 100000 个旧 stepid 并新增 5040 个唯一 transition；稳态 policy FPS 为
`40.51`，新增 100K 训练预计约 45--50 分钟。独立三回合评测历史实测约 51 分钟，因此完整
训练、评测和材料周期按 1.5--2 小时预算，并保持世界种子不可控的限制口径。

## 下一项决策

从原始 EXP-0012 100K 源另做独立 CoW 克隆，正式训练到绝对 200K；随后沿用 agent seed 10000、
三回合单环境评测和固定 episode-0 视频，比较 L1 是否保持以及 wooden-pickaxe/L2 是否出现。
