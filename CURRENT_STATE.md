# CURRENT_STATE

> Updated: 2026-08-12T07:47:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0016` 已在恢复场景的 `envs=2/4/8` 阶梯中选择 `envs=4`；当前可从原始
EXP-0012 100K 基线另开正式增量训练，而不混入任一 5K 诊断结果。

## 当前主要矛盾

`envs=4` 稳态 policy FPS 为 `53.84`，较 `envs=2` 提升 `32.33%`，预测新增 100K
约 `34.01` 分钟；`envs=8` 因 16 核 CPU 配额争用退化至 `42.05 FPS` 和 `43.70` 分钟。
因此当前瓶颈在 Minecraft 环境侧 CPU 并发，不在显存容量；独立三回合评测仍按历史约 51 分钟预算。

## 下一项决策

从原始 EXP-0012 100K 源另做独立 CoW 克隆，以 `envs=4` 正式训练到绝对 200K；通过
checkpoint/replay/step 完整性门后，沿用 agent seed 10000 的三回合单环境评测和固定 episode-0 视频。
