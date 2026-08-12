# CURRENT_STATE

> Updated: 2026-08-12T09:44:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0019 已排除 `envs=5`：它相对 `envs=4` 稳态 FPS 低 `6.67%`、预测新增 100K 时间
高 `6.51%`，因此 `envs=4` 是当前 `script=train` 路线已测 `2/4/5/8` 档的实用最优。

## 当前主要矛盾

EXP-0019 的同预算对照中，`envs=4/5` 稳态 FPS 分别为 `61.39/57.30`，预测新增 100K
分别为 `30.33/32.30` 分钟。`envs=5` 平均 CPU 从 `7.60` 增至 `8.80` 核，throttled
CPU-sec/wall-sec 从 `2.44` 增至 `3.84`，峰值内存从 `28.70` 增至 `34.07 GiB`；两臂显存峰值
同为约 `24.64 GiB`。这说明瓶颈是本地 Minecraft/同步 Driver 的 CPU 与尾延迟，而不是显存容量。

## 下一项决策

从原始 EXP-0012 新输出，以 patched runtime `6723fc1`、`envs=4` 和独立冷缓存精确重跑到
200K；完整性通过后沿用固定三回合评测。`script=parallel`、NUMA/taskset 和更多环境数不混入本轮。
