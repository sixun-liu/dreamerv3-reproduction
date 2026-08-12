# CURRENT_STATE

> Updated: 2026-08-12T06:57:54Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` 已完成 100K 受限闭环；Minecraft 临时实例已安全迁移到数据盘。`EXP-0014` 的
同预算配对诊断选择 `envs=2`，当前只需验证 EXP-0012 checkpoint/replay/step 的等价恢复。

## 当前主要矛盾

同为 5040 步与 `debug=false` 时，`envs=2` 相对 `envs=1` 的端到端吞吐提升 `27.36%`，固定稳态
policy FPS 提升 `46.10%`，明显通过 `10%/5%` 双门；CPU、内存和磁盘代价仍在资源边界内。
尚未证明的是从 EXP-0012 终点恢复时，checkpoint、replay 与 environment-step 语义是否保持等价。

## 下一项决策

在隔离输出中做一个最小 `envs=2` 恢复 smoke，核验起止 step、checkpoint、replay transition 集合、
metrics 连续性和资源清场；通过后再按实测 ETA 冻结有界增量训练预算。
