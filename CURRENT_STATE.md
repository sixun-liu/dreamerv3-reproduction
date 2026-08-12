# CURRENT_STATE

> Updated: 2026-08-12T09:18:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0018 已证明 patched runtime 能从 `100000` 精确结束于 `100040` 并恰好新增 40 个
transition；当前 `envs=4` 是已测最优档，正式重跑前只补一个可达的 `envs=5` 局部对照。

## 当前主要矛盾

EXP-0017 长跑显示 `envs=4` 的 120K--195K policy FPS 中位为 `68.90`，GPU 利用率均值
`16.85%`、中位 `1%`，但显存峰值 `24645 MiB`；这是同步环境与 learner 交替等待，不是显存不足。
EXP-0016 中 `envs=8` 已因 CPU 配额争用退化，故不能靠继续增加环境数填满 GPU。精确停止根因
和补丁已由 EXP-0018 的单测与真实 Minecraft smoke 闭合，原始短 smoke 的 ratio32 日志不足已与
checkpoint/replay 完整性门分离。

## 下一项决策

只比较同一 patched runtime 下的 `envs=4` 与 `envs=5`。若 `envs=5` 未带来至少 `5%` 的稳定
policy FPS 和预测 ETA 收益，则冻结 `envs=4`；随后从原始 EXP-0012 新输出重跑正式 200K，
并沿用固定三回合评测。`script=parallel` 不混入本轮。
