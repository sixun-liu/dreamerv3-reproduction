# CURRENT_STATE

> Updated: 2026-08-12T08:44:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`envs=4` 仍是当前吞吐最优档，但 EXP-0017 的正式 200K 门因同步 Driver 确定性结束于
`200008` 而失效；当前先修复并验证精确停止，不能评测或采用该越界 checkpoint。

## 当前主要矛盾

EXP-0017 训练主体在 `28.66` 分钟完成，120K--195K policy FPS 中位 `68.90`，峰值内存
`31.37 GiB`、峰值显存 `24645 MiB` 且无 OOM，进一步确认瓶颈不在显存容量。但 `train.py`
每轮固定请求 10 步，4 环境同步 Driver 每轮实际推进 12 步，导致
`100000 + ceil(100000/12) * 12 = 200008`；严格完整性门已正确拒绝该输出。

## 下一项决策

在独立 runtime 中把最后一轮 Driver 请求限制为剩余步数，先以单测和真实 `envs=4` 恢复 smoke
证明精确终点；通过后才从原始 EXP-0012 新输出重跑正式 200K，并沿用冻结的三回合评测。
