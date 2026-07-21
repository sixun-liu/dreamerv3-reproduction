# CURRENT_STATE

> Updated: 2026-07-21T10:20:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

DreamerV3 `walker_walk` 三 seed clean replication 已完成：工程与checkpoint/eval完整性全绿，但
final-30K aggregate `785.53±90.34` 低于官方 `935.75±25.90`，预注册性能门失败。当前结论是
post-Nature作者重实现的有效负复现，不是工程失败或整篇论文否定。

## 当前主要矛盾

三个本地seed均在250K显著落后官方，seed0末段追入，seed1/2未达到官方最终范围；独立checkpoint
eval维持同样排序。主要不确定性已从“单seed偶然性”收敛为2023到2026代码/环境代际、seed语义和
未公开score导出流水线的差异。

## 下一项决策

停止新增GPU计算，先离线恢复2023参考曲线对应runtime/config与dm-control/MuJoCo seed语义，找出
能区分代码代际和环境协议的最便宜证据。人工审查两张EXP-0004主图后再决定是否需要旧代码pilot。
