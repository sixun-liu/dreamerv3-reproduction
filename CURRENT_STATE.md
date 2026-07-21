# CURRENT_STATE

> Updated: 2026-07-21T13:14:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

旧版作者 runtime `2411f7d` 的单 run 已完整结束：final-30K mean `930.72` 对齐官方 `935.75`，
但 250K median `658.26` 低于官方下限 `826.85`。当前证据支持“终值性能量级可复现”，不支持
“旧代码谱系单独恢复了论文早期样本效率”，也不构成稳定多 seed 数值复现。

## 当前主要矛盾

`EXP-0005` 在100K明显快于2026 runtime三seed，但250K与2026最佳seed0基本相同，并在约370K后
才持续进入官方包络。代码代际影响了局部轨迹和终值，但不足以解释官方曲线的整体领先；主要
不确定性转为未受控DMC环境随机性、历史dm-control/MuJoCo版本和官方score导出流水线。

## 下一项决策

停止自动新增GPU计算。复现主验收回到500K固定预算终值和重复稳定性，曲线形状只作诊断。先人工
审查并独立复算；之后只需在“当前单run证据足够转入论文理解”和“补旧runtime两个独立重复验证
终值稳定性”之间裁决。配对environment seed的谱系归因降为可选研究问题。
