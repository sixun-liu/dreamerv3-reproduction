# CURRENT_STATE

> Updated: 2026-08-06T09:54:43Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0009 Reacher Hard 三臂 pilot 全部自然完成；真实模型梯度 gate 证明 intervention 语义正确，
1M environment-step fixed-bin AUC 为 `867.42 > 391.61 > 7.84`，通过 Figure 18 定性方向门。
证据权限仅为单任务、单配对 seed、论文十分之一预算的机制信号。

## 当前主要矛盾

当前已从“消融到底阻断什么梯度”的实现不确定性推进到“该排序是否跨 seed 稳定”的统计不确定性。
本轮结果足以支持继续验证，但不足以声称复现论文 14-task Figure 18；直接升到 10M 或全矩阵的成本收益
仍不合理。

## 下一项决策

计算已停止，GPU 空闲。用户先审查 `EXP-0009` 学习曲线、contact sheet 与受限表述；若继续，唯一
下一计算候选是在相同 1M 预算下补 paired seeds 1、2，先排除 seed 0 偶然，再讨论 10M 或跨任务扩展。
