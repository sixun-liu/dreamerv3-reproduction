# CURRENT_STATE

> Updated: 2026-07-27T04:31:50Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0007 已把 EXP-0006 的现象转化为共同数据支持上的功能诊断：P4 在两个训练 seed 中均呈现
更低 posterior-prior KL 与更差 prior/teacher-forced prediction 并存；独立复算通过，但三个模型
各自在自身 replay 来源上表现最好，访问分布效应不可忽略。

## 当前主要矛盾

共同 panel 上的 pooled 退化不是少数窗口或单一 observation key 驱动，但当前交叉矩阵不能区分
representation specialization、各策略访问分布难度与训练目标的贡献。证据权限仍是 walker_walk、
两 seed、固定真实 action 的 offline probe，不支持闭环因果、跨任务或论文 Figure 6/17 主张。

## 下一项决策

计算已停止。先由用户审查 `EXP-0007` 主图和受限结论；若继续机制归因，唯一下一判别问题是：
在匹配或交换训练访问分布后，P4 相对 E1 的 teacher-forced 退化是否仍跨两个 seed 成立。
不得自动启动新实验。
