# CURRENT_STATE

> Updated: 2026-08-05T22:08:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0008 五 seed Cheetah Run 全部自然完成且同向学习，但 final-30K 聚合 `550.54`
低于官方终值范围下限 `584.00`；当前作者重实现在第二个 DMC 任务上复现了学习行为，
没有通过论文 Table 11 主数值门。

## 当前主要矛盾

Walker 单 seed 终值对齐与 Cheetah 五 seed 数值门失败同时存在，说明继续盲目加任务不会自动解决
2023 参考产物、2024 runtime、未受控 DMC env RNG 和未公开导出流水线之间的谱系缺口。
下一阶段应转向论文原生机制问题，而非继续追单条曲线。

## 下一项决策

计算已停止。用户先审查 `EXP-0008` 主图与受限负复现表述；后续唯一优先候选是
Figure 18 learning-signal 消融。新计算前必须先恢复“阻断任务信号塑形表征”与“阻断重建信号塑形表征”
的精确梯度语义，并用参数组梯度 known-answer test 验证。
