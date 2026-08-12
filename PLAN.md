# PLAN

> Updated: 2026-08-12T05:08:54Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在现有硬件和存储条件下建立可复现、资源安全且高效的 DreamerV3 Minecraft 训练方案，
  并依据实测性价比闭环适当规模的增量训练、评测与材料。
- 当前问题：以 EXP-0012 为单环境基线，在 16 核/90 GiB 容器和 86 GiB 数据盘边界内建立资源安全、
  可恢复且具备实测吞吐收益的 Minecraft 运行方案，再按收益选择有界增量预算。

## 阶段退出门

- [x] EXP-0010 DMC Vision 完成 100K gate、1M 续训、独立评测和图像材料。
- [x] EXP-0011 Atari100K Breakout 完成 100K decisions、独立评测和 DQN 协议差异表。
- [x] EXP-0012 Minecraft 完成依赖/L0、size50m smoke、100K 训练、独立评测与审查材料。
- [ ] Minecraft 临时实例、日志和可控缓存不再消耗系统盘，且多环境 L0 通过。
- [ ] 完成自适应多环境吞吐探针，找到稳定甜点或形成证据充分的无增益裁决。
- [ ] 若探针通过成本门，完成 EXP-0012 等价恢复、一个有界增量训练及独立评测；否则明确停车。
- [ ] 固化可复用运行脚本、资源账、恢复方法、展示材料和受限结论。

## 活动路线

1. `EXP-0012` 保持 canonical baseline；新吞吐探针是诊断工作，不覆盖或混入原训练结果。
2. 首阶段仅改变环境并发数和运行时存储位置，不改变模型、task、reward、wrapper、ratio 或 batch。
3. 按 `2 -> 4 -> 条件性 8` 自适应扩档，每档 5040 累计 environment steps；无增益或资源异常即停。
4. 只有吞吐、replay ratio、checkpoint、CPU/RAM/Java 和磁盘门均通过，才进入恢复验证。
5. 后续训练预算由实测 ETA、磁盘成本和学习价值共同决定；训练、评测与展示材料必须独立闭环。
6. `script=parallel` 仅在普通 `train` 多环境收益不足时作为独立候选，不假设跨模式可恢复。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 论文尺度预算；本轮只允许证据驱动的有界增量训练。
- `script=parallel` 的有限停止、三部分 checkpoint 与跨模式恢复验证。
