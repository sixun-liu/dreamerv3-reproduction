# PLAN

> Updated: 2026-08-12T08:44:00Z
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
- [x] Minecraft 临时实例、日志和可控缓存不再消耗系统盘，且多环境 L0 通过。
- [x] 完成同预算单/双环境吞吐归因，选择资源安全的 `envs=2` 候选配置。
- [x] 完成 EXP-0012 checkpoint/replay/step 的隔离等价恢复与资源验证。
- [x] 完成恢复场景 `envs=2/4/8` 阶梯，选择 `envs=4` 为当前硬件的最高性价比档。
- [ ] 修复并以低成本真实恢复 smoke 验证多环境精确停止；EXP-0017 的 `200008` 输出不晋级。
- [ ] 完成一个有界增量训练及独立评测；若完整性或成本门失败则明确停车。
- [ ] 固化可复用运行脚本、资源账、恢复方法、展示材料和受限结论。

## 活动路线

1. `EXP-0012` 保持 canonical baseline；新吞吐探针是诊断工作，不覆盖或混入原训练结果。
2. EXP-0016 已选择 `envs=4`：相对恢复基线稳态 FPS 提升 `32.33%`、预测新增 100K ETA
   缩短 `22.02%`；`envs=8` 因 CPU 配额争用被淘汰。该结果仅用于执行配置。
3. EXP-0015 已通过隔离恢复门；正式 cycle 必须从原始 EXP-0012 重新克隆，不续接诊断 105040 输出。
4. EXP-0017 因固定 10-step Driver 调用在 `envs=4` 下结束于 `200008`，已按
   `invalid_provenance` 关闭且未评测；先验证 remaining-step 停止修复，再用新输出重跑。
5. 有界增量预算仍固定为新增 100K、绝对终点 200K；任何 step 越界、重复/丢失、源漂移或
   checkpoint 不完整均先停止。
6. 沿用三回合单环境独立评测与固定 episode-0 视频；世界种子不可控，不做逐世界配对因果主张。
7. 训练、评测、资源账和展示材料必须独立闭环；`script=parallel` 与更深系统调优继续 parked。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 论文尺度预算；本轮只允许证据驱动的有界增量训练。
- `script=parallel` 的有限停止、三部分 checkpoint 与跨模式恢复验证。
