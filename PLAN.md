# PLAN

> Updated: 2026-08-12T12:16:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在现有硬件和存储条件下建立可复现、资源安全且高效的 DreamerV3 Minecraft 训练方案，
  并依据实测性价比闭环适当规模的增量训练、评测与材料。
- 当前问题：训练侧已固定 patched runtime `6723fc1` + `envs=4`，评测侧已选择可信的三环境并行
  evaluator；下一步从精确 200K 有界训练到 500K，再独立探索 `size50m/size100m` 时间到结果性价比。

## 阶段退出门

- [x] EXP-0010 DMC Vision 完成 100K gate、1M 续训、独立评测和图像材料。
- [x] EXP-0011 Atari100K Breakout 完成 100K decisions、独立评测和 DQN 协议差异表。
- [x] EXP-0012 Minecraft 完成依赖/L0、size50m smoke、100K 训练、独立评测与审查材料。
- [x] Minecraft 临时实例、日志和可控缓存不再消耗系统盘，且多环境 L0 通过。
- [x] 完成同预算单/双环境吞吐归因，选择资源安全的 `envs=2` 候选配置。
- [x] 完成 EXP-0012 checkpoint/replay/step 的隔离等价恢复与资源验证。
- [x] 完成恢复场景 `envs=2/4/8` 阶梯，选择 `envs=4` 为当前硬件的最高性价比档。
- [x] 修复并以低成本真实恢复 smoke 验证多环境精确停止；EXP-0017 的 `200008` 输出不晋级。
- [x] 完成 `envs=4/5` 局部缩放对照并冻结当前硬件的实用最优档；不追求显存占满。
- [x] 完成一个有界增量训练及独立评测；精确 200K、木镐推进和无圆石边界均已闭环。
- [x] 固化可复用运行脚本、资源账、恢复方法、展示材料和受限结论。
- [x] 修复运行时 inventory 里程碑映射，并将固定三回合评测从 `1972s` 降至 `796s`。

## 活动路线

1. `EXP-0012` 保持 canonical 100K baseline；EXP-0020 是精确 200K active candidate，不改写历史基线。
2. 同步训练固定 patched runtime `6723fc1`、`envs=4`、ratio32 和数据盘隔离；已测 `2/4/5/8`
   档不再扩搜，也不以显存占满为目标。
3. EXP-0020 精确新增 100K 用时 `32.18` 分钟；工程完整性全绿，新增 replay 与固定评测均出现
   wooden pickaxe，但 cobblestone 为 0，因此只支持木镐推进，不支持完整 L2 或论文 Diamond 主张。
4. EXP-0021 因外部静态 inventory 顺序造成伪里程碑，以 `invalid_provenance` 结案；其调度、资源和
   视频局部证据保留，但不得引用其里程碑或 green verdict。
5. EXP-0022 改由每个环境的运行时 `_inv_keys` 导出里程碑，完整性全绿；完整三局 `796s`，墙钟
   `2.48x`、动作归一吞吐 `1.42x`，后续默认采用并行 evaluator，串行路径保留作抽验。
6. 下一 cycle 固定 `size50m`，从精确 200K 新增 300K 到 500K；模型规模是之后的独立变量，不与
   增量训练混在同一 cycle。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 论文尺度预算；500K 是下一项有界增量，完成后再裁决是否继续。
- 模型 `size100m/200m/400m` 正式学习对照；先做 100M 初始化、活跃内存和约 5K 吞吐 gate。
- `script=parallel` 的有限停止、三部分 checkpoint 与跨模式恢复验证。
- NUMA/taskset、CPU 亲和性和共享编译缓存微调；当前新增 100K 仅约 29--31 分钟，收益不足以
  抵消额外系统复杂度和可迁移性成本。
