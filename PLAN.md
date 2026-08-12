# PLAN

> Updated: 2026-08-12T11:07:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在现有硬件和存储条件下建立可复现、资源安全且高效的 DreamerV3 Minecraft 训练方案，
  并依据实测性价比闭环适当规模的增量训练、评测与材料。
- 当前问题：训练侧已固定 patched runtime `6723fc1` + `envs=4`；下一步降低固定三回合终点评测的
  32.87 分钟墙钟，并在协议等价性通过后再选择 500K 增量预算。

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

## 活动路线

1. `EXP-0012` 保持 canonical 100K baseline；EXP-0020 是精确 200K active candidate，不改写历史基线。
2. 同步训练固定 patched runtime `6723fc1`、`envs=4`、ratio32 和数据盘隔离；已测 `2/4/5/8`
   档不再扩搜，也不以显存占满为目标。
3. EXP-0020 精确新增 100K 用时 `32.18` 分钟；工程完整性全绿，新增 replay 与固定评测均出现
   wooden pickaxe，但 cobblestone 为 0，因此只支持木镐推进，不支持完整 L2 或论文 Diamond 主张。
4. 下一 cycle 只改变 evaluator 的环境调度：短 smoke 后配对比较串行 3 局与并行 3 环境；固定
   checkpoint、agent seed、每局 36K 上限、逐局统计和 episode-0 视频选择。
5. 并行 evaluator 需至少 `1.5x` 端到端加速且通过 episode、视频、资源与清场门；否则保留串行协议。
6. 500K 增量等待评测 probe 结案；不得按当前木镐正例提前追加预算或选择成功视频。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 论文尺度预算；500K 也等待评测吞吐 probe 后再预注册。
- `script=parallel` 的有限停止、三部分 checkpoint 与跨模式恢复验证。
- NUMA/taskset、CPU 亲和性和共享编译缓存微调；当前新增 100K 仅约 29--31 分钟，收益不足以
  抵消额外系统复杂度和可迁移性成本。
