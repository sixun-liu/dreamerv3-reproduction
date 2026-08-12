# TODO

> Updated: 2026-08-12T12:16:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [x] [codex] 完成 Minecraft 终点评测吞吐 probe。EXP-0021 的静态 inventory 顺序伪里程碑已被
  阻断；EXP-0022 采用各环境运行时 `_inv_keys` 并通过 reset 全零、索引有效和跨 worker 一致性门。
  完整三局用时 `796s`，相对串行 `1972s` 为 `2.48x` 墙钟加速，动作归一吞吐为 `1.42x`；后续默认
  使用三环境并行 evaluator，串行脚本保留作回退和抽验。

## Next

- [ ] [codex] 从精确 200K 有界推进到 500K；trigger: EXP-0022 已选择并行 evaluator。固定
  patched runtime `6723fc1`、`envs=4`、`size50m`、ratio32 和源隔离，精确新增 300K 后使用三环境
  并行 evaluator；不得依据中途或单局里程碑选择 checkpoint、视频或提前追加到 1M。
- [ ] [codex] 探索 Minecraft 模型规模的时间到结果性价比；trigger: 评测吞吐 probe 结案且 500K
  路线裁决后。先以 `size50m` 为基线，对 `size100m` 做初始化、显存活跃用量/JAX 预分配区分和约
  5K 同预算吞吐 gate；只有资源安全且成本合理时，才预注册从零、等环境交互预算的 50M/100M
  学习对照，比较墙钟、每步训练成本、里程碑出现率与 wall-time-to-milestone。不得把模型更大直接
  等同于训练更快，也不在 100M 未通过前测试 200M/400M。当前 `size50m` 实际 optimizer 参数
  `46,812,213`，JAX 约 `24.6 GiB` 显存分配主要含预分配，不能据此推断 100M 一定可行或必然 OOM。

## Waiting

- [ ] [user] 审查 `EXP-0016` 的 `envs=2/4/8` 吞吐和资源图；trigger: `EVT-0100` 已关闭。
- [ ] [user] 审查 `EXP-0012` 训练图、固定第 0 局视频及“L1 成功、L2 未闭合、非论文 Diamond
  复现”口径；trigger: `EVT-0086` 已关闭。
- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
