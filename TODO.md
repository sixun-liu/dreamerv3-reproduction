# TODO

> Updated: 2026-08-12T11:07:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 建立 Minecraft 终点评测吞吐 probe；trigger: EXP-0020 评测与材料闭环后。先用短回合
  smoke 验证三环境并行 evaluator 的逐环境终止、回报/里程碑统计、固定 episode-0 视频、checkpoint
  绑定和清场语义，再以同一冻结 checkpoint、agent seed、3 回合和 36K 上限比较串行与并行墙钟。
  只有并行方案端到端至少加速 `1.5x`，且无 episode 丢失、best-of-N、资源越界或协议漂移，才晋级为
  后续 500K/1M checkpoint 的默认评测路径；x264 preset 和 video stride 仅作次要写入开销对照。

## Next

- [ ] [codex] 依据评测 probe 裁决是否从精确 200K 有界推进到 500K；trigger: 并行 evaluator
  晋级或串行协议明确保留。不得因本轮 wooden_pickaxe 正例自动追加训练预算。
- [ ] [codex] 探索 Minecraft 模型规模的时间到结果性价比；trigger: 评测吞吐 probe 结案且 500K
  路线裁决后。先以 `size50m` 为基线，对 `size100m` 做初始化、显存活跃用量/JAX 预分配区分和约
  5K 同预算吞吐 gate；只有资源安全且成本合理时，才预注册从零、等环境交互预算的 50M/100M
  学习对照，比较墙钟、每步训练成本、里程碑出现率与 wall-time-to-milestone。不得把模型更大直接
  等同于训练更快，也不在 100M 未通过前测试 200M/400M。

## Waiting

- [ ] [user] 审查 `EXP-0016` 的 `envs=2/4/8` 吞吐和资源图；trigger: `EVT-0100` 已关闭。
- [ ] [user] 审查 `EXP-0012` 训练图、固定第 0 局视频及“L1 成功、L2 未闭合、非论文 Diamond
  复现”口径；trigger: `EVT-0086` 已关闭。
- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
