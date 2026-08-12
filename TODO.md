# TODO

> Updated: 2026-08-12T06:35:22Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 审计 runtime 的 `from_checkpoint`、step 终点与 replay 恢复语义；trigger: EXP-0014 已关闭。
- [ ] [codex] 另开 EXP-0012 checkpoint/replay/step 等价恢复 cycle；trigger: 恢复协议审计完成。
- [ ] [codex] 运行隔离的 `envs=2` 恢复 smoke 并核验 transition 连续性；trigger: freeze 与 launch gate 通过。
- [ ] [codex] 依据恢复实测 ETA 冻结有界增量预算、独立评测和材料；trigger: 恢复完整性通过。

## Waiting

- [ ] [user] 审查 `EXP-0012` 训练图、固定第 0 局视频及“L1 成功、L2 未闭合、非论文 Diamond
  复现”口径；trigger: `EVT-0086` 已关闭。
- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
