# TODO

> Updated: 2026-08-11T20:02:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 建立 Atari 独立环境并冻结 `EXP-0011`；trigger: `EXP-0010` 已关闭且 GPU 空闲。
- [ ] [codex] 依次完成 ALE L0、2048-decision smoke 和 100K-decision 正式运行；trigger: 上一层 gate 通过。
- [ ] [codex] 关闭 `EXP-0011` 后建立 Minecraft 独立环境并执行 `EXP-0012` L0；trigger: Atari 已审计提交。
- [ ] [codex] Minecraft L0/资源门通过时训练至硬上限 100K；trigger: 不越过当前授权。

## Waiting

- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
