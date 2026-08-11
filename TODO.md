# TODO

> Updated: 2026-08-11T23:58:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 冻结 `EXP-0012` 并运行真实 Minecraft L0；trigger: 独立环境与静态兼容门已通过。
- [ ] [codex] L0 通过后运行 4096-step size50m smoke；trigger: checkpoint/replay/有限损失及资源门。
- [ ] [codex] smoke gate 通过后训练至硬上限 100K 并完成评测/材料；trigger: 不越过当前授权。

## Waiting

- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
