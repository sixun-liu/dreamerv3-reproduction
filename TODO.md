# TODO

> Updated: 2026-08-12T07:47:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 冻结并运行从原始 EXP-0012 到绝对 200K 的新增 100K 正式训练；trigger: EXP-0016 已选择 `envs=4`。
- [ ] [codex] 沿用固定 agent seed 10000 的三回合独立评测与 episode-0 视频；trigger: 200K 完整性通过。
- [ ] [codex] 对比 100K/200K 里程碑、评测、资源账并生成审查/日报材料；trigger: 独立评测完成。

## Waiting

- [ ] [user] 审查 `EXP-0016` 的 `envs=2/4/8` 吞吐和资源图；trigger: `EVT-0100` 已关闭。
- [ ] [user] 审查 `EXP-0012` 训练图、固定第 0 局视频及“L1 成功、L2 未闭合、非论文 Diamond
  复现”口径；trigger: `EVT-0086` 已关闭。
- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
