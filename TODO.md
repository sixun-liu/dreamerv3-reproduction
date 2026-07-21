# TODO

> Updated: 2026-07-21T14:23:46Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 预注册并冻结 `EXP-0006` 三臂两 seed 矩阵；trigger: runtime smoke 与 hash 已通过。
- [ ] [codex] 顺序完成六条 500K run；trigger: GPU 空闲且所有 freeze/started 信标就绪。
- [ ] [codex] 生成 raw-KL、entropy/reconstruction、score/AUC 配对图表并闭环实验。

## Waiting

- [ ] [user] 审查 `EXP-0006` 紧凑主图；trigger: 六条主 run 与分析完成。
- 第三配对 seed 仅在主矩阵闭环、磁盘与时间预算允许时启动。
