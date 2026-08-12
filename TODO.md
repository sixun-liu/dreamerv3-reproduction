# TODO

> Updated: 2026-08-12T05:08:54Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 冻结 EXP-0013 的数据盘重定向、资源采样、完整性与停止门；trigger: 新 goal 已授权。
- [ ] [codex] 运行多环境 reset/step L0，确认 MineRL 临时实例只落数据盘且进程完整退出；trigger: 工具测试通过。
- [ ] [codex] 运行 `envs=2` 的 5040-step 首探针；trigger: L0 与 launch gate 通过。
- [ ] [codex] 按前档证据决定 `envs=4` 和条件性 `envs=8`，随后关闭吞吐 probe；trigger: 完整性、资源与增益门。
- [ ] [codex] 若吞吐 probe 改变成本判断，另开恢复与增量训练 cycle；trigger: 稳定增益和用户 goal 边界同时满足。

## Waiting

- [ ] [user] 审查 `EXP-0012` 训练图、固定第 0 局视频及“L1 成功、L2 未闭合、非论文 Diamond
  复现”口径；trigger: `EVT-0086` 已关闭。
- [ ] [user] 审查 `EXP-0007` 主图与“共同支持退化 + own-replay advantage”表述；trigger: `EVT-0042` 已关闭。
- [ ] [user] 审查 `EXP-0008` 主图和“同向学习通过、主数值门失败”表述；trigger: `EVT-0050` 已关闭。
- [ ] [user] 审查 `EXP-0009` 主图和单任务机制顺序表述；trigger: `EVT-0057` 已关闭。
- [ ] [user] 审查 `EXP-0011` Breakout 曲线和固定第 0 局视频；trigger: `EVT-0080` 已关闭。
- Figure 18 paired seeds 与完整矩阵保持 parked；跨域长期作业不改变既有结论。
