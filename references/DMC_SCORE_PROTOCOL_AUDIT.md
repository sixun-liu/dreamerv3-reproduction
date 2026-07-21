# DMC proprio 官方分数谱系与聚合协议审计

> Updated: 2026-07-21T04:53:18Z
> Maintainer: codex
> Source of truth: paper source, author repository history, frozen score JSON, and EXP-0001 raw metrics

## 结论

论文 Table 4 的 DMC proprio 单任务分数可由公开五 seed 曲线的最后 3 个 10K-step 点取平均恢复。
对表中 18 个任务逐项复算，预测值与论文整数表值的 RMSE 为 `0.30`；`walker_walk` 为
`935.752 -> 936`。因此 936 的聚合口径视为已闭合，不再列为 unknown。

公开曲线高度可能是 training episode return，而非独立 deterministic evaluation：2023 代码的
`dmc_proprio` 使用默认 `script=train`，对应训练循环对每个 episode 的 reward 求和并记录
`episode/score`。当前代码的 `script=train` 保留相同的人读指标语义。仓库未提供从原始日志导出
JSON 的完整脚本，因此保留“导出流水线未公开”这一限制。

## Score 文件谱系

| 项目 | 事实 |
|---|---|
| 原始加入 | Author commit [`423291a`](https://github.com/danijar/dreamerv3/commit/423291a9875bb9af43b6db7150aaa972ba889266)，2023-05，`Add scores` |
| 原始路径 | `scores/data/dmcproprio_dreamerv3.json.gz` |
| 迁移提交 | Author commit [`2411f7d`](https://github.com/danijar/dreamerv3/commit/2411f7d136832378c0291c587cdbf2fca6506873)，2024-04，agent/infrastructure 大幅更新 |
| 当前路径 | `/root/autodl-tmp/dreamerv3/scores/dmc_proprio-dreamerv3.json.gz` |
| 当前文件 SHA256 | `8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4` |
| 数值一致性 | 2023 与当前文件的 `walker_walk` 五 seed、49 个 x/y 点数值一致；格式和 seed 类型发生变化 |
| 本地 runtime | `e3f02248693a79dc8b0ebd62c93683888ddaccfe`，2026-05，晚于曲线生成约三年 |

这说明参考曲线本身可追溯到论文时期，但本地执行器是 post-Nature 作者重实现，不能归类为
`exact_artifact`。

## Table 4 聚合恢复

对每个任务构造：

```text
score(task, n) = mean(seed 0..4, final n curve points)
```

并与论文 Table 4 Dreamer 列的 18 个整数比较：

| final n points | 全任务 RMSE | 平均偏差 |
|---:|---:|---:|
| 1 | 25.21 | +4.29 |
| 2 | 11.58 | +3.90 |
| 3 | **0.30** | **-0.01** |
| 4 | 19.30 | -4.76 |
| 10 | 21.00 | -10.44 |

`n=3` 时 18 个任务全部在四舍五入误差内。例如：

| Task | 论文 | 官方 JSON 最后 3 点五 seed 均值 |
|---|---:|---:|
| `walker_walk` | 936 | 935.752 |
| `walker_run` | 649 | 648.631 |
| `walker_stand` | 964 | 964.369 |
| `cheetah_run` | 614 | 613.613 |

官方 x 轴间隔为 10K environment steps，因此表格统计对应五 seed 最后约 30K-step 曲线窗口。

## EXP-0001 的同口径观察

`EXP-0001` 因 16 个同步环境的 episode 成组结束，在 `(470K, 500K]` 只有 32 个完整 episodes，
没有与官方 470K/480K/490K 三个 bin 完全一一对应的非空本地点。直接对该窗口原始 episodes 聚合：

- mean `891.713`；
- median `902.313`；
- std `54.813`；
- 相对论文 936 低 `4.73%`。

此前报告的 490K 单 bin median `914.217` 仍是正确观察，但不能直接替代论文的五 seed、三点平均。

## 剩余限制与后续协议

1. 2023 score 曲线与 2026 runtime 存在明确代码代际差异。
2. `EXP-0001` 只有 seed 0，不能估计重复性。
3. `EXP-0001` 最新 checkpoint 位于 462,720 environment steps，不是训练终点。
4. JSON 的原始日志导出脚本未公开；“training episode return”有强代码证据，但不是完整 artifact。

后续 multi-seed replication 的主指标使用每 seed 最后 30K environment-step 完整 episode mean，
再跨预注册 seeds 汇总 mean/std；同时单独报告固定 checkpoint `eval_only`，不把 deterministic
evaluation 与论文 training-return 表值混为同一指标。

