# TODO

> Updated: 2026-07-21T05:51:00Z
> Maintainer: codex
> Source of truth: manual action view; long-lived tasks use research/tasks.jsonl

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [ ] [codex] 验证自然结束 final checkpoint 和独立评估管道；trigger: eval-only 诊断完成。
- [ ] [claude] 独立核对 2023 DMC 配置与 Table 4 聚合；trigger: 读取 `references/DMC_SCORE_PROTOCOL_AUDIT.md`。

## Waiting

- [ ] [codex] 冻结并顺序运行三个 clean seeds；trigger: final-checkpoint smoke 通过。
- 后续 GPU 实验不自动续跑。
