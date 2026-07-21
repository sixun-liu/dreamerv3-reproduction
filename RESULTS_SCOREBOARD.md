# RESULTS_SCOREBOARD

> Updated: 2026-07-21T10:20:00Z
> Maintainer: codex
> Source of truth: research/experiments.jsonl and research/artifacts.jsonl

只收录协议一致、可比较的 formal 结果；probe/oracle/instrumentation 不进入正式表。

| Experiment | Protocol | Primary | Tail | Mapping/Visual | Verdict |
|---|---|---:|---:|---|---|
| `EXP-0001` | DMC proprio walker_walk, 12M, repeat2, ratio512, seed0, 500K env steps | 490K bin median 914.2；official range 735.6--955.0 | 250K 显著偏低，约 400K 后进入包络；无 NaN/崩溃 | `artifacts/dreamerv3/EXP-0001/curve_comparison.png` | `promising_unresolved` |
| `EXP-0004` | Author reimplementation；walker_walk；12M；repeat2；ratio512；seeds0,1,2；500K env steps | final-30K means 909.21/695.94/751.43；aggregate 785.53 vs official 935.75 | local seed std90.34；250K三seed全低于官方；checkpoint eval 913.90/781.85/765.16 | `/root/autodl-tmp/artifacts/dreamerv3/review/EXP-0004-walker-three-seed/` | `negative`；完整性通过、两项性能门失败 |
