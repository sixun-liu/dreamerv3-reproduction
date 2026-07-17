# RESULTS_SCOREBOARD

只收录协议一致、可比较的 formal 结果；probe/oracle/instrumentation 不进入正式表。

| Experiment | Protocol | Primary | Tail | Mapping/Visual | Verdict |
|---|---|---:|---:|---|---|
| `EXP-0001` | DMC proprio walker_walk, 12M, repeat2, ratio512, seed0, 500K env steps | 490K bin median 914.2；official range 735.6--955.0 | 250K 显著偏低，约 400K 后进入包络；无 NaN/崩溃 | `artifacts/dreamerv3/EXP-0001/curve_comparison.png` | `promising_unresolved` |
