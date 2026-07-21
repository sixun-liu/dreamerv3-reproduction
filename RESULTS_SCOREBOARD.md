# RESULTS_SCOREBOARD

> Updated: 2026-07-21T21:45:00Z
> Maintainer: codex
> Source of truth: research/experiments.jsonl and research/artifacts.jsonl

只收录协议一致、可比较的 formal 结果；probe/oracle/instrumentation 不进入正式表。

| Experiment | Protocol | Primary | Tail | Mapping/Visual | Verdict |
|---|---|---:|---:|---|---|
| `EXP-0001` | DMC proprio walker_walk, 12M, repeat2, ratio512, seed0, 500K env steps | 490K bin median 914.2；official range 735.6--955.0 | 250K 显著偏低，约 400K 后进入包络；无 NaN/崩溃 | `artifacts/dreamerv3/EXP-0001/curve_comparison.png` | `promising_unresolved` |
| `EXP-0004` | Author reimplementation；walker_walk；12M；repeat2；ratio512；seeds0,1,2；500K env steps | final-30K means 909.21/695.94/751.43；aggregate 785.53 vs official 935.75 | local seed std90.34；250K三seed全低于官方；checkpoint eval 913.90/781.85/765.16 | `/root/autodl-tmp/artifacts/dreamerv3/review/EXP-0004-walker-three-seed/` | `negative`；完整性通过、两项性能门失败 |
| `EXP-0005` | Author runtime `2411f7d`+compat；walker_walk；12M；repeat2；ratio512；agent seed0；500K env steps | final-30K mean930.72、median937.94；official mean935.75 | 250K median658.26低于官方826.85--945.81；约370K后持续入包络；DMC env seed未受控 | `/root/autodl-tmp/artifacts/dreamerv3/review/EXP-0005-runtime-2411-walker/` | `promising_unresolved`；终值门通过、早期门失败 |
| `EXP-0006` | Author 2026 runtime `ad49802`；seeded walker_walk；baseline/E1/P4-reconstructed；paired seeds0,1；500K env steps | E1 vs baseline late raw-KL delta -0.803/+0.337，双seed门失败；P4 vs E1 KL ratio0.0399/0.0161且entropy双seed更低，P4机制门通过 | reconstruction ratio2.99/2.46；final-30K baseline903.37/833.47、E1 638.62/616.56、P4 466.52/445.43；无严格collapse candidate | `/root/autodl-tmp/artifacts/dreamerv3/review/EXP-0006-walker-kl-three-arm/` | `promising_unresolved`；P4机制强支持，E1 raw-KL假说未复现；非论文Figure6/17数值复现 |
