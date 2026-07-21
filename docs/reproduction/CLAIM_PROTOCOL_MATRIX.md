# DreamerV3 Claim-Protocol Matrix

Updated: 2026-07-21

This document separates paper facts, code-lineage facts, available reference artifacts, and unresolved reproduction choices. Unknown fields are not filled from repository defaults without evidence.

## Paper And Code Ledger

| Item | Frozen fact | Status |
|---|---|---|
| DreamerV3 paper | arXiv `2301.04104`; local PDF SHA256 `d0385798e8bada8e81b915c1743be81d9ce8776f12ed937e09351995e099ce37` | available |
| DreamerV3 source | Local arXiv source SHA256 `012b1d12794056d746027e1d371328fc827f04d9ddfd0a1468644cdfb8b7dc19` | available |
| Runtime code | Author snapshot `e3f0224` plus opt-in instrumentation commits through `5168475`，2026 post-Nature lineage | available；算法语义未改；EGL由runner设置 |
| Code lineage | Author-maintained public reimplementation based on DreamerV2; repository README says it is unrelated to Google or DeepMind | must be reported explicitly |
| Version relation | DMC score 于 2023 commit `423291a` 加入；当前 runtime 是 2026 作者公开重实现 | `author_reimplementation`，显式记录三年代际漂移 |
| DQN bridge paper | arXiv `1312.5602`; local PDF SHA256 `8db04120cace173151c77e0faa6f3eaa4207009da66b9417597dc70bfee56d9c` | reading material available |

## Candidate Result Claims

| Candidate | Paper/reference evidence | Local protocol evidence | Blocking mismatch | Decision |
|---|---|---|---|---|
| DMC proprioceptive `walker_walk` | `scores/dmc_proprio-dreamerv3.json.gz`, 5 seeds；最后3点mean `935.752 -> 936` | `EXP-0004` seeds0/1/2 final-30K means `909.21/695.94/751.43`；aggregate `785.53±90.34` | 2026 runtime与2023 score代码代际；本地原始episode window与官方导出点非完全同构 | `negative`；完整性门通过，aggregate和individual两项性能门失败 |
| DMC visual `walker_walk` | `scores/dmc_vision-dreamerv3.json.gz`, reference curves near 1M | Current `dmc_vision` defaults to the large model, 1.1M steps, replay ratio 256, repeat 1 | Paper table reports 12M model, action repeat 2, replay ratio 512; reference file contains more runs than the paper's stated 5 seeds | secondary candidate |
| Crafter scaling | Paper Figure 4c/4d reports model-size and replay-ratio scaling | Stopped 200M/ratio-512 pilot is healthy and recoverable | Repository has no `crafter*.json.gz`; one configuration cannot reproduce a scaling claim | parked |
| Full DMC suite mean/median | Paper tables and official JSON exist | Environment support exists | Requires many tasks and repeated seeds; cost is not appropriate for the first target | convergence-stage expansion |
| Minecraft diamond | Paper result and official JSON exist | Environment wrapper exists | About 9 A100 GPU-days in the paper; environment/version burden is high | parked |
| DQN Atari result | 2013 paper gives 7-game scores; Breakout average 168 and best 225 after the paper budget | Independent 2013-style implementation under `/root/autodl-tmp/dqn-reproduction`; CleanRL commit `fe8d8a0` is engineering reference only | Modern ALE/runtime, random no-op/FIRE reset, optimizer constants and the paper's ambiguous frame-count semantics prevent exact equivalence | separate single-game conceptual/independent replication |

## Protocol Fields Required Before Reproduction

- [x] Target publication/version: Nature-era paper result using the current author public reimplementation with explicit code-lineage drift.
- [x] Reproduction kind: `author_reimplementation`.
- [x] Target claim: DMC proprio `walker_walk`, paper score 936 at 500K environment steps; reference artifact SHA256 `8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4`.
- [x] Environment: `dm-control==1.0.43`, MuJoCo 3.10.0, vector proprioception, repeat 2, seed 0, 16 envs.
- [x] Budget semantics: logger counter counts agent decisions across envs and multiplies by repeat for output; 250K decisions = 500K environment steps. Replay ratio 512 gives 0.5 gradient updates/decision for batch 16x64.
- [x] Model mapping: runtime `size12m` (deter 2048, hidden/units 256, classes/depth 16); the paper's 1024 recurrent-width cell is recorded as an internal table inconsistency.
- [x] Evaluation: 当前和 2023 `script=train` 均记录 undiscounted training episode score；论文表值为官方五 seed 最后 3 个 10K 点的 mean。`eval_only` 仍采样连续动作，固定 seed checkpoint 评估另列，不混用或称 deterministic。
- [x] Repetitions: `EXP-0004` clean seeds 0、1、2全部完成且未替换；官方参考为 seeds 0--4。
- [x] Acceptance envelope: 预注册aggregate range `[883.942,987.562]`且至少2/3 seed进入官方range；实际aggregate`785.527`、1/3，失败。
- [x] Cost: `EXP-0001` 实测约 1.1 GPU h、显存约 25.6 GB；三 seed clean replication 预计 3--4 GPU h。

## Current Recommendation

`EXP-0004` 已给出有效负复现：工程链闭合但三seed性能包络未复现。下一步只做2023 score对应代码、
配置、dm-control/MuJoCo与seed语义的离线谱系审计；在解释代码/环境漂移前不进入算法消融或新增seed。
