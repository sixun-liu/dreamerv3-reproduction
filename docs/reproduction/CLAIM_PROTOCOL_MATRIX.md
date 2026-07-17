# DreamerV3 Claim-Protocol Matrix

Updated: 2026-07-17

This document separates paper facts, code-lineage facts, available reference artifacts, and unresolved reproduction choices. Unknown fields are not filled from repository defaults without evidence.

## Paper And Code Ledger

| Item | Frozen fact | Status |
|---|---|---|
| DreamerV3 paper | arXiv `2301.04104`; local PDF SHA256 `d0385798e8bada8e81b915c1743be81d9ce8776f12ed937e09351995e099ce37` | available |
| DreamerV3 source | Local arXiv source SHA256 `012b1d12794056d746027e1d371328fc827f04d9ddfd0a1468644cdfb8b7dc19` | available |
| Runtime code | `danijar/dreamerv3` commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe` | available |
| Code lineage | Author-maintained public reimplementation based on DreamerV2; repository README says it is unrelated to Google or DeepMind | must be reported explicitly |
| Version relation | Current code is the Nature-era author public reimplementation and differs from the 2023 implementation lineage | target fixed to Nature-era result with explicit lineage drift |
| DQN bridge paper | arXiv `1312.5602`; local PDF SHA256 `8db04120cace173151c77e0faa6f3eaa4207009da66b9417597dc70bfee56d9c` | reading material available |

## Candidate Result Claims

| Candidate | Paper/reference evidence | Local protocol evidence | Blocking mismatch | Decision |
|---|---|---|---|---|
| DMC proprioceptive `walker_walk` | `scores/dmc_proprio-dreamerv3.json.gz`, 5 seeds, reference x-axis 10K--490K; paper Table 4 reports 500K budget and score 936 | Runtime overrides current defaults with `size12m`, DMC repeat 2, 16 envs, ratio 512 and 250K agent decisions = 500K environment steps | Author public reimplementation is newer than the paper code; paper table's 12M recurrent width 1024 conflicts with its own 8d rule and runtime width 2048 | selected first replication; single seed can at most be `promising_unresolved` |
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
- [x] Evaluation: current `script=train` records undiscounted training episode score; compare 10K environment-step bins to the supplied per-seed reference curves. It is not an independent deterministic evaluation.
- [x] Repetitions: seed 0 only in the 12-hour cycle; official reference has seeds 0--4. No failed seed replacement.
- [x] Acceptance envelope: engineering success requires healthy metrics/checkpoint; partial scientific success requires an increasing curve entering the official seed envelope. A single seed cannot support `promote`.
- [ ] Cost: paper reports 0.3 A100 GPU-day; local 5090 throughput, memory, checkpoint size and ETA will be measured at the first 10K environment-step checkpoint before continuation.

## Current Recommendation

Freeze DMC proprioceptive `walker_walk` as the first replication. Start the full target configuration with a
10K-environment-step pilot gate; continue only if measured ETA fits the remaining wall-time budget. Report
training-return comparability and code-lineage drift explicitly.
