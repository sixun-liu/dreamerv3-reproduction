# EXP-0006 KL Mechanism Protocol

> Updated: 2026-07-21T14:23:46Z
> Maintainer: codex
> Source of truth: `research/experiments.jsonl` and the EXP-0006 freeze event

## Scope

This is a controlled mechanism study on the 2026 author reimplementation, not a numerical
reproduction of paper Figure 6 or Figure 17. All arms use DMC `walker_walk`, proprioception,
`size12m`, action repeat 2, 16 environments, replay ratio 512, and 250K agent decisions equal to
500K environment steps.

The environment wrapper explicitly receives deterministic per-worker seeds. Matching seeds align
random sources and initial conditions across arms; they do not imply identical trajectories after
the policies diverge.

## Arms

| Arm | `free_nats` | dynamics KL scale | representation KL scale | Meaning |
|---|---:|---:|---:|---|
| baseline | 1.0 | 1.0 | 0.1 | Full current DreamerV3 control |
| E1 | 0.0 | 1.0 | 0.1 | Remove free bits only |
| P4-reconstructed | 0.0 | 1.0 | 1.0 | Remove free bits and unequal KL gradient weighting |

P4 is a code-semantic reconstruction of `No KL balance & free bits`. The public paper and repository
do not provide the original complete ablation config, so this arm is not `exact_artifact` evidence.
With split stop-gradient KL terms, equal unit scales reproduce the parameter gradients of an
unbalanced KL objective; the reported summed scalar loss is not itself directly comparable across
arms.

## Preregistered Predictions

1. E1 versus baseline: removing the one-nat floor should reduce late raw KL and increase the
   fractions of samples below fixed KL thresholds in both paired seeds.
2. P4 versus E1: increasing the representation-side KL gradient from 0.1 to 1.0 should further push
   posterior representations toward the prior, visible in raw KL and/or posterior entropy.
3. Score effects are secondary and directionally open. A mechanism can be active without changing
   walker performance; walker cannot establish cross-domain scale robustness.

A `posterior collapse candidate` requires late-window raw KL below 0.1 nat and more than 90% of
samples below 0.1 nat. It is called harmful only if accompanied by at least 20% lower paired
final-window score or at least 20% higher reconstruction loss than baseline.

## Metrics And Windows

- Raw KL mean/std and fractions below 0.1, 0.5, and 1.0 nat.
- Prior and posterior entropy; vector reconstruction loss is the sum of orientations, height, and
  velocity reconstruction losses.
- Training episode score in 10K environment-step bins.
- Normalized score AUC for 0--250K and 250--500K, plus `(470K, 500K]` final-window score.
- Fixed `eval_only` seed 10000 checkpoint score is reported separately from training-return metrics.
- Early mechanism window: `(0, 250K]`; late mechanism window: `(400K, 500K]`.

## Integrity And Stop Rules

- Every arm/seed must finish naturally with exact checkpoint step 250000, finite metrics and scores,
  semantically matching expanded config, and at least 64 finite eval episodes.
- No arm is stopped or modified because of score or KL outcomes.
- Stop the matrix on NaN/Inf, OOM, traceback, duplicate GPU process, checkpoint/config mismatch, or
  less than 10 GiB free on the data disk. Engineering failures are retained and never silently retried.
- Formal completion requires baseline/E1/P4 seeds 0 and 1. A third seed is optional and cannot replace
  either preregistered seed.
