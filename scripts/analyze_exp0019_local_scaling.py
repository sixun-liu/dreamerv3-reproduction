#!/usr/bin/env python3
"""Compare EXP-0019 patched-runtime envs=4 and envs=5 throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_exp0016_recovery_scaling import compare, summarize_arm


def analyze(
    run_root: Path,
    minimum_steady_speedup: float = 1.05,
    minimum_eta_reduction: float = 0.05,
    minimum_samples: int = 5,
    maximum_memory_bytes: int = 77_309_411_328,
) -> dict:
    incumbent = summarize_arm(run_root / "envs4", 4, 100_000, 5_040, minimum_samples)
    candidate = summarize_arm(run_root / "envs5", 5, 100_000, 5_040, minimum_samples)
    comparison = compare(
        incumbent,
        candidate,
        minimum_steady_speedup,
        minimum_eta_reduction,
        minimum_samples,
        maximum_memory_bytes,
    )
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0019",
        "diagnostic_only": True,
        "arms": {"envs4": incumbent, "envs5": candidate},
        "comparison": comparison,
        "interpretation_boundary": (
            "This probe selects the local script=train time-to-result optimum under fixed "
            "algorithm semantics. It does not evaluate policy quality, reward, or the "
            "separate script=parallel architecture."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--minimum-steady-speedup", type=float, default=1.05)
    parser.add_argument("--minimum-eta-reduction", type=float, default=0.05)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--maximum-memory-bytes", type=int, default=77_309_411_328)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.run_root,
        args.minimum_steady_speedup,
        args.minimum_eta_reduction,
        args.minimum_samples,
        args.maximum_memory_bytes,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["comparison"]["comparison_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
