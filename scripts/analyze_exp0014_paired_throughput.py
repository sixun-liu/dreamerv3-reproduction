#!/usr/bin/env python3
"""Compare equal-budget envs=1 and envs=2 Minecraft throughput evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_arm(
    metrics: list[dict],
    system: list[dict[str, str]],
    gpu: list[dict[str, str]],
    completed: dict,
    integrity: dict,
    expected_steps: int,
) -> dict:
    lower = expected_steps * 0.40
    upper = expected_steps * 0.85
    steady_rows = [
        row
        for row in metrics
        if lower <= float(row.get("step", -1)) <= upper
        and math.isfinite(float(row.get("fps/policy", math.nan)))
        and 28.0 <= float(row.get("replay/replay_ratio", math.nan)) <= 36.0
    ]
    policy = [float(row["fps/policy"]) for row in steady_rows]
    replay = [float(row["replay/replay_ratio"]) for row in steady_rows]
    first = system[0]
    last = system[-1]
    sampled_seconds = max(
        float(last["timestamp_epoch"]) - float(first["timestamp_epoch"]), 1e-9
    )
    gpu_util = [float(row["gpu_util_percent"]) for row in gpu]
    gpu_memory = [float(row["memory_used_mib"]) for row in gpu]
    return {
        "arm": completed["arm"],
        "requested_environment_steps": expected_steps,
        "training_wall_seconds": completed["training_wall_seconds"],
        "end_to_end_steps_per_second": expected_steps
        / float(completed["training_wall_seconds"]),
        "steady_window": {
            "lower_step_inclusive": lower,
            "upper_step_inclusive": upper,
            "selection": "finite fps/policy and replay ratio in [28,36]",
            "sample_count": len(policy),
            "policy_fps_values": policy,
            "policy_fps_mean": statistics.mean(policy) if policy else 0.0,
            "policy_fps_median": statistics.median(policy) if policy else 0.0,
            "replay_ratio_values": replay,
        },
        "integrity_passed": bool(integrity.get("passed")),
        "resource": {
            "sample_count": len(system),
            "sampled_seconds": sampled_seconds,
            "cpu_usage_cores_average": (
                int(last["cpu_usage_usec"]) - int(first["cpu_usage_usec"])
            )
            / 1e6
            / sampled_seconds,
            "cpu_throttled_wall_fraction": (
                int(last["cpu_throttled_usec"])
                - int(first["cpu_throttled_usec"])
            )
            / 1e6
            / sampled_seconds,
            "max_cgroup_memory_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in system
            ),
            "max_java_count": max(int(row["java_count"]) for row in system),
            "system_disk_loss_bytes": int(last["system_disk_used_bytes"])
            - int(first["system_disk_used_bytes"]),
            "data_disk_growth_bytes": int(last["data_disk_used_bytes"])
            - int(first["data_disk_used_bytes"]),
            "gpu_util_mean_percent": statistics.mean(gpu_util) if gpu_util else 0.0,
            "gpu_util_peak_percent": max(gpu_util, default=0.0),
            "gpu_memory_peak_mib": max(gpu_memory, default=0.0),
        },
    }


def compare(envs1: dict, envs2: dict) -> dict:
    comparable = bool(
        envs1["integrity_passed"]
        and envs2["integrity_passed"]
        and envs1["requested_environment_steps"]
        == envs2["requested_environment_steps"]
        and envs1["steady_window"]["sample_count"] >= 3
        and envs2["steady_window"]["sample_count"] >= 3
    )
    end_to_end_speedup = (
        envs2["end_to_end_steps_per_second"]
        / envs1["end_to_end_steps_per_second"]
    )
    steady_speedup = (
        envs2["steady_window"]["policy_fps_mean"]
        / envs1["steady_window"]["policy_fps_mean"]
    )
    envs2_preferred = bool(
        comparable and end_to_end_speedup >= 1.10 and steady_speedup >= 1.05
    )
    return {
        "comparison_valid": comparable,
        "end_to_end_speedup_envs2_over_envs1": end_to_end_speedup,
        "steady_policy_speedup_envs2_over_envs1": steady_speedup,
        "selection_gates": {
            "minimum_end_to_end_speedup": 1.10,
            "minimum_steady_policy_speedup": 1.05,
        },
        "selected_environment_count": 2 if envs2_preferred else (1 if comparable else None),
        "envs2_preferred": envs2_preferred,
        "verdict": (
            "envs2_preferred"
            if envs2_preferred
            else "envs1_preferred"
            if comparable
            else "inconclusive"
        ),
    }


def load_arm(run_dir: Path, expected_steps: int) -> dict:
    return summarize_arm(
        load_jsonl(run_dir / "train/metrics.jsonl"),
        load_csv(run_dir / "resource_system.csv"),
        load_csv(run_dir / "resource_gpu.csv"),
        json.loads((run_dir / ".completed").read_text(encoding="utf-8")),
        json.loads((run_dir / "integrity.json").read_text(encoding="utf-8")),
        expected_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs1-run", type=Path, required=True)
    parser.add_argument("--envs2-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5040)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    envs1 = load_arm(args.envs1_run, args.steps)
    envs2 = load_arm(args.envs2_run, args.steps)
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0014",
        "diagnostic_only": True,
        "arms": {"envs1": envs1, "envs2": envs2},
        "comparison": compare(envs1, envs2),
        "interpretation_boundary": (
            "This paired diagnostic selects a local execution configuration only; "
            "it does not measure policy quality or reproduce a Minecraft paper score."
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["comparison"]["comparison_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
