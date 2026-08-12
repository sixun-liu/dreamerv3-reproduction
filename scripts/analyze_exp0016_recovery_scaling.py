#!/usr/bin/env python3
"""Compare bounded Minecraft recovery throughput across envs=2, 4, and 8."""

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
    run_dir: Path,
    envs: int,
    source_step: int,
    added_steps: int,
    minimum_samples: int,
) -> dict:
    metrics = load_jsonl(run_dir / "train/metrics.jsonl")
    system = load_csv(run_dir / "resource_system.csv")
    gpu = load_csv(run_dir / "resource_gpu.csv")
    completed = json.loads((run_dir / ".completed").read_text(encoding="utf-8"))
    integrity = json.loads((run_dir / "integrity.json").read_text(encoding="utf-8"))
    lower = source_step + added_steps * 0.20
    upper = source_step + added_steps * 0.90
    steady_rows = [
        row
        for row in metrics
        if lower <= float(row.get("step", -1)) <= upper
        and math.isfinite(float(row.get("fps/policy", math.nan)))
        and 28.0 <= float(row.get("replay/replay_ratio", math.nan)) <= 36.0
    ]
    policy = [float(row["fps/policy"]) for row in steady_rows]
    replay = [float(row["replay/replay_ratio"]) for row in steady_rows]
    training_wall = float(completed["training_wall_seconds"])
    steady_mean = statistics.mean(policy) if policy else 0.0
    startup_seconds = max(0.0, training_wall - added_steps / steady_mean) if steady_mean else math.inf
    predicted_100k_seconds = (
        startup_seconds + 100_000 / steady_mean if steady_mean else math.inf
    )
    first = system[0]
    last = system[-1]
    sampled_seconds = max(
        float(last["timestamp_epoch"]) - float(first["timestamp_epoch"]), 1e-9
    )
    gpu_util = [float(row["gpu_util_percent"]) for row in gpu]
    gpu_memory = [float(row["memory_used_mib"]) for row in gpu]
    cpu_cores = (
        int(last["cpu_usage_usec"]) - int(first["cpu_usage_usec"])
    ) / 1e6 / sampled_seconds
    return {
        "arm": f"envs{envs}",
        "environment_count": envs,
        "source_step": source_step,
        "added_environment_steps": added_steps,
        "training_wall_seconds": training_wall,
        "end_to_end_steps_per_second": added_steps / training_wall,
        "startup_seconds_estimate": startup_seconds,
        "predicted_added_100k_seconds": predicted_100k_seconds,
        "predicted_added_100k_minutes": predicted_100k_seconds / 60,
        "steady_window": {
            "lower_step_inclusive": lower,
            "upper_step_inclusive": upper,
            "selection": "finite fps/policy and replay ratio in [28,36]",
            "minimum_samples": minimum_samples,
            "sample_count": len(policy),
            "steps": [int(row["step"]) for row in steady_rows],
            "policy_fps_values": policy,
            "policy_fps_mean": steady_mean,
            "policy_fps_median": statistics.median(policy) if policy else 0.0,
            "replay_ratio_values": replay,
        },
        "integrity_passed": bool(integrity.get("passed")),
        "resource": {
            "sample_count": len(system),
            "sampled_seconds": sampled_seconds,
            "cpu_usage_cores_average": cpu_cores,
            "cpu_efficiency_policy_fps_per_core": steady_mean / cpu_cores if cpu_cores else 0.0,
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
            "max_temp_dir_count": max(int(row["temp_dir_count"]) for row in system),
            "system_disk_loss_bytes": int(last["system_disk_used_bytes"])
            - int(first["system_disk_used_bytes"]),
            "data_disk_growth_bytes": int(last["data_disk_used_bytes"])
            - int(first["data_disk_used_bytes"]),
            "gpu_util_mean_percent": statistics.mean(gpu_util) if gpu_util else 0.0,
            "gpu_util_peak_percent": max(gpu_util, default=0.0),
            "gpu_memory_peak_mib": max(gpu_memory, default=0.0),
        },
    }


def compare(
    incumbent: dict,
    candidate: dict,
    minimum_steady_speedup: float,
    minimum_eta_reduction: float,
    minimum_samples: int,
    maximum_memory_bytes: int,
) -> dict:
    steady_speedup = (
        candidate["steady_window"]["policy_fps_mean"]
        / incumbent["steady_window"]["policy_fps_mean"]
    )
    eta_reduction = 1.0 - (
        candidate["predicted_added_100k_seconds"]
        / incumbent["predicted_added_100k_seconds"]
    )
    comparable = bool(
        incumbent["integrity_passed"]
        and candidate["integrity_passed"]
        and incumbent["added_environment_steps"]
        == candidate["added_environment_steps"]
        and incumbent["steady_window"]["sample_count"] >= minimum_samples
        and candidate["steady_window"]["sample_count"] >= minimum_samples
    )
    resource_safe = bool(
        candidate["resource"]["max_cgroup_memory_bytes"] <= maximum_memory_bytes
        and candidate["resource"]["max_java_count"] >= candidate["environment_count"]
        and candidate["resource"]["max_temp_dir_count"] >= candidate["environment_count"]
    )
    promoted = bool(
        comparable
        and resource_safe
        and steady_speedup >= minimum_steady_speedup
        and eta_reduction >= minimum_eta_reduction
    )
    return {
        "incumbent": incumbent["arm"],
        "candidate": candidate["arm"],
        "comparison_valid": comparable,
        "resource_safe": resource_safe,
        "steady_policy_speedup": steady_speedup,
        "predicted_100k_eta_reduction": eta_reduction,
        "selection_gates": {
            "minimum_samples_per_arm": minimum_samples,
            "minimum_steady_policy_speedup": minimum_steady_speedup,
            "minimum_predicted_100k_eta_reduction": minimum_eta_reduction,
            "maximum_cgroup_memory_bytes": maximum_memory_bytes,
        },
        "candidate_promoted": promoted,
        "selected_environment_count": (
            candidate["environment_count"] if promoted else incumbent["environment_count"]
        )
        if comparable
        else None,
        "verdict": (
            "candidate_promoted"
            if promoted
            else "incumbent_retained"
            if comparable
            else "inconclusive"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-envs", type=int, choices=(4, 8), required=True)
    parser.add_argument("--source-step", type=int, default=100_000)
    parser.add_argument("--added-steps", type=int, default=5_040)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--minimum-steady-speedup", type=float, required=True)
    parser.add_argument("--minimum-eta-reduction", type=float, required=True)
    parser.add_argument("--maximum-memory-bytes", type=int, default=77309411328)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    incumbent_envs = 2 if args.candidate_envs == 4 else 4
    incumbent = summarize_arm(
        args.baseline_run,
        incumbent_envs,
        args.source_step,
        args.added_steps,
        args.minimum_samples,
    )
    candidate = summarize_arm(
        args.candidate_run,
        args.candidate_envs,
        args.source_step,
        args.added_steps,
        args.minimum_samples,
    )
    comparison = compare(
        incumbent,
        candidate,
        args.minimum_steady_speedup,
        args.minimum_eta_reduction,
        args.minimum_samples,
        args.maximum_memory_bytes,
    )
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0016",
        "diagnostic_only": True,
        "arms": {incumbent["arm"]: incumbent, candidate["arm"]: candidate},
        "comparison": comparison,
        "interpretation_boundary": (
            "This diagnostic selects time-to-result under fixed algorithm semantics; "
            "it does not evaluate policy quality or reward."
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not comparison["comparison_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
