#!/usr/bin/env python3
"""Compute diagnostic throughput and resource gates from one EXP-0013 arm."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def floats(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--historical-wall-fps", type=float, required=True)
    parser.add_argument("--historical-policy-fps", type=float, required=True)
    parser.add_argument("--expand-min-policy-speedup", type=float, default=1.10)
    parser.add_argument("--target-policy-speedup", type=float, default=1.50)
    parser.add_argument("--envs8-max-memory", type=int, default=75161927680)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = load_jsonl(args.run_dir / "train/metrics.jsonl")
    system = load_csv(args.run_dir / "resource_system.csv")
    gpu = load_csv(args.run_dir / "resource_gpu.csv")
    integrity = json.loads((args.run_dir / "integrity.json").read_text(encoding="utf-8"))
    policy_values = floats(metrics, "fps/policy")
    replay_values = floats(metrics, "replay/replay_ratio")
    policy_tail = policy_values[-min(3, len(policy_values)) :]
    replay_tail = replay_values[-min(3, len(replay_values)) :]
    end_to_end_fps = args.steps / args.wall_seconds
    policy_tail_mean = statistics.mean(policy_tail) if policy_tail else 0.0
    policy_speedup = policy_tail_mean / args.historical_policy_fps
    wall_speedup = end_to_end_fps / args.historical_wall_fps

    first = system[0]
    last = system[-1]
    elapsed = max(float(last["timestamp_epoch"]) - float(first["timestamp_epoch"]), 1e-9)
    throttled_usec = int(last["cpu_throttled_usec"]) - int(first["cpu_throttled_usec"])
    nr_periods = int(last["cpu_nr_periods"]) - int(first["cpu_nr_periods"])
    nr_throttled = int(last["cpu_nr_throttled"]) - int(first["cpu_nr_throttled"])
    cpu_usage_usec = int(last["cpu_usage_usec"]) - int(first["cpu_usage_usec"])
    max_memory = max(int(row["cgroup_memory_current_bytes"]) for row in system)
    max_tree_rss = max(int(row["tree_rss_bytes"]) for row in system)
    max_java_rss = max(int(row["java_rss_bytes"]) for row in system)
    max_java_count = max(int(row["java_count"]) for row in system)
    max_temp_dirs = max(int(row["temp_dir_count"]) for row in system)
    system_disk_loss = int(last["system_disk_used_bytes"]) - int(first["system_disk_used_bytes"])
    data_disk_growth = int(last["data_disk_used_bytes"]) - int(first["data_disk_used_bytes"])
    gpu_utils = [float(row["gpu_util_percent"]) for row in gpu]
    gpu_memory = [float(row["memory_used_mib"]) for row in gpu]
    gpu_power = [float(row["power_watts"]) for row in gpu]
    replay_tail_last = replay_tail[-1] if replay_tail else 0.0
    replay_ratio_ok = 28.0 <= replay_tail_last <= 36.0
    resource_ok = (
        max_memory <= args.envs8_max_memory
        and system_disk_loss <= 536870912
        and integrity["no_cgroup_oom"]
    )
    expand_recommended = bool(
        integrity["passed"]
        and resource_ok
        and replay_ratio_ok
        and policy_speedup >= args.expand_min_policy_speedup
    )
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0013",
        "arm": f"envs{args.envs}",
        "diagnostic_only": True,
        "requested_environment_steps": args.steps,
        "wall_seconds": args.wall_seconds,
        "end_to_end_steps_per_second": end_to_end_fps,
        "historical_wall_steps_per_second": args.historical_wall_fps,
        "end_to_end_speedup_vs_historical_long_run": wall_speedup,
        "policy_fps_values": policy_values,
        "policy_fps_tail": policy_tail,
        "policy_fps_tail_mean": policy_tail_mean,
        "historical_policy_fps_mean": args.historical_policy_fps,
        "policy_speedup_vs_historical": policy_speedup,
        "target_policy_speedup_met": policy_speedup >= args.target_policy_speedup,
        "replay_ratio_values": replay_values,
        "replay_ratio_tail": replay_tail,
        "replay_ratio_tail_last": replay_tail_last,
        "replay_ratio_ok": replay_ratio_ok,
        "resource": {
            "sample_count": len(system),
            "sampled_seconds": elapsed,
            "cpu_usage_cores_average": cpu_usage_usec / 1e6 / elapsed,
            "cpu_throttled_usec_delta": throttled_usec,
            "cpu_throttled_wall_fraction": throttled_usec / 1e6 / elapsed,
            "cpu_throttled_period_fraction": nr_throttled / nr_periods if nr_periods else 0.0,
            "max_cgroup_memory_bytes": max_memory,
            "max_tree_rss_bytes": max_tree_rss,
            "max_java_count": max_java_count,
            "max_java_rss_bytes": max_java_rss,
            "max_temp_dir_count": max_temp_dirs,
            "system_disk_loss_bytes": system_disk_loss,
            "data_disk_growth_bytes": data_disk_growth,
            "gpu_samples": len(gpu),
            "gpu_util_mean_percent": statistics.mean(gpu_utils) if gpu_utils else 0.0,
            "gpu_util_peak_percent": max(gpu_utils, default=0.0),
            "gpu_memory_peak_mib": max(gpu_memory, default=0.0),
            "gpu_power_mean_watts": statistics.mean(gpu_power) if gpu_power else 0.0,
        },
        "integrity_passed": integrity["passed"],
        "resource_gate_passed": resource_ok,
        "expand_recommended": expand_recommended,
        "interpretation_boundary": (
            "This short probe selects an execution configuration only; it does not "
            "measure policy quality or reproduce the paper's Minecraft score."
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
