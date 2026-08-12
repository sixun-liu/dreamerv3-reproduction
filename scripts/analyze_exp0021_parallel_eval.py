#!/usr/bin/env python3
"""Verify and summarize the EXP-0021 parallel Minecraft evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import av
import numpy as np


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def inspect_video(path: Path) -> dict[str, object]:
    frame_count = 0
    dynamic_pairs = 0
    previous = None
    codec = None
    width = None
    height = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        width = stream.codec_context.width
        height = stream.codec_context.height
        for frame in container.decode(stream):
            image = frame.to_ndarray(format="rgb24")
            if previous is not None and np.any(previous != image):
                dynamic_pairs += 1
            previous = image
            frame_count += 1
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "dynamic_adjacent_pairs": dynamic_pairs,
        "passed": bool(
            codec == "h264"
            and width == 64
            and height == 64
            and frame_count >= 2
            and dynamic_pairs > 0
        ),
    }


def summarize_resources(system: list[dict[str, str]], gpu: list[dict[str, str]]) -> dict:
    if len(system) < 2:
        raise ValueError("Resource sampler did not collect at least two system rows")
    first, last = system[0], system[-1]
    sampled_seconds = max(
        float(last["timestamp_epoch"]) - float(first["timestamp_epoch"]), 1e-9
    )
    gpu_util = [float(row["gpu_util_percent"]) for row in gpu]
    gpu_memory = [float(row["memory_used_mib"]) for row in gpu]
    return {
        "sample_count": len(system),
        "sampled_seconds": sampled_seconds,
        "average_cpu_cores": (
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
        "peak_cgroup_memory_bytes": max(
            int(row["cgroup_memory_current_bytes"]) for row in system
        ),
        "peak_tree_rss_bytes": max(int(row["tree_rss_bytes"]) for row in system),
        "peak_java_count": max(int(row["java_count"]) for row in system),
        "peak_temp_dir_count": max(int(row["temp_dir_count"]) for row in system),
        "system_disk_loss_bytes": int(last["system_disk_used_bytes"])
        - int(first["system_disk_used_bytes"]),
        "data_disk_growth_bytes": int(last["data_disk_used_bytes"])
        - int(first["data_disk_used_bytes"]),
        "oom_delta": int(last["memory_events_oom"])
        - int(first["memory_events_oom"]),
        "oom_kill_delta": int(last["memory_events_oom_kill"])
        - int(first["memory_events_oom_kill"]),
        "mean_gpu_util_percent": statistics.fmean(gpu_util) if gpu_util else 0.0,
        "peak_gpu_util_percent": max(gpu_util, default=0.0),
        "peak_gpu_memory_mib": max(gpu_memory, default=0.0),
    }


def analyze(
    evaluation: dict,
    completed: dict,
    system: list[dict[str, str]],
    gpu: list[dict[str, str]],
    cleanup: dict,
    video: dict,
    *,
    expected_checkpoint_agent_sha256: str,
    expected_checkpoint_step_sha256: str,
    expected_episode_length: int,
    serial_wall_seconds: float,
    serial_active_actions: int,
    minimum_wall_speedup: float,
    maximum_memory_bytes: int,
) -> dict:
    episodes = evaluation.get("episodes", [])
    workers = [int(row.get("worker", -1)) for row in episodes]
    episode_boundaries_ok = bool(
        len(episodes) == 3
        and workers == [0, 1, 2]
        and all(int(row.get("is_first_count", 0)) == 1 for row in episodes)
        and all(int(row.get("is_last_count", 0)) == 1 for row in episodes)
        and all(
            0 <= int(row.get("actions_excluding_reset", -1)) <= expected_episode_length
            for row in episodes
        )
        and int(evaluation.get("extra_episode_count", -1)) == 0
    )
    hold_counts = {
        int(key): int(value)
        for key, value in evaluation.get("worker_hold_counts", {}).items()
    }
    callback_counts = {
        int(key): int(value)
        for key, value in evaluation.get("worker_callback_counts", {}).items()
    }
    hold_accounting_ok = bool(
        sorted(hold_counts) == [0, 1, 2]
        and sorted(callback_counts) == [0, 1, 2]
        and all(
            callback_counts[worker]
            == int(episodes[worker]["transitions_including_reset"])
            + hold_counts[worker]
            for worker in range(3)
        )
    ) if len(episodes) == 3 else False
    checkpoint_ok = bool(
        evaluation.get("checkpoint_agent_sha256")
        == expected_checkpoint_agent_sha256
        and evaluation.get("checkpoint_step_sha256")
        == expected_checkpoint_step_sha256
    )
    protocol_ok = bool(
        evaluation.get("experiment_id") == "EXP-0021"
        and evaluation.get("agent_seed") == 10000
        and evaluation.get("environment_count") == 3
        and evaluation.get("episode_count") == 3
        and evaluation.get("episode_length") == expected_episode_length
        and evaluation.get("video_worker") == 0
        and evaluation.get("video_stride") == 4
        and evaluation.get("video_selection")
        == "worker 0 / episode index 0 preregistered; not best-of-N"
    )
    returns_finite = all(math.isfinite(float(row.get("return", math.nan))) for row in episodes)
    resources = summarize_resources(system, gpu)
    resource_safe = bool(
        resources["peak_cgroup_memory_bytes"] <= maximum_memory_bytes
        and resources["peak_java_count"] >= 3
        and resources["oom_delta"] == 0
        and resources["oom_kill_delta"] == 0
        and resources["system_disk_loss_bytes"] <= 536_870_912
        and cleanup.get("passed") is True
    )
    total_wall_seconds = float(completed["total_wall_seconds"])
    active_actions = int(evaluation["active_actions"])
    wall_speedup = serial_wall_seconds / total_wall_seconds
    serial_actions_per_second = serial_active_actions / serial_wall_seconds
    parallel_actions_per_second = active_actions / total_wall_seconds
    normalized_action_speedup = parallel_actions_per_second / serial_actions_per_second
    integrity_passed = bool(
        episode_boundaries_ok
        and hold_accounting_ok
        and checkpoint_ok
        and protocol_ok
        and returns_finite
        and video.get("passed") is True
        and resource_safe
    )
    promoted = bool(integrity_passed and wall_speedup >= minimum_wall_speedup)
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0021",
        "diagnostic_only": True,
        "integrity": {
            "passed": integrity_passed,
            "episode_boundaries_ok": episode_boundaries_ok,
            "hold_accounting_ok": hold_accounting_ok,
            "checkpoint_ok": checkpoint_ok,
            "protocol_ok": protocol_ok,
            "returns_finite": returns_finite,
            "video_ok": video.get("passed") is True,
            "resource_safe": resource_safe,
        },
        "timing": {
            "serial_total_wall_seconds": serial_wall_seconds,
            "parallel_total_wall_seconds": total_wall_seconds,
            "wall_speedup": wall_speedup,
            "minimum_wall_speedup": minimum_wall_speedup,
            "serial_active_actions": serial_active_actions,
            "parallel_active_actions": active_actions,
            "serial_active_actions_per_second": serial_actions_per_second,
            "parallel_active_actions_per_second": parallel_actions_per_second,
            "normalized_active_action_speedup": normalized_action_speedup,
            "parallel_synchronized_cycles": evaluation["synchronized_cycles"],
            "parallel_worker_hold_counts": evaluation["worker_hold_counts"],
        },
        "resource": resources,
        "video": video,
        "parallel_evaluator_promoted": promoted,
        "verdict": (
            "parallel_eval_promoted"
            if promoted
            else "serial_eval_retained"
            if integrity_passed
            else "inconclusive_integrity_failure"
        ),
        "interpretation_boundary": (
            "The wall-clock gate selects an execution path only. Uncontrolled Minecraft "
            "worlds make returns, milestones, episode lengths, and total actions descriptive; "
            "they are not paired policy-quality evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-agent-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step-sha256", required=True)
    parser.add_argument("--expected-episode-length", type=int, required=True)
    parser.add_argument("--serial-wall-seconds", type=float, default=1972.0)
    parser.add_argument("--serial-active-actions", type=int, default=55979)
    parser.add_argument("--minimum-wall-speedup", type=float, default=1.5)
    parser.add_argument("--maximum-memory-bytes", type=int, default=77309411328)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    evaluation = json.loads(
        (run / "evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    completed = json.loads((run / ".completed").read_text(encoding="utf-8"))
    cleanup = json.loads(
        (run / "temp_cleanup_postprocess.json").read_text(encoding="utf-8")
    )
    video = inspect_video(run / "evaluation/episode_000_preregistered_stride4.mp4")
    result = analyze(
        evaluation,
        completed,
        load_csv(run / "resource_system.csv"),
        load_csv(run / "resource_gpu.csv"),
        cleanup,
        video,
        expected_checkpoint_agent_sha256=args.expected_checkpoint_agent_sha256,
        expected_checkpoint_step_sha256=args.expected_checkpoint_step_sha256,
        expected_episode_length=args.expected_episode_length,
        serial_wall_seconds=args.serial_wall_seconds,
        serial_active_actions=args.serial_active_actions,
        minimum_wall_speedup=args.minimum_wall_speedup,
        maximum_memory_bytes=args.maximum_memory_bytes,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["integrity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
