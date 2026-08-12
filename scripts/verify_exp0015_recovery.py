#!/usr/bin/env python3
"""Verify isolated DreamerV3 recovery without assuming a single replay chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cloudpickle
import numpy as np
import yaml

from prepare_exp0015_recovery import indexed, latest_checkpoint, sha256, tree_manifest
from verify_exp0013_throughput import ERROR_MARKERS, unexpected_nonfinite_metrics


ALLOWED_CONFIG_CHANGES = {
    "logdir",
    "run.debug",
    "run.envs",
    "run.log_every",
    "run.report_every",
    "run.save_every",
    "run.steps",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix}
    if isinstance(left, dict):
        result: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(changed_paths(left[key], right[key], path))
        return result
    if isinstance(left, list):
        return set() if left == right else {prefix}
    return set() if left == right else {prefix}


def replay_rows(root: Path) -> tuple[dict[bytes, bool], int]:
    rows: dict[bytes, bool] = {}
    total = 0
    for path in sorted(root.glob("*.npz")):
        with np.load(path) as chunk:
            stepids = chunk["stepid"]
            is_first = chunk["is_first"]
            if stepids.ndim != 2 or stepids.shape[1] != 20:
                raise ValueError(f"Unexpected stepid shape in {path}: {stepids.shape}")
            if len(stepids) != len(is_first):
                raise ValueError(f"Replay field length mismatch in {path}")
            total += len(stepids)
            for stepid, first in zip(stepids, is_first, strict=True):
                key = stepid.tobytes()
                if key in rows:
                    raise ValueError(f"Duplicate stepid: {key.hex()}")
                rows[key] = bool(first)
    return rows, total


def live_process_state() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_pids = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    exact_processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in {"java", "Xvfb"}:
            exact_processes.append({"pid": int(entry.name), "comm": comm})
    return {
        "nvidia_smi_exit_code": gpu.returncode,
        "gpu_compute_pids": gpu_pids,
        "java_or_xvfb": exact_processes,
        "passed": gpu.returncode == 0 and not gpu_pids and not exact_processes,
    }


def matching_original_files(source_manifest: dict, target_root: Path) -> dict:
    source = indexed(source_manifest)
    missing = []
    changed = []
    for name, expected in source.items():
        path = target_root / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            changed.append(name)
    return {
        "expected": len(source),
        "missing": missing,
        "changed": changed,
        "passed": not missing and not changed,
    }


def verify(args: argparse.Namespace) -> dict:
    train = args.run_dir / "train"
    required = (
        args.clone_manifest,
        args.frozen_config,
        args.source_config,
        args.stdout_log,
        args.resource_system,
        train / "config.yaml",
        train / "metrics.jsonl",
        train / "scores.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    clone = json.loads(args.clone_manifest.read_text(encoding="utf-8"))
    source_checkpoint_after = tree_manifest(args.source_train / "ckpt")
    source_replay_after = tree_manifest(args.source_train / "replay")
    source_unchanged = {
        "checkpoint": (
            source_checkpoint_after["tree_sha256"]
            == clone["source"]["checkpoint"]["tree_sha256"]
        ),
        "replay": (
            source_replay_after["tree_sha256"]
            == clone["source"]["replay"]["tree_sha256"]
        ),
    }

    source_config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    generated = yaml.safe_load((train / "config.yaml").read_text(encoding="utf-8"))
    config_differences = changed_paths(source_config, frozen)
    config_checks = {
        "runtime_matches_frozen": generated == frozen,
        "only_allowed_changes": config_differences == ALLOWED_CONFIG_CHANGES,
        "changed_paths": sorted(config_differences),
        "task": generated.get("task") == "minecraft_diamond",
        "seed": generated.get("seed") == 0,
        "envs": generated.get("run", {}).get("envs") == 2,
        "debug_false": generated.get("run", {}).get("debug") is False,
        "absolute_final_step": generated.get("run", {}).get("steps") == float(args.final_step),
        "train_ratio": generated.get("run", {}).get("train_ratio") == 32.0,
        "local_checkpoint_load": generated.get("run", {}).get("from_checkpoint") == "",
    }

    checkpoint = latest_checkpoint(train)
    checkpoint_step = int(cloudpickle.loads((checkpoint / "step.pkl").read_bytes()))
    source_agent = clone["source"]["checkpoint"]["files"]
    source_agent_hash = next(
        row["sha256"] for row in source_agent if row["path"].endswith("/agent.pkl")
    )
    final_agent_hash = sha256(checkpoint / "agent.pkl")

    source_rows, source_total = replay_rows(args.source_train / "replay")
    target_rows, target_total = replay_rows(train / "replay")
    source_ids = set(source_rows)
    target_ids = set(target_rows)
    new_ids = target_ids - source_ids
    new_first_ids = [stepid for stepid in new_ids if target_rows[stepid]]
    new_first_prefixes = {stepid[:16] for stepid in new_first_ids}
    replay_checks = {
        "source_rows": source_total,
        "source_unique_stepids": len(source_ids),
        "target_rows": target_total,
        "target_unique_stepids": len(target_ids),
        "source_stepids_preserved": source_ids <= target_ids,
        "new_unique_stepids": len(new_ids),
        "expected_new_stepids": args.final_step - args.source_step,
        "new_is_first_count": len(new_first_ids),
        "new_is_first_chunk_prefix_count": len(new_first_prefixes),
        "two_new_worker_starts": len(new_first_prefixes) >= 2,
        "original_replay_files": matching_original_files(
            clone["source"]["replay"], train / "replay"
        ),
    }

    metrics = load_jsonl(train / "metrics.jsonl")
    metric_steps = [int(row["step"]) for row in metrics if "step" in row]
    loss_rows = [row for row in metrics if any(key.startswith("train/loss/") for key in row)]
    replay_ratios = [
        float(row["replay/replay_ratio"])
        for row in metrics
        if isinstance(row.get("replay/replay_ratio"), (int, float))
        and math.isfinite(float(row["replay/replay_ratio"]))
    ]
    unexpected = unexpected_nonfinite_metrics(metrics)
    metric_checks = {
        "rows": len(metrics),
        "minimum_step": min(metric_steps, default=None),
        "maximum_step": max(metric_steps, default=None),
        "only_post_source_steps": bool(metric_steps)
        and min(metric_steps) > args.source_step
        and max(metric_steps) <= args.final_step,
        "has_training_loss": bool(loss_rows),
        "unexpected_nonfinite": unexpected,
        "finite": bool(metrics) and not unexpected,
        "finite_replay_ratios": replay_ratios,
        "ratio32_observed": any(28.0 <= value <= 36.0 for value in replay_ratios),
    }

    system = load_csv(args.resource_system)
    first_system = system[0] if system else {}
    last_system = system[-1] if system else {}
    oom_delta = int(last_system.get("memory_events_oom", 0)) - int(
        first_system.get("memory_events_oom", 0)
    )
    oom_kill_delta = int(last_system.get("memory_events_oom_kill", 0)) - int(
        first_system.get("memory_events_oom_kill", 0)
    )
    disk_loss = int(last_system.get("system_disk_used_bytes", 0)) - int(
        first_system.get("system_disk_used_bytes", 0)
    )
    cleanup_path = args.run_dir / "temp_cleanup_postprocess.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.is_file() else {}
    temp_remaining = list(args.temp_root.iterdir()) if args.temp_root.is_dir() else []
    resource_checks = {
        "samples": len(system),
        "max_cgroup_memory_bytes": max(
            (int(row["cgroup_memory_current_bytes"]) for row in system), default=0
        ),
        "max_java_count": max((int(row["java_count"]) for row in system), default=0),
        "max_temp_dir_count": max(
            (int(row["temp_dir_count"]) for row in system), default=0
        ),
        "system_disk_loss_bytes": disk_loss,
        "system_disk_loss_within_limit": disk_loss <= args.max_system_disk_loss,
        "oom_delta": oom_delta,
        "oom_kill_delta": oom_kill_delta,
        "no_oom": oom_delta == 0 and oom_kill_delta == 0,
        "cleanup": cleanup,
        "temp_remaining": [str(path) for path in temp_remaining],
        "temp_clean": bool(cleanup.get("passed")) and not temp_remaining,
    }
    process_checks = {"passed": True, "skipped": True}
    if not getattr(args, "skip_live_process_check", False):
        process_checks = live_process_state()

    stdout = args.stdout_log.read_text(encoding="utf-8", errors="replace")
    recovery_markers = {
        marker: marker in stdout
        for marker in (
            "Found existing checkpoint.",
            "Loading checkpoint:",
            "Loaded checkpoint.",
            "Start training loop",
        )
    }
    error_markers = [marker for marker in ERROR_MARKERS if marker in stdout]
    checks = {
        "schema_version": 1,
        "experiment_id": "EXP-0015",
        "clone_preflight_passed": clone.get("passed") is True,
        "source_unchanged": source_unchanged,
        "config": config_checks,
        "checkpoint": {
            "path": str(checkpoint),
            "step": checkpoint_step,
            "step_matches": checkpoint_step == args.final_step,
            "source_agent_sha256": source_agent_hash,
            "final_agent_sha256": final_agent_hash,
            "agent_updated": final_agent_hash != source_agent_hash,
        },
        "replay": replay_checks,
        "metrics": metric_checks,
        "resources": resource_checks,
        "postrun_processes": process_checks,
        "recovery_markers": recovery_markers,
        "error_markers": error_markers,
    }
    checks["passed"] = bool(
        checks["clone_preflight_passed"]
        and all(source_unchanged.values())
        and config_checks["runtime_matches_frozen"]
        and config_checks["only_allowed_changes"]
        and all(
            config_checks[key]
            for key in (
                "task",
                "seed",
                "envs",
                "debug_false",
                "absolute_final_step",
                "train_ratio",
                "local_checkpoint_load",
            )
        )
        and checks["checkpoint"]["step_matches"]
        and checks["checkpoint"]["agent_updated"]
        and replay_checks["source_rows"] == args.source_step
        and replay_checks["source_unique_stepids"] == args.source_step
        and replay_checks["target_rows"] == args.final_step
        and replay_checks["target_unique_stepids"] == args.final_step
        and replay_checks["source_stepids_preserved"]
        and replay_checks["new_unique_stepids"] == replay_checks["expected_new_stepids"]
        and replay_checks["two_new_worker_starts"]
        and replay_checks["original_replay_files"]["passed"]
        and metric_checks["only_post_source_steps"]
        and metric_checks["has_training_loss"]
        and metric_checks["finite"]
        and metric_checks["ratio32_observed"]
        and resource_checks["samples"] >= 2
        and resource_checks["max_java_count"] >= 2
        and resource_checks["max_temp_dir_count"] >= 2
        and resource_checks["system_disk_loss_within_limit"]
        and resource_checks["no_oom"]
        and resource_checks["temp_clean"]
        and process_checks["passed"]
        and all(recovery_markers.values())
        and not error_markers
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--clone-manifest", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--resource-system", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-system-disk-loss", type=int, default=536870912)
    parser.add_argument("--skip-live-process-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
