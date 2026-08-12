#!/usr/bin/env python3
"""Verify one EXP-0013 arm without judging its throughput benefit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cloudpickle
import yaml


ERROR_MARKERS = (
    "Traceback",
    "AssertionError",
    "Out of memory",
    "CUDA_ERROR",
    "ResourceExhaustedError",
    "FAILED_PRECONDITION",
    "Terminating workers due to an exception",
)
ALLOWED_NONFINITE_KEYS = {
    "replay/replay_ratio",
    "replay/insert_wait_avg",
    "replay/insert_wait_frac",
    "replay/sample_wait_avg",
    "replay/sample_wait_frac",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_latest_checkpoint(train_dir: Path) -> Path:
    checkpoint_root = train_dir / "ckpt"
    latest_file = checkpoint_root / "latest"
    if not latest_file.is_file():
        raise ValueError(f"Missing checkpoint index: {latest_file}")
    name = latest_file.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid checkpoint name: {name!r}")
    checkpoint = checkpoint_root / name
    required = (checkpoint / "agent.pkl", checkpoint / "step.pkl", checkpoint / "done")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    return checkpoint


def unexpected_nonfinite_metrics(rows: list[dict]) -> list[dict]:
    unexpected = []
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if not isinstance(value, float) or math.isfinite(value):
                continue
            conditional_empty = (
                key.startswith(("train/", "report/"))
                and ("/constats/" in key or "/rewstats/" in key)
                and key.rsplit("/", 1)[-1]
                in {"neg_acc", "neg_loss", "pos_acc", "pos_loss"}
            )
            if not (
                key in ALLOWED_NONFINITE_KEYS
                or conditional_empty
                or (key.startswith("timer/") and key.endswith("/min"))
            ):
                unexpected.append({"row": row_index, "key": key, "value": str(value)})
    return unexpected


def natural_stop_step(requested_step: int, driver_step_quantum: int) -> int:
    return (
        (requested_step + driver_step_quantum - 1) // driver_step_quantum
        * driver_step_quantum
    )


def verify(args: argparse.Namespace) -> dict:
    train_dir = args.run_dir / "train"
    generated_config = train_dir / "config.yaml"
    scores_path = train_dir / "scores.jsonl"
    metrics_path = train_dir / "metrics.jsonl"
    required = (
        args.frozen_config,
        generated_config,
        scores_path,
        metrics_path,
        args.stdout_log,
        args.resource_system,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required files: {missing}")
    frozen = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    generated = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
    metrics = load_jsonl(metrics_path)
    system = load_csv(args.resource_system)
    checkpoint = resolve_latest_checkpoint(train_dir)
    checkpoint_step = int(cloudpickle.loads((checkpoint / "step.pkl").read_bytes()))
    replay_files = [path for path in (train_dir / "replay").rglob("*") if path.is_file()]
    stdout = args.stdout_log.read_text(encoding="utf-8", errors="replace")
    unexpected = unexpected_nonfinite_metrics(metrics)
    loss_rows = [row for row in metrics if any(key.startswith("train/loss/") for key in row)]
    minecraft = generated.get("env", {}).get("minecraft", {})
    config_checks = {
        "script_train": generated.get("script") == "train",
        "task_is_diamond": generated.get("task") == "minecraft_diamond",
        "image_is_64_rgb": minecraft.get("size") == [64, 64],
        "break_speed_is_100": float(minecraft.get("break_speed", -1)) == 100.0,
        "environment_count": generated.get("run", {}).get("envs") == args.expected_envs,
        "run_debug_false": generated.get("run", {}).get("debug") is False,
        "size50m_deter": generated.get("agent", {}).get("dyn", {}).get("rssm", {}).get("deter") == 4096,
        "train_ratio_is_32": float(generated.get("run", {}).get("train_ratio", -1)) == 32.0,
    }
    first_system = system[0] if system else {}
    last_system = system[-1] if system else {}
    max_temp_dirs = max((int(row["temp_dir_count"]) for row in system), default=0)
    max_java_count = max((int(row["java_count"]) for row in system), default=0)
    system_disk_loss = (
        int(last_system.get("system_disk_used_bytes", 0))
        - int(first_system.get("system_disk_used_bytes", 0))
    )
    oom_delta = (
        int(last_system.get("memory_events_oom", 0))
        - int(first_system.get("memory_events_oom", 0))
    )
    oom_kill_delta = (
        int(last_system.get("memory_events_oom_kill", 0))
        - int(first_system.get("memory_events_oom_kill", 0))
    )
    remaining_temp_dirs = [path for path in args.temp_root.iterdir() if path.is_dir()]
    cleanup_path = args.run_dir / "temp_cleanup_postprocess.json"
    cleanup = (
        json.loads(cleanup_path.read_text(encoding="utf-8"))
        if cleanup_path.is_file()
        else {}
    )
    malmo_logs = list((args.run_dir / "work/malmo/logs").glob("mc_*.log"))
    natural_step = natural_stop_step(args.expected_step, args.driver_step_quantum)
    checks = {
        "frozen_config_sha256": sha256(args.frozen_config),
        "generated_config_sha256": sha256(generated_config),
        "config_semantically_equal": frozen == generated,
        "config_checks": config_checks,
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "natural_checkpoint_step": natural_step,
        "checkpoint_step_matches": checkpoint_step == natural_step,
        "checkpoint_sha256": sha256(checkpoint / "agent.pkl"),
        "replay_file_count": len(replay_files),
        "replay_bytes": sum(path.stat().st_size for path in replay_files),
        "replay_nonempty": bool(replay_files),
        "metric_rows": len(metrics),
        "has_post_warmup_loss": bool(loss_rows),
        "metrics_unexpected_nonfinite": unexpected,
        "metrics_no_unexpected_nonfinite": bool(metrics) and not unexpected,
        "resource_samples": len(system),
        "max_temp_dir_count": max_temp_dirs,
        "temp_instances_observed": max_temp_dirs >= args.expected_envs,
        "remaining_temp_dirs": [str(path) for path in remaining_temp_dirs],
        "temp_cleanup_postprocess": cleanup,
        "temp_instances_cleaned": bool(cleanup.get("passed")) and not remaining_temp_dirs,
        "max_java_count": max_java_count,
        "java_instances_observed": max_java_count >= args.expected_envs,
        "malmo_log_count": len(malmo_logs),
        "malmo_logs_observed": len(malmo_logs) >= args.expected_envs,
        "system_disk_loss_bytes": system_disk_loss,
        "system_disk_loss_within_limit": system_disk_loss <= args.max_system_disk_loss,
        "cgroup_oom_delta": oom_delta,
        "cgroup_oom_kill_delta": oom_kill_delta,
        "no_cgroup_oom": oom_delta == 0 and oom_kill_delta == 0,
        "error_markers": [marker for marker in ERROR_MARKERS if marker in stdout],
    }
    checks["passed"] = bool(
        checks["config_semantically_equal"]
        and all(config_checks.values())
        and checks["checkpoint_step_matches"]
        and checks["replay_nonempty"]
        and checks["has_post_warmup_loss"]
        and checks["metrics_no_unexpected_nonfinite"]
        and checks["resource_samples"] >= 2
        and checks["temp_instances_observed"]
        and checks["temp_instances_cleaned"]
        and checks["java_instances_observed"]
        and checks["malmo_logs_observed"]
        and checks["system_disk_loss_within_limit"]
        and checks["no_cgroup_oom"]
        and not checks["error_markers"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-envs", type=int, required=True)
    parser.add_argument("--driver-step-quantum", type=int, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--resource-system", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-system-disk-loss", type=int, default=536870912)
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
