#!/usr/bin/env python3
"""Verify a quickstart run without judging policy performance."""

from __future__ import annotations

import argparse
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


def all_numbers_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


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
            allowed = (
                key in ALLOWED_NONFINITE_KEYS
                or conditional_empty
                or (key.startswith("timer/") and key.endswith("/min"))
            )
            if not allowed:
                unexpected.append({"row": row_index, "key": key, "value": str(value)})
    return unexpected


def natural_stop_step(requested_step: int, quantum: int) -> int:
    if requested_step < 0 or quantum <= 0:
        raise ValueError("Steps must be non-negative and quantum must be positive")
    return (requested_step + quantum - 1) // quantum * quantum


def resolve_checkpoint(train_dir: Path) -> tuple[Path, int, list[Path]]:
    legacy = train_dir / "checkpoint.ckpt"
    if legacy.is_file():
        data = cloudpickle.loads(legacy.read_bytes())
        return legacy, int(data["step"]), [legacy]

    checkpoint_root = train_dir / "ckpt"
    latest_file = checkpoint_root / "latest"
    if not latest_file.is_file():
        raise ValueError(f"Missing checkpoint index: {latest_file}")
    name = latest_file.read_text(encoding="utf-8").strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Invalid checkpoint name: {name!r}")
    checkpoint = checkpoint_root / name
    required = [checkpoint / name for name in ("agent.pkl", "step.pkl", "done")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    step = int(cloudpickle.loads((checkpoint / "step.pkl").read_bytes()))
    return (
        checkpoint,
        step,
        sorted(path for path in checkpoint.iterdir() if path.is_file()),
    )


def config_checks(task: str, config: dict) -> dict[str, bool]:
    if task == "dmc-vision":
        dmc = config.get("env", {}).get("dmc", {})
        return {
            "task": config.get("task") == "dmc_walker_walk",
            "image_observation": dmc.get("image") is True,
            "action_repeat": dmc.get("repeat") == 2,
            "size12m": config.get("dyn", {}).get("rssm", {}).get("deter") == 2048,
            "train_ratio": float(config.get("run", {}).get("train_ratio", -1)) == 512.0,
        }
    if task == "breakout":
        atari = config.get("env", {}).get("atari100k", {})
        return {
            "task": config.get("task") == "atari100k_breakout",
            "image_64_rgb": atari.get("size") == [64, 64]
            and atari.get("gray") is False,
            "action_repeat": atari.get("repeat") == 4,
            "raw_reward": atari.get("clip_reward") is False,
            "size50m": config.get("agent", {})
            .get("dyn", {})
            .get("rssm", {})
            .get("deter")
            == 4096,
            "train_ratio": float(config.get("run", {}).get("train_ratio", -1)) == 256.0,
        }
    minecraft = config.get("env", {}).get("minecraft", {})
    return {
        "task": config.get("task") == "minecraft_diamond",
        "image_64": minecraft.get("size") == [64, 64],
        "episode_length": int(minecraft.get("length", -1)) == 36000,
        "size50m": config.get("agent", {}).get("dyn", {}).get("rssm", {}).get("deter")
        == 4096,
        "train_ratio": float(config.get("run", {}).get("train_ratio", -1)) == 32.0,
    }


def verify(args: argparse.Namespace) -> dict:
    train_dir = args.run_dir / "train"
    config_path = train_dir / "config.yaml"
    metrics_path = train_dir / "metrics.jsonl"
    scores_path = train_dir / "scores.jsonl"
    required = (config_path, metrics_path, args.stdout_log)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = load_jsonl(metrics_path)
    scores = load_jsonl(scores_path) if scores_path.is_file() else []
    checkpoint, checkpoint_step, checkpoint_files = resolve_checkpoint(train_dir)
    replay_files = [
        path for path in (train_dir / "replay").rglob("*") if path.is_file()
    ]
    stdout = args.stdout_log.read_text(encoding="utf-8", errors="replace")
    unexpected = unexpected_nonfinite_metrics(metrics)
    losses = [
        row
        for row in metrics
        if any(
            key.startswith("train/loss/")
            or (key.startswith("train/") and key.endswith("_loss"))
            for key in row
        )
    ]
    expected_checkpoint_step = natural_stop_step(
        args.expected_step, args.driver_step_quantum
    )
    checks = {
        "schema_version": 1,
        "task": args.task,
        "run_dir": str(args.run_dir.resolve()),
        "config_sha256": sha256(config_path),
        "config_checks": config_checks(args.task, config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "expected_checkpoint_step": expected_checkpoint_step,
        "checkpoint_step_matches": checkpoint_step == expected_checkpoint_step,
        "checkpoint_files": {path.name: sha256(path) for path in checkpoint_files},
        "replay_file_count": len(replay_files),
        "replay_bytes": sum(path.stat().st_size for path in replay_files),
        "replay_nonempty": bool(replay_files),
        "metric_rows": len(metrics),
        "loss_rows": len(losses),
        "metrics_no_unexpected_nonfinite": bool(metrics) and not unexpected,
        "metrics_unexpected_nonfinite": unexpected,
        "score_rows": len(scores),
        "scores_finite": all_numbers_finite(scores),
        "error_markers": [marker for marker in ERROR_MARKERS if marker in stdout],
    }
    checks["passed"] = bool(
        all(checks["config_checks"].values())
        and checks["checkpoint_step_matches"]
        and checks["replay_nonempty"]
        and checks["loss_rows"]
        and checks["metrics_no_unexpected_nonfinite"]
        and checks["scores_finite"]
        and not checks["error_markers"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("dmc-vision", "breakout", "minecraft"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--driver-step-quantum", type=int, default=1)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
