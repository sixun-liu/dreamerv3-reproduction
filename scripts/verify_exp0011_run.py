#!/usr/bin/env python3
"""Verify one frozen EXP-0011 training stage without judging performance."""

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


def all_numbers_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def verify(
    run_dir: Path,
    frozen_config: Path,
    expected_step: int,
    stdout_path: Path,
    require_scores: bool,
) -> dict:
    train_dir = run_dir / "train"
    generated_config = train_dir / "config.yaml"
    checkpoint = train_dir / "checkpoint.ckpt"
    scores_path = train_dir / "scores.jsonl"
    metrics_path = train_dir / "metrics.jsonl"
    required = (
        frozen_config,
        generated_config,
        checkpoint,
        scores_path,
        metrics_path,
        stdout_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    frozen = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    generated = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
    scores = load_jsonl(scores_path)
    metrics = load_jsonl(metrics_path)
    checkpoint_data = cloudpickle.loads(checkpoint.read_bytes())
    checkpoint_step = int(checkpoint_data["step"])
    replay_files = [path for path in (train_dir / "replay").rglob("*") if path.is_file()]
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    unexpected = unexpected_nonfinite_metrics(metrics)
    loss_rows = [row for row in metrics if any(key.startswith("train/loss/") for key in row)]

    atari = generated.get("env", {}).get("atari100k", {})
    config_checks = {
        "task_is_breakout": generated.get("task") == "atari100k_breakout",
        "image_is_64_rgb": atari.get("size") == [64, 64] and atari.get("gray") is False,
        "action_repeat_is_four": atari.get("repeat") == 4,
        "sticky_is_false": atari.get("sticky") is False,
        "minimal_actions": atari.get("actions") == "needed",
        "random_noops_0_to_30": atari.get("noops") == 30,
        "raw_reward": atari.get("clip_reward") is False,
        "single_environment": generated.get("run", {}).get("envs") == 1,
        "size50m_deter": generated.get("agent", {}).get("dyn", {}).get("rssm", {}).get("deter") == 4096,
        "train_ratio_is_256": float(generated.get("run", {}).get("train_ratio", -1)) == 256.0,
    }
    score_rows_finite = all_numbers_finite(scores)
    checks = {
        "frozen_config_sha256": sha256(frozen_config),
        "generated_config_sha256": sha256(generated_config),
        "config_semantically_equal": frozen == generated,
        "config_checks": config_checks,
        "checkpoint_step": checkpoint_step,
        "checkpoint_step_matches": checkpoint_step == expected_step,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "replay_file_count": len(replay_files),
        "replay_bytes": sum(path.stat().st_size for path in replay_files),
        "replay_nonempty": bool(replay_files),
        "score_rows": len(scores),
        "score_rows_required": require_scores,
        "scores_finite": score_rows_finite and (bool(scores) or not require_scores),
        "metric_rows": len(metrics),
        "post_warmup_loss_rows": len(loss_rows),
        "has_post_warmup_loss": bool(loss_rows),
        "metrics_unexpected_nonfinite": unexpected,
        "metrics_no_unexpected_nonfinite": bool(metrics) and not unexpected,
        "error_markers": [marker for marker in ERROR_MARKERS if marker in stdout],
    }
    checks["passed"] = bool(
        checks["config_semantically_equal"]
        and all(config_checks.values())
        and checks["checkpoint_step_matches"]
        and checks["replay_nonempty"]
        and checks["scores_finite"]
        and checks["has_post_warmup_loss"]
        and checks["metrics_no_unexpected_nonfinite"]
        and not checks["error_markers"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--require-scores", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.run_dir,
        args.frozen_config,
        args.expected_step,
        args.stdout_log,
        args.require_scores,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
