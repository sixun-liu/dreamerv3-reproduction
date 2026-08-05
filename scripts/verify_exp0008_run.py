#!/usr/bin/env python3
"""Verify one EXP-0008 training run without interpreting its score."""

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
)
ALLOWED_NONFINITE_METRIC_KEYS = {
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
            if isinstance(value, float) and not math.isfinite(value):
                allowed = (
                    key in ALLOWED_NONFINITE_METRIC_KEYS
                    or key in {
                        "train/constats/neg_acc",
                        "train/constats/neg_loss",
                        "report/constats/neg_acc",
                        "report/constats/neg_loss",
                    }
                    or (key.startswith("timer/") and key.endswith("/min"))
                )
                if not allowed:
                    unexpected.append({"row": row_index, "key": key, "value": str(value)})
    return unexpected


def verify(run_dir: Path, frozen_config: Path, expected_step: int) -> dict:
    train_dir = run_dir / "train"
    generated_config = train_dir / "config.yaml"
    checkpoint = train_dir / "checkpoint.ckpt"
    stdout_path = run_dir / "train_stdout.log"
    required = (
        frozen_config,
        generated_config,
        checkpoint,
        train_dir / "scores.jsonl",
        train_dir / "metrics.jsonl",
        stdout_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required files: {missing}")

    frozen = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    generated = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
    scores = load_jsonl(train_dir / "scores.jsonl")
    metrics = load_jsonl(train_dir / "metrics.jsonl")
    if not scores or not metrics:
        raise ValueError("scores.jsonl and metrics.jsonl must both be non-empty")
    checkpoint_data = cloudpickle.loads(checkpoint.read_bytes())
    checkpoint_step = int(checkpoint_data["step"])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    error_markers = [marker for marker in ERROR_MARKERS if marker in stdout]

    unexpected_metric_values = unexpected_nonfinite_metrics(metrics)
    checks = {
        "config_semantically_equal": frozen == generated,
        "checkpoint_step": checkpoint_step,
        "checkpoint_step_matches": checkpoint_step == expected_step,
        "checkpoint_sha256": sha256(checkpoint),
        "score_rows": len(scores),
        "metric_rows": len(metrics),
        "scores_finite": all_numbers_finite(scores),
        "metrics_unexpected_nonfinite": unexpected_metric_values,
        "metrics_no_unexpected_nonfinite": not unexpected_metric_values,
        "error_markers": error_markers,
    }
    checks["passed"] = bool(
        checks["config_semantically_equal"]
        and checks["checkpoint_step_matches"]
        and checks["scores_finite"]
        and checks["metrics_no_unexpected_nonfinite"]
        and not checks["error_markers"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = verify(args.run_dir, args.frozen_config, args.expected_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
