#!/usr/bin/env python3
"""Verify EXP-0009 parameter-group gradient routing from three probe runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("baseline", "no_reward_value", "no_reconstruction")
PREFIX = "report/gradnorm"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metric(loss: str, group: str) -> str:
    return f"{PREFIX}/{loss}/{group}"


EXPECTED = {
    "baseline": {
        "positive": (
            metric("reconstruction", "enc"),
            metric("reconstruction", "dyn"),
            metric("reconstruction", "dec"),
            metric("reward", "enc"),
            metric("reward", "dyn"),
            metric("reward", "rew"),
            metric("replay_critic", "enc"),
            metric("replay_critic", "dyn"),
            metric("replay_critic", "critic"),
        ),
        "zero": (),
    },
    "no_reward_value": {
        "positive": (
            metric("reconstruction", "enc"),
            metric("reconstruction", "dyn"),
            metric("reconstruction", "dec"),
            metric("reward", "rew"),
            metric("replay_critic", "critic"),
        ),
        "zero": (
            metric("reward", "enc"),
            metric("reward", "dyn"),
            metric("replay_critic", "enc"),
            metric("replay_critic", "dyn"),
        ),
    },
    "no_reconstruction": {
        "positive": (
            metric("reconstruction", "dec"),
            metric("reward", "enc"),
            metric("reward", "dyn"),
            metric("reward", "rew"),
            metric("replay_critic", "enc"),
            metric("replay_critic", "dyn"),
            metric("replay_critic", "critic"),
        ),
        "zero": (
            metric("reconstruction", "enc"),
            metric("reconstruction", "dyn"),
        ),
    },
}


def find_complete_row(rows: list[dict], keys: tuple[str, ...]) -> dict:
    matches = [row for row in rows if all(key in row for key in keys)]
    if not matches:
        raise ValueError(f"No metrics row contains all gradient keys: {keys}")
    return matches[-1]


def verify(root: Path) -> dict:
    result = {"experiment_id": "EXP-0009", "arms": {}, "passed": True}
    for arm in ARMS:
        metrics_path = root / arm / "train" / "metrics.jsonl"
        if not metrics_path.is_file():
            raise ValueError(f"Missing metrics: {metrics_path}")
        expected = EXPECTED[arm]
        keys = (*expected["positive"], *expected["zero"])
        row = find_complete_row(load_rows(metrics_path), keys)
        observed = {key: float(row[key]) for key in keys}
        nonfinite = [key for key, value in observed.items() if not math.isfinite(value)]
        failed_positive = [key for key in expected["positive"] if not observed[key] > 0.0]
        failed_zero = [key for key in expected["zero"] if observed[key] != 0.0]
        passed = not nonfinite and not failed_positive and not failed_zero
        result["arms"][arm] = {
            "step_env": row.get("step"),
            "observed": observed,
            "nonfinite": nonfinite,
            "failed_positive": failed_positive,
            "failed_zero": failed_zero,
            "passed": passed,
        }
        result["passed"] = result["passed"] and passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
