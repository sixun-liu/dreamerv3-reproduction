#!/usr/bin/env python3
"""Analyze EXP-0010 against the frozen DMC Vision reference."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_ID = "EXP-0010"
TASK = "dmc_walker_walk"
REFERENCE_SHA256 = "68dd7e83ee6af076734e1edb843a3816cd241736c92e91837e7cf460231b7b08"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot describe an empty sequence")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "all_finite": all(math.isfinite(value) for value in values),
    }


def fixed_bins(rows: list[dict], max_step: int, width: int = 10_000) -> list[dict]:
    points = np.asarray(
        [(float(row["step"]), float(row["episode/score"])) for row in rows],
        dtype=float,
    )
    output = []
    for end in range(width, max_step + 1, width):
        values = points[(points[:, 0] > end - width) & (points[:, 0] <= end), 1]
        if len(values):
            output.append(
                {
                    "step": end,
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return output


def window_scores(rows: list[dict], low: int, high: int) -> list[float]:
    return [
        float(row["episode/score"])
        for row in rows
        if low < float(row["step"]) <= high
    ]


def load_reference(path: Path) -> list[dict]:
    if sha256(path) != REFERENCE_SHA256:
        raise ValueError("Official DMC Vision reference hash drift")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for row in json.load(handle) if row["task"] == TASK]
    rows.sort(key=lambda row: int(row["seed"]))
    if len(rows) != 10 or [int(row["seed"]) for row in rows] != list(range(10)):
        raise ValueError("Expected official walker vision seeds 0 through 9")
    for row in rows:
        if not np.isfinite(np.asarray(row["ys"], dtype=float)).all():
            raise ValueError("Official reference contains non-finite values")
    return rows


def reference_window(rows: list[dict], low: int, high: int) -> dict:
    per_seed = []
    for row in rows:
        xs = np.asarray(row["xs"], dtype=float)
        ys = np.asarray(row["ys"], dtype=float)
        selected = ys[(xs > low) & (xs <= high)]
        if not len(selected):
            raise ValueError(f"Official seed {row['seed']} lacks window {low}--{high}")
        per_seed.append(float(np.mean(selected)))
    return {
        "window": [low, high],
        "per_seed_means": per_seed,
        "mean": statistics.fmean(per_seed),
        "std_population": statistics.pstdev(per_seed),
        "range": [min(per_seed), max(per_seed)],
    }


def normalized_reference_auc(row: dict, high: int) -> float:
    xs = np.asarray(row["xs"], dtype=float)
    ys = np.asarray(row["ys"], dtype=float)
    inner = (xs > 0) & (xs < high)
    auc_x = np.concatenate(([0.0], xs[inner], [float(high)]))
    auc_y = np.interp(auc_x, xs, ys)
    return float(np.trapz(auc_y, auc_x) / high)


def parse_gpu_samples(path: Path) -> dict:
    if not path.is_file():
        return {"sample_count": 0}
    memory, utilization, power, temperature = [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                memory.append(float(row["memory_used_mib"]))
                utilization.append(float(row["gpu_util_percent"]))
                power.append(float(row["power_watts"]))
                temperature.append(float(row["temperature_c"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "sample_count": len(memory),
        "memory_used_mib_mean": statistics.fmean(memory) if memory else None,
        "memory_used_mib_max": max(memory) if memory else None,
        "gpu_util_percent_mean": statistics.fmean(utilization) if utilization else None,
        "gpu_util_percent_max": max(utilization) if utilization else None,
        "power_watts_mean": statistics.fmean(power) if power else None,
        "temperature_c_max": max(temperature) if temperature else None,
    }


def compute_continuation_gate(
    early_mean: float,
    tail_mean: float,
    integrity_passed: bool,
    wall_seconds: int,
    output_bytes: int,
    free_bytes: int,
) -> dict:
    estimated_additional_hours = wall_seconds * 9 / 3600
    projected_additional_bytes = output_bytes * 9
    projected_remaining_bytes = free_bytes - projected_additional_bytes
    gates = {
        "integrity_passed": integrity_passed,
        "tail_mean_at_least_200": tail_mean >= 200,
        "improvement_at_least_100": tail_mean - early_mean >= 100,
        "estimated_additional_wall_hours": estimated_additional_hours,
        "eta_at_most_12_hours": estimated_additional_hours <= 12,
        "pilot_output_bytes": output_bytes,
        "free_bytes_after_pilot": free_bytes,
        "projected_additional_bytes_linear": projected_additional_bytes,
        "projected_remaining_bytes_linear": projected_remaining_bytes,
        "projected_disk_floor_at_least_5_gib": projected_remaining_bytes >= 5 * 1024**3,
    }
    gates["continuation_gate"] = bool(
        gates["integrity_passed"]
        and gates["tail_mean_at_least_200"]
        and gates["improvement_at_least_100"]
        and gates["eta_at_most_12_hours"]
        and gates["projected_disk_floor_at_least_5_gib"]
    )
    return gates


def analyze(args: argparse.Namespace) -> dict:
    max_step = 100_000 if args.stage == "pilot" else 1_000_000
    stage_name = "pilot" if args.stage == "pilot" else "full"
    integrity = json.loads(
        (args.run_dir / f"integrity_{stage_name}.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (args.run_dir / f".{stage_name}.completed").read_text(encoding="utf-8")
    )
    scores = load_jsonl(args.run_dir / "train" / "scores.jsonl")
    if not scores or not all(math.isfinite(float(row["episode/score"])) for row in scores):
        raise ValueError("Local scores are empty or non-finite")
    reference = load_reference(args.reference)
    bins = fixed_bins(scores, max_step)
    if not bins:
        raise ValueError("No fixed-bin local curve could be computed")

    early = describe(window_scores(scores, 0, 20_000))
    tail_low = 80_000 if args.stage == "pilot" else 900_000
    tail = describe(window_scores(scores, tail_low, max_step))
    official_tail = reference_window(reference, tail_low, max_step)
    official_auc_values = [normalized_reference_auc(row, max_step) for row in reference]
    resource = parse_gpu_samples(args.run_dir / f"resource_{stage_name}_gpu.csv")

    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "protocol": {
            "runtime_commit": "6642b941f578cd72147bc2be3c3343d5bc72931c",
            "task": TASK,
            "model": "size12m",
            "seed": 0,
            "environment_seed_controlled": False,
            "action_repeat": 2,
            "train_ratio": 512,
            "environment_steps": max_step,
        },
        "integrity": integrity,
        "local": {
            "episode_count": len(scores),
            "early_0_20k": early,
            "tail": tail,
            "tail_minus_early_mean": tail["mean"] - early["mean"],
            "fixed_bin_count": len(bins),
        },
        "official": {
            "seed_count": len(reference),
            "tail": official_tail,
            "normalized_auc_per_seed": official_auc_values,
            "normalized_auc_range": [min(official_auc_values), max(official_auc_values)],
            "normalized_auc_mean": statistics.fmean(official_auc_values),
        },
        "resource": {
            "wall_seconds": int(completion["wall_seconds"]),
            "output_bytes": int(completion["output_bytes"]),
            "gpu_samples": resource,
        },
        "limitations": [
            "Single agent seed; DMC environment RNG is not controlled by this runtime.",
            "Local complete training episodes and the public smoothed curve are not sample-identical.",
            "The runtime is a 2024 author reimplementation lineage, not the exact 2023 training artifact.",
        ],
    }

    if args.stage == "pilot":
        disk = shutil.disk_usage(args.run_dir)
        gate = compute_continuation_gate(
            early["mean"],
            tail["mean"],
            bool(integrity["passed"]),
            int(completion["wall_seconds"]),
            int(completion["output_bytes"]),
            disk.free,
        )
        summary["gate"] = gate
        (args.run_dir / "gate_100k.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )
    else:
        official_range = official_tail["range"]
        summary["final_envelope"] = {
            "local_tail_mean": tail["mean"],
            "official_seed_tail_range": official_range,
            "inside_official_seed_tail_range": official_range[0]
            <= tail["mean"]
            <= official_range[1],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"summary_{args.stage}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"curve_{args.stage}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)

    grid = np.arange(10_000, max_step + 1, 10_000, dtype=float)
    official_interp = np.asarray(
        [
            np.interp(grid, np.asarray(row["xs"], dtype=float), np.asarray(row["ys"], dtype=float))
            for row in reference
        ]
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.fill_between(
        grid,
        np.min(official_interp, axis=0),
        np.max(official_interp, axis=0),
        color="#d0d0d0",
        alpha=0.55,
        label="Official 10-seed range",
    )
    ax.plot(grid, np.mean(official_interp, axis=0), color="#202020", linewidth=2, label="Official mean")
    ax.plot(
        [row["step"] for row in bins],
        [row["mean"] for row in bins],
        color="#c23b22",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Local seed 0, 10K episode bins",
    )
    ax.axvspan(tail_low, max_step, color="#e8b04f", alpha=0.12, label="Tail window")
    ax.set(
        title=f"EXP-0010 DMC Vision Walker ({args.stage})",
        xlabel="Environment steps",
        ylabel="Training episode score",
        xlim=(0, max_step),
        ylim=(0, 1000),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.review_dir / f"dmcvision_walker_{args.stage}.png", dpi=180)
    plt.close(fig)

    gate_line = ""
    if args.stage == "pilot":
        gate_line = f"- 100K continuation gate: `{summary['gate']['continuation_gate']}`\n"
    else:
        gate_line = (
            "- 末 100K 是否进入官方逐 seed 末段范围："
            f"`{summary['final_envelope']['inside_official_seed_tail_range']}`\n"
        )
    result = (
        f"# EXP-0010 {args.stage} 结果\n\n"
        f"- 完整性：`{integrity['passed']}`\n"
        f"- 本地末窗口均值：`{tail['mean']:.2f}`\n"
        f"- 官方 10-seed 末窗口范围：`{official_tail['range'][0]:.2f}` -- "
        f"`{official_tail['range'][1]:.2f}`\n"
        f"{gate_line}\n"
        "该比较是作者重实现的单 seed 描述性包络检查；environment RNG 与官方导出口径限制不变。\n"
    )
    (args.output_dir / f"RESULT_{args.stage}.md").write_text(result, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
