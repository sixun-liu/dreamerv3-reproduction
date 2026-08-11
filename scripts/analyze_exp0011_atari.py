#!/usr/bin/env python3
"""Analyze EXP-0011 smoke resources or formal Atari100K performance."""

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


EXPERIMENT_ID = "EXP-0011"
REFERENCE_TASK = "atari_breakout"
REFERENCE_SHA256 = "c839ef2fd28a5c4b7c6f36151a8d16cb41d4d2fb72338c265850b2b4dd5b5e15"


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
        raise ValueError("Official Atari100K reference hash drift")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for row in json.load(handle) if row["task"] == REFERENCE_TASK]
    rows.sort(key=lambda row: int(row["seed"]))
    if len(rows) != 5 or [int(row["seed"]) for row in rows] != list(range(5)):
        raise ValueError("Expected official Breakout seeds 0 through 4")
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


def compute_smoke_gate(
    integrity_passed: bool,
    policy_fps: list[float],
    smoke_steps: int,
    formal_steps: int,
    output_bytes: int,
    free_bytes: int,
) -> dict:
    positive = [value for value in policy_fps if math.isfinite(value) and value > 0]
    steady = statistics.median(positive[len(positive) // 2 :]) if positive else 0.0
    estimated_hours = (formal_steps - smoke_steps) / steady / 3600 if steady else math.inf
    projected_bytes = output_bytes * formal_steps / smoke_steps
    projected_remaining = free_bytes - projected_bytes
    gate = {
        "integrity_passed": integrity_passed,
        "policy_fps_samples": len(positive),
        "steady_policy_fps_median": steady,
        "estimated_remaining_wall_hours": estimated_hours,
        "eta_at_most_12_hours": estimated_hours <= 12,
        "smoke_output_bytes": output_bytes,
        "projected_formal_bytes_linear": projected_bytes,
        "projected_remaining_bytes_linear": projected_remaining,
        "projected_disk_floor_at_least_5_gib": projected_remaining >= 5 * 1024**3,
    }
    gate["formal_gate"] = bool(
        gate["integrity_passed"]
        and positive
        and gate["eta_at_most_12_hours"]
        and gate["projected_disk_floor_at_least_5_gib"]
    )
    return gate


def analyze_smoke(args: argparse.Namespace) -> dict:
    integrity = json.loads((args.run_dir / "integrity_smoke.json").read_text())
    completion = json.loads((args.run_dir / ".smoke.completed").read_text())
    metrics = load_jsonl(args.run_dir / "train/metrics.jsonl")
    fps = [float(row["fps/policy"]) for row in metrics if "fps/policy" in row]
    disk = shutil.disk_usage(args.run_dir)
    gate = compute_smoke_gate(
        bool(integrity["passed"]),
        fps,
        int(completion["checkpoint_step"]),
        100_000,
        int(completion["output_bytes"]),
        disk.free,
    )
    (args.run_dir / "gate_smoke.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "smoke",
        "integrity": integrity,
        "gate": gate,
        "resource": {
            "wall_seconds": int(completion["wall_seconds"]),
            "output_bytes": int(completion["output_bytes"]),
            "gpu_samples": parse_gpu_samples(args.run_dir / "resource_smoke_gpu.csv"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_smoke.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def analyze_formal(args: argparse.Namespace) -> dict:
    max_step = 400_000
    integrity = json.loads((args.run_dir / "integrity_formal.json").read_text())
    completion = json.loads((args.run_dir / ".formal.completed").read_text())
    scores = load_jsonl(args.run_dir / "train/scores.jsonl")
    if not scores or not all(math.isfinite(float(row["episode/score"])) for row in scores):
        raise ValueError("Local Atari scores are empty or non-finite")
    bins = fixed_bins(scores, max_step)
    early = describe(window_scores(scores, 0, 40_000))
    tail = describe(window_scores(scores, 360_000, 400_000))
    reference = load_reference(args.reference)
    official_tail = reference_window(reference, 360_000, 400_000)
    learning_gate = {
        "tail_mean_at_least_3": tail["mean"] >= 3.0,
        "tail_minus_early_at_least_1": tail["mean"] - early["mean"] >= 1.0,
    }
    learning_gate["passed"] = all(learning_gate.values()) and bool(integrity["passed"])
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "formal",
        "protocol": {
            "runtime_commit": "5168475b7a4413f9575933b4580e7073caea2114",
            "task": "atari100k_breakout",
            "model": "size50m",
            "agent_seed": 0,
            "training_environment_seed_controlled": False,
            "action_repeat": 4,
            "train_ratio": 256,
            "agent_decisions": 100_000,
            "emulator_frames": 400_000,
        },
        "integrity": integrity,
        "local": {
            "episode_count": len(scores),
            "early_0_40k_frames": early,
            "tail_360_400k_frames": tail,
            "tail_minus_early_mean": tail["mean"] - early["mean"],
            "fixed_bin_count": len(bins),
            "learning_gate": learning_gate,
        },
        "official_context": {"seed_count": 5, "tail": official_tail},
        "resource": {
            "wall_seconds": int(completion["wall_seconds"]),
            "output_bytes": int(completion["output_bytes"]),
            "gpu_samples": parse_gpu_samples(args.run_dir / "resource_formal_gpu.csv"),
        },
        "limitations": [
            "Local size50m/ratio256 differs from the paper size200m/ratio128 protocol.",
            "Single agent seed; the training ALE/no-op RNG is not explicitly controlled.",
            "Independent evaluation is separate from the public training-curve export.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_formal.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "curve_formal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)

    grid = np.arange(10_000, max_step + 1, 10_000, dtype=float)
    official_interp = np.asarray(
        [
            np.interp(grid, np.asarray(row["xs"], float), np.asarray(row["ys"], float))
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
        label="Official 5-seed range (200M/ratio128)",
    )
    ax.plot(grid, np.mean(official_interp, axis=0), color="#202020", linewidth=2, label="Official mean")
    ax.plot(
        [row["step"] for row in bins],
        [row["mean"] for row in bins],
        color="#c23b22",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Local seed 0 (50M/ratio256)",
    )
    ax.axvspan(360_000, 400_000, color="#e8b04f", alpha=0.12, label="Tail window")
    ax.axhline(3.0, color="#2b7a78", linestyle="--", linewidth=1, label="Frozen trend threshold")
    ax.set(
        title="EXP-0011 Atari100K Breakout",
        xlabel="Emulator frames (4 per agent decision)",
        ylabel="Training episode raw score",
        xlim=(0, max_step),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.review_dir / "atari100k_breakout_formal.png", dpi=180)
    plt.close(fig)

    (args.output_dir / "RESULT.md").write_text(
        "# EXP-0011 Atari100K Breakout 结果\n\n"
        f"- 完整性：`{integrity['passed']}`\n"
        f"- 末 40K emulator frames 训练均值：`{tail['mean']:.2f}`\n"
        f"- 相对首 40K 改善：`{tail['mean'] - early['mean']:.2f}`\n"
        f"- 预注册学习趋势门：`{learning_gate['passed']}`\n"
        f"- 官方 5-seed 末段范围（仅作上下文）：`{official_tail['range'][0]:.2f}` -- `{official_tail['range'][1]:.2f}`\n\n"
        "本地使用 50M/ratio256，不能作为论文 200M/ratio128 的严格数值复现。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "formal" and args.review_dir is None:
        parser.error("--review-dir is required for formal analysis")
    result = analyze_smoke(args) if args.stage == "smoke" else analyze_formal(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
