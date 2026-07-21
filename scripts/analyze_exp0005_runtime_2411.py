#!/usr/bin/env python3
"""Analyze the frozen 2411f7d walker_walk lineage replication."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ERROR_MARKERS = ("Traceback", "AssertionError", "Out of memory", "CUDA_ERROR")
REFERENCE_SHA256 = "8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4"
RUN_TAG = "EXP-0005__walker_walk__s000-agent__500k-env__20260721T120000Z"


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "all_finite": all(math.isfinite(value) for value in values),
    }


def load_reference(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for row in json.load(handle)
                if row["task"] == "dmc_walker_walk"]
    if len(rows) != 5:
        raise ValueError(f"Expected 5 official seeds, found {len(rows)}")
    xs = np.asarray(rows[0]["xs"], dtype=float)
    ys = np.asarray([row["ys"] for row in rows], dtype=float)
    return xs, ys


def aggregate_curve(rows, xs, width=10_000):
    points = np.asarray([
        (float(row["step"]), float(row["episode/score"])) for row in rows])
    output = []
    for x in xs:
        values = points[(points[:, 0] > x - width) & (points[:, 0] <= x), 1]
        if len(values):
            output.append({
                "step": int(x),
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return output


def parse_time(path, field):
    value = json.loads(path.read_text(encoding="utf-8"))[field]
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current-runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.run_dir.name != RUN_TAG:
        raise ValueError(f"Unexpected run tag: {args.run_dir.name}")
    if sha256(args.reference) != REFERENCE_SHA256:
        raise ValueError("Official reference hash drift")

    train_rows = load_jsonl(args.run_dir / "train" / "scores.jsonl")
    metrics_rows = load_jsonl(args.run_dir / "train" / "metrics.jsonl")
    scores = [float(row["episode/score"]) for row in train_rows]
    ref_x, ref_y = load_reference(args.reference)
    curve = aggregate_curve(train_rows, ref_x)
    curve_by_step = {row["step"]: row for row in curve}

    for row in curve:
        index = int(np.where(ref_x == row["step"])[0][0])
        official = ref_y[:, index]
        row.update({
            "official_mean": float(np.mean(official)),
            "official_median": float(np.median(official)),
            "official_min": float(np.min(official)),
            "official_max": float(np.max(official)),
            "median_inside_official_range": bool(
                np.min(official) <= row["median"] <= np.max(official)),
        })

    final_scores = [
        float(row["episode/score"]) for row in train_rows
        if 470_000 < float(row["step"]) <= 500_000]
    if not final_scores:
        raise ValueError("No complete episodes in the frozen final window")
    final_stats = describe(final_scores)
    official_seed_finals = [float(np.mean(seed[-3:])) for seed in ref_y]
    official_final_range = [min(official_seed_finals), max(official_seed_finals)]
    final_gate = official_final_range[0] <= final_stats["mean"] <= official_final_range[1]
    early_gate = bool(curve_by_step.get(250_000, {}).get(
        "median_inside_official_range", False))

    current_curves = []
    for seed in range(3):
        tag = ("EXP-0004__walker_walk__s"
               f"{seed:03d}__500k-env__20260721T061500Z")
        rows = load_jsonl(args.current_runs_root / tag / "train" / "scores.jsonl")
        current_curves.append(aggregate_curve(rows, ref_x))

    with args.frozen_config.open(encoding="utf-8") as handle:
        frozen_config = yaml.safe_load(handle)
    with (args.run_dir / "train" / "config.yaml").open(encoding="utf-8") as handle:
        generated_config = yaml.safe_load(handle)
    checkpoint = args.run_dir / "train" / "checkpoint.ckpt"
    with checkpoint.open("rb") as handle:
        checkpoint_data = pickle.load(handle)
    stdout = (args.run_dir / "train_stdout.log").read_text(
        encoding="utf-8", errors="replace")
    error_markers = [marker for marker in ERROR_MARKERS if marker in stdout]
    started = parse_time(Path(str(args.run_dir) + ".started"), "started_at")
    completed = parse_time(Path(str(args.run_dir) + ".completed"), "completed_at")
    replay_ratios = [float(row["replay/replay_ratio"]) for row in metrics_rows
                     if "replay/replay_ratio" in row
                     and math.isfinite(float(row["replay/replay_ratio"]))]

    integrity = {
        "completion_signal": Path(str(args.run_dir) + ".completed").exists(),
        "checkpoint_step": int(checkpoint_data["step"]),
        "checkpoint_sha256": sha256(checkpoint),
        "config_semantically_equal": frozen_config == generated_config,
        "all_scores_finite": all(math.isfinite(value) for value in scores),
        "error_markers": error_markers,
    }
    integrity_gate = bool(
        integrity["completion_signal"]
        and integrity["checkpoint_step"] == 250_000
        and integrity["config_semantically_equal"]
        and integrity["all_scores_finite"]
        and not integrity["error_markers"])

    summary = {
        "experiment_id": "EXP-0005",
        "run_tag": RUN_TAG,
        "runtime": {
            "upstream_commit": "2411f7d136832378c0291c587cdbf2fca6506873",
            "compatibility_commit": "6642b941f578cd72147bc2be3c3343d5bc72931c",
        },
        "wall_hours": (completed - started).total_seconds() / 3600,
        "episode_count": len(scores),
        "final_30k": final_stats,
        "official_final": {
            "seed_means": official_seed_finals,
            "mean": statistics.fmean(official_seed_finals),
            "range": official_final_range,
        },
        "checkpoints": {
            str(step): curve_by_step.get(step) for step in (50_000, 100_000, 250_000, 500_000)
        },
        "gates": {
            "integrity": integrity_gate,
            "early_250k_inside_official_seed_envelope": early_gate,
            "final_30k_mean_inside_official_seed_final_range": final_gate,
        },
        "integrity": integrity,
        "replay_ratio_last": replay_ratios[-1] if replay_ratios else None,
        "interpretation_guardrail": (
            "One run can support lineage plausibility but cannot establish stable replication; "
            "DMC environment randomness is not controlled by --seed in this runtime."),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(ref_x, np.min(ref_y, axis=0), np.max(ref_y, axis=0),
                    color="#c8c8c8", alpha=0.45, label="Official 5-seed range")
    ax.plot(ref_x, np.mean(ref_y, axis=0), color="#202020", linewidth=2,
            label="Official mean")
    for index, current in enumerate(current_curves):
        ax.plot([row["step"] for row in current],
                [row["median"] for row in current], color="#4575b4",
                alpha=0.25, linewidth=1,
                label="2026 runtime seeds" if index == 0 else None)
    ax.plot([row["step"] for row in curve], [row["median"] for row in curve],
            color="#b2182b", linewidth=2.2, marker="o", markersize=3,
            label="2411f7d lineage, one run")
    ax.set(xlabel="Environment steps", ylabel="Episode score",
           title="EXP-0005 walker_walk runtime-lineage comparison")
    ax.set_xlim(0, 500_000)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    figure = args.review_dir / "curve_comparison.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = "# EXP-0005 结果\n\n"
    result += f"- 完整性门：`{integrity_gate}`\n"
    result += f"- 250K 早期曲线门：`{early_gate}`\n"
    result += f"- final-30K 门：`{final_gate}`\n"
    result += f"- final-30K mean：`{final_stats['mean']:.2f}`\n"
    result += f"- 墙钟时间：`{summary['wall_hours']:.3f} h`\n\n"
    result += "单次运行只能判断旧代码谱系是否具有对齐迹象，不能证明稳定复现。\n"
    (args.output_dir / "RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
