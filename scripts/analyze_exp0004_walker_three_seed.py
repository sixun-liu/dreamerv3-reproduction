#!/usr/bin/env python3
"""Analyze the frozen three-seed DreamerV3 walker_walk replication."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ERROR_MARKERS = ("Traceback", "AssertionError", "Out of memory")


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
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
    if any(not np.array_equal(xs, np.asarray(row["xs"], dtype=float))
           for row in rows):
        raise ValueError("Official seeds do not share an x-axis")
    return xs, ys


def aggregate_curve(rows, xs, width=10_000):
    points = np.asarray([
        (float(row["step"]), float(row["episode/score"])) for row in rows])
    output = []
    for x in xs:
        values = points[(points[:, 0] > x - width) &
                        (points[:, 0] <= x), 1]
        if len(values):
            output.append({
                "step": float(x),
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return output


def parse_time(path, field):
    data = json.loads(path.read_text(encoding="utf-8"))
    return datetime.fromisoformat(data[field].replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()

    reference_sha256 = sha256(args.reference)
    expected_reference = (
        "8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4")
    if reference_sha256 != expected_reference:
        raise ValueError(f"Reference hash drift: {reference_sha256}")

    ref_x, ref_y = load_reference(args.reference)
    official_seed_final = [float(np.mean(seed[-3:])) for seed in ref_y]
    official_mean = statistics.fmean(official_seed_final)
    official_std = statistics.pstdev(official_seed_final)
    official_seed_min = min(official_seed_final)
    official_seed_max = max(official_seed_final)
    acceptance_lower = official_mean - 2 * official_std
    acceptance_upper = official_mean + 2 * official_std
    analysis_x = np.unique(np.append(ref_x, 500_000.0))

    run_summaries = []
    curve_rows = []
    eval_values_by_seed = []
    colors = ["#b34234", "#2e7145", "#6b4c9a"]
    run_tag = "EXP-0004__walker_walk__s{seed:03d}__500k-env__20260721T061500Z"

    for seed in range(3):
        root = args.runs_root / run_tag.format(seed=seed)
        train_rows = load_jsonl(root / "train" / "scores.jsonl")
        eval_rows = load_jsonl(root / "eval" / "scores.jsonl")
        train_values = [float(row["episode/score"]) for row in train_rows]
        eval_values = [float(row["episode/score"]) for row in eval_rows]
        eval_values_by_seed.append(eval_values)

        final_values = [
            float(row["episode/score"]) for row in train_rows
            if 470_000 < float(row["step"]) <= 500_000]
        if not final_values:
            raise ValueError(f"Seed {seed} has no final-window episodes")
        final_stats = stats(final_values)

        bins = aggregate_curve(train_rows, analysis_x)
        first_entry = None
        bin_250k = None
        for row in bins:
            ref_index = int(np.argmin(np.abs(ref_x - row["step"])))
            ref_values = ref_y[:, ref_index]
            row.update({
                "seed": seed,
                "reference_step": float(ref_x[ref_index]),
                "reference_median": float(np.median(ref_values)),
                "reference_min": float(np.min(ref_values)),
                "reference_max": float(np.max(ref_values)),
                "mean_inside_reference_range": bool(
                    np.min(ref_values) <= row["mean"] <= np.max(ref_values)),
                "median_inside_reference_range": bool(
                    np.min(ref_values) <= row["median"] <= np.max(ref_values)),
            })
            if first_entry is None and row["median_inside_reference_range"]:
                first_entry = int(row["step"])
            if row["step"] == 250_000:
                bin_250k = dict(row)
            curve_rows.append(row)

        frozen_path = (args.configs_dir /
                       f"exp0004_walker_s{seed:03d}_500k_env.yaml")
        with frozen_path.open(encoding="utf-8") as handle:
            frozen_config = yaml.safe_load(handle)
        with (root / "train" / "config.yaml").open(encoding="utf-8") as handle:
            generated_config = yaml.safe_load(handle)
        eval_metrics = load_jsonl(root / "eval" / "metrics.jsonl")
        eval_train_keys = sorted({
            key for row in eval_metrics for key in row
            if key.startswith("train/")})
        checkpoint = Path((root / "checkpoint_path.txt").read_text().strip())
        error_markers = []
        for logfile in (root / "train_stdout.log", root / "eval_stdout.log"):
            text = logfile.read_text(encoding="utf-8", errors="replace")
            error_markers.extend(
                f"{logfile.name}:{marker}" for marker in ERROR_MARKERS
                if marker in text)
        started = parse_time(root / ".started", "started_at")
        completed = parse_time(root / ".completed", "completed_at")

        run_summaries.append({
            "seed": seed,
            "run_dir": str(root),
            "wall_hours": (completed - started).total_seconds() / 3600,
            "integrity": {
                "root_completed": (root / ".completed").exists(),
                "train_completed": (root / "train.completed").exists(),
                "eval_completed": (root / "eval.completed").exists(),
                "checkpoint_step": int(
                    (root / "checkpoint_step.txt").read_text().strip()),
                "checkpoint_agent_sha256": sha256(checkpoint / "agent.pkl"),
                "checkpoint_step_sha256": sha256(checkpoint / "step.pkl"),
                "train_config_semantically_equal": frozen_config == generated_config,
                "eval_train_metric_keys": eval_train_keys,
                "error_markers": error_markers,
            },
            "train": {
                "episode_count": len(train_rows),
                "last_logged_environment_step": int(max(
                    float(row["step"]) for row in train_rows)),
                "all_scores_finite": all(math.isfinite(value)
                                          for value in train_values),
                "final_30k": final_stats,
                "final_mean_inside_official_seed_range": bool(
                    official_seed_min <= final_stats["mean"] <= official_seed_max),
                "bin_250k": bin_250k,
                "first_median_entry_into_official_curve_range": first_entry,
            },
            "checkpoint_eval": {
                **stats(eval_values),
                "eval_seed": 10000,
                "policy_semantics": (
                    "mode=eval with stochastic continuous-action sampling"),
            },
        })

    local_final_means = [
        row["train"]["final_30k"]["mean"] for row in run_summaries]
    inside_count = sum(
        row["train"]["final_mean_inside_official_seed_range"]
        for row in run_summaries)
    aggregate_mean = statistics.fmean(local_final_means)
    aggregate = {
        "seed_count": len(run_summaries),
        "per_seed_final_30k_means": local_final_means,
        "mean": aggregate_mean,
        "std_population": statistics.pstdev(local_final_means),
        "min": min(local_final_means),
        "max": max(local_final_means),
        "absolute_difference_from_official_mean": aggregate_mean - official_mean,
        "relative_difference_from_official_mean_percent": (
            100 * (aggregate_mean - official_mean) / official_mean),
        "seeds_inside_official_seed_range": inside_count,
    }
    aggregate_gate_passed = bool(
        acceptance_lower <= aggregate_mean <= acceptance_upper)
    individual_gate_passed = inside_count >= 2
    integrity_gate_passed = all(
        row["integrity"]["root_completed"]
        and row["integrity"]["train_completed"]
        and row["integrity"]["eval_completed"]
        and row["integrity"]["checkpoint_step"] == 250_000
        and row["integrity"]["train_config_semantically_equal"]
        and not row["integrity"]["eval_train_metric_keys"]
        and not row["integrity"]["error_markers"]
        and row["train"]["all_scores_finite"]
        and row["checkpoint_eval"]["count"] >= 64
        and row["checkpoint_eval"]["all_finite"]
        for row in run_summaries)

    summary = {
        "experiment_id": "EXP-0004",
        "reproduction_kind": "author_reimplementation",
        "reference": {
            "path": str(args.reference),
            "sha256": reference_sha256,
            "official_seed_final_three_point_means": official_seed_final,
            "mean": official_mean,
            "std_population": official_std,
            "seed_range": [official_seed_min, official_seed_max],
        },
        "preregistered_acceptance": {
            "aggregate_mean_range": [acceptance_lower, acceptance_upper],
            "minimum_individual_seeds_inside_official_range": 2,
            "aggregate_gate_passed": aggregate_gate_passed,
            "individual_gate_passed": individual_gate_passed,
            "integrity_gate_passed": integrity_gate_passed,
            "all_gates_passed": bool(
                aggregate_gate_passed and individual_gate_passed
                and integrity_gate_passed),
        },
        "local_final_30k_aggregate": aggregate,
        "runs": run_summaries,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    curve_fields = list(curve_rows[0].keys())
    with (args.output_dir / "train_curve_bins.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    per_seed_rows = []
    for row in run_summaries:
        per_seed_rows.append({
            "seed": row["seed"],
            "wall_hours": row["wall_hours"],
            "train_episode_count": row["train"]["episode_count"],
            "final_30k_count": row["train"]["final_30k"]["count"],
            "final_30k_mean": row["train"]["final_30k"]["mean"],
            "final_30k_median": row["train"]["final_30k"]["median"],
            "final_30k_std_population": row["train"]["final_30k"]["std_population"],
            "first_curve_entry_step": row["train"][
                "first_median_entry_into_official_curve_range"],
            "eval_count": row["checkpoint_eval"]["count"],
            "eval_mean": row["checkpoint_eval"]["mean"],
            "eval_median": row["checkpoint_eval"]["median"],
            "eval_std_population": row["checkpoint_eval"]["std_population"],
        })
    with (args.output_dir / "per_seed_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_seed_rows)

    plot_ref_x = np.append(ref_x, 500_000.0)
    plot_ref_y = np.column_stack([ref_y, ref_y[:, -1]])
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.fill_between(plot_ref_x, plot_ref_y.min(0), plot_ref_y.max(0),
                    color="#c9d4df", alpha=0.7,
                    label="Official seeds 0-4 range")
    ax.plot(plot_ref_x, np.median(plot_ref_y, axis=0), color="#263b50",
            linewidth=2, label="Official median")
    aggregate_by_step = {}
    for seed, color in zip(range(3), colors):
        rows = [row for row in curve_rows if row["seed"] == seed]
        xs = np.asarray([row["step"] for row in rows])
        means = np.asarray([row["mean"] for row in rows])
        ax.plot(xs, means, color=color, linewidth=1.5, marker="o",
                markersize=3, alpha=0.9, label=f"Local seed {seed} mean")
        for row in rows:
            aggregate_by_step.setdefault(row["step"], []).append(row["mean"])
    aggregate_x = sorted(
        step for step, values in aggregate_by_step.items() if len(values) == 3)
    aggregate_y = [statistics.fmean(aggregate_by_step[step])
                   for step in aggregate_x]
    ax.plot(aggregate_x, aggregate_y, color="#111111", linewidth=2.5,
            label="Local 3-seed mean")
    ax.axhline(official_mean, color="#9b6b25", linestyle=":", linewidth=1.5,
               label=f"Official final aggregate {official_mean:.1f}")
    ax.set(xlabel="Environment steps", ylabel="Training episode return",
           title="EXP-0004 DreamerV3 walker_walk three-seed replication",
           xlim=(0, 500_000), ylim=(0, 1_000))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.review_dir / "curve_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    official_x = np.full(len(official_seed_final), 0.85)
    axes[0].scatter(official_x, official_seed_final, color="#6f7f8f",
                    label="Official seed final means", zorder=3)
    for seed, (value, color) in enumerate(zip(local_final_means, colors)):
        axes[0].scatter(1.15, value, color=color, s=65,
                        label=f"Local seed {seed}", zorder=3)
    axes[0].axhspan(acceptance_lower, acceptance_upper, color="#dce6dc",
                    alpha=0.7, label="Preregistered aggregate range")
    axes[0].axhline(official_mean, color="#263b50", linewidth=1.5,
                    linestyle="--", label="Official aggregate")
    axes[0].scatter([1.15], [aggregate_mean], color="#111111", marker="D",
                    s=70, label="Local 3-seed aggregate", zorder=4)
    axes[0].set(xticks=[0.85, 1.15], xticklabels=["Official", "Local"],
                xlim=(0.65, 1.35), ylabel="Final-window training return",
                title="Paper-comparable primary metric")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7, loc="lower left")

    box = axes[1].boxplot(eval_values_by_seed, patch_artist=True,
                          tick_labels=["Seed 0", "Seed 1", "Seed 2"])
    for patch_box, color in zip(box["boxes"], colors):
        patch_box.set_facecolor(color)
        patch_box.set_alpha(0.65)
    axes[1].set(ylabel="Checkpoint eval episode return",
                title="Separate fixed-seed stochastic evaluation")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("EXP-0004 final-window and checkpoint evidence")
    fig.tight_layout()
    fig.savefig(args.review_dir / "final_window_and_eval.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
