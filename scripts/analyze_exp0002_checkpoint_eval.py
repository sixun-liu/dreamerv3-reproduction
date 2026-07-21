#!/usr/bin/env python3
"""Summarize the frozen EXP-0002 checkpoint evaluation."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=64)
    parser.add_argument("--minimum-mean-score", type=float, default=800.0)
    args = parser.parse_args()

    scores = load_jsonl(args.run_dir / "scores.jsonl")
    metrics = load_jsonl(args.run_dir / "metrics.jsonl")
    values = [float(row["episode/score"]) for row in scores]
    if not values:
        raise ValueError("No episode scores found")

    with args.frozen_config.open(encoding="utf-8") as handle:
        frozen_config = yaml.safe_load(handle)
    with (args.run_dir / "config.yaml").open(encoding="utf-8") as handle:
        generated_config = yaml.safe_load(handle)

    train_keys = sorted({
        key for row in metrics for key in row if key.startswith("train/")})
    summary = {
        "experiment_id": "EXP-0002",
        "run_dir": str(args.run_dir),
        "policy_semantics": "mode=eval with stochastic continuous-action sampling",
        "episode_count": len(values),
        "score": {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std_population": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
            "all_finite": all(math.isfinite(value) for value in values),
        },
        "logged_environment_steps": {
            "first_complete_episode": int(scores[0]["step"]),
            "last_complete_episode": int(scores[-1]["step"]),
        },
        "integrity": {
            "completed_signal": (args.run_dir / ".completed").exists(),
            "config_semantically_equal": frozen_config == generated_config,
            "train_metric_keys": train_keys,
        },
        "preregistered_gate": {
            "minimum_episodes": args.minimum_episodes,
            "minimum_mean_score": args.minimum_mean_score,
        },
    }
    summary["preregistered_gate"]["passed"] = bool(
        summary["episode_count"] >= args.minimum_episodes
        and summary["score"]["mean"] >= args.minimum_mean_score
        and summary["score"]["all_finite"]
        and summary["integrity"]["completed_signal"]
        and summary["integrity"]["config_semantically_equal"]
        and not train_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "episode_scores.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "episode_score"))
        writer.writeheader()
        writer.writerows({
            "step": int(row["step"]),
            "episode_score": float(row["episode/score"]),
        } for row in scores)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(values, bins=12, color="#3f6f8f", edgecolor="white")
    axes[0].axvline(summary["score"]["mean"], color="#b34234", linewidth=2,
                    label=f"Mean {summary['score']['mean']:.1f}")
    axes[0].axvline(summary["score"]["median"], color="#2e7145", linewidth=2,
                    linestyle="--", label=f"Median {summary['score']['median']:.1f}")
    axes[0].set(xlabel="Episode score", ylabel="Count", title="Score distribution")
    axes[0].legend()

    episode_index = range(1, len(values) + 1)
    axes[1].plot(episode_index, values, color="#3f6f8f", marker="o",
                 markersize=3, linewidth=1)
    axes[1].axhline(args.minimum_mean_score, color="#b34234", linestyle=":",
                    linewidth=1.5, label="Preregistered mean threshold")
    axes[1].set(xlabel="Completed episode", ylabel="Score",
                title="Fixed-seed stochastic evaluation")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("EXP-0002 checkpoint evaluation (64 episodes)")
    fig.tight_layout()
    fig.savefig(args.review_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
