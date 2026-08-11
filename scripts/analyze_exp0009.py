#!/usr/bin/env python3
"""Analyze the frozen EXP-0009 Reacher Hard three-arm pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("baseline", "no_reward_value", "no_reconstruction")
LABELS = {
    "baseline": "Dreamer baseline",
    "no_reward_value": "No reward & value gradients",
    "no_reconstruction": "No reconstruction gradients",
}
COLORS = {
    "baseline": "#167D4A",
    "no_reward_value": "#D18B16",
    "no_reconstruction": "#B23A48",
}
TOTAL_ENV_STEPS = 1_000_000
BIN_WIDTH = 100_000


def load_jsonl(path: Path) -> list[dict]:
  with path.open(encoding="utf-8") as handle:
    return [json.loads(line) for line in handle if line.strip()]


def finite(values: list[float]) -> bool:
  return all(math.isfinite(value) for value in values)


def arm_summary(run_root: Path, arm: str) -> tuple[dict, list[dict]]:
  arm_dir = run_root / arm
  scores = load_jsonl(arm_dir / "train" / "scores.jsonl")
  metrics = load_jsonl(arm_dir / "train" / "metrics.jsonl")
  points = [
      {"step": int(row["step"]), "score": float(row["episode/score"])}
      for row in scores if "episode/score" in row and 0 < int(row["step"]) <= TOTAL_ENV_STEPS
  ]
  if not points or not finite([point["score"] for point in points]):
    raise ValueError(f"{arm}: missing or nonfinite episode scores")
  bins = [[] for _ in range(TOTAL_ENV_STEPS // BIN_WIDTH)]
  for point in points:
    index = min((point["step"] - 1) // BIN_WIDTH, len(bins) - 1)
    bins[index].append(point["score"])
  if any(not values for values in bins):
    raise ValueError(f"{arm}: one or more 100K environment-step bins are empty")
  means = [statistics.fmean(values) for values in bins]
  centers = [BIN_WIDTH * (index + 0.5) for index in range(len(bins))]
  x = np.asarray([0, *centers, TOTAL_ENV_STEPS], dtype=np.float64)
  y = np.asarray([means[0], *means, means[-1]], dtype=np.float64)
  normalized_auc = float(np.trapz(y, x) / TOTAL_ENV_STEPS)
  tail = [point["score"] for point in points if point["step"] > 800_000]
  if not tail:
    raise ValueError(f"{arm}: final 200K window has no complete episodes")
  completed = json.loads((run_root / f"{arm}.completed").read_text(encoding="utf-8"))
  fps = [float(row["fps/policy"]) for row in metrics if "fps/policy" in row]
  ratios = [float(row["replay/replay_ratio"]) for row in metrics
            if "replay/replay_ratio" in row and math.isfinite(float(row["replay/replay_ratio"]))]
  gpu_mem = [float(value) for row in metrics for key, value in row.items()
             if key == "usage/nvsmi/memory_max/gpu0" and math.isfinite(float(value))]
  summary = {
      "arm": arm,
      "episodes": len(points),
      "bin_width_env_steps": BIN_WIDTH,
      "bin_centers_env_steps": centers,
      "bin_means": means,
      "bin_counts": [len(values) for values in bins],
      "normalized_auc": normalized_auc,
      "final_200k_mean": statistics.fmean(tail),
      "final_200k_std": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
      "final_200k_episodes": len(tail),
      "worst_bin_mean": min(means),
      "wall_seconds": int(completed["wall_seconds"]),
      "output_bytes": int(completed["output_bytes"]),
      "policy_fps_median": statistics.median(fps) if fps else None,
      "replay_ratio_median": statistics.median(ratios) if ratios else None,
      "gpu_memory_metric_max": max(gpu_mem) if gpu_mem else None,
  }
  return summary, points


def write_plot(output: Path, summaries: dict[str, dict], points: dict[str, list[dict]]) -> None:
  fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
  for arm in ARMS:
    raw = points[arm]
    ax.scatter(
        [row["step"] / 1e6 for row in raw], [row["score"] for row in raw],
        s=7, alpha=0.08, color=COLORS[arm], linewidths=0)
    ax.plot(
        np.asarray(summaries[arm]["bin_centers_env_steps"]) / 1e6,
        summaries[arm]["bin_means"], marker="o", markersize=4, linewidth=2.2,
        color=COLORS[arm], label=LABELS[arm])
  ax.set_xlim(0, 1.0)
  ax.set_ylim(bottom=0)
  ax.set_xlabel("Environment steps (million)")
  ax.set_ylabel("Raw episode return")
  ax.set_title("DreamerV3 Reacher Hard: 1M-step single-seed conceptual pilot")
  ax.grid(True, alpha=0.22)
  ax.legend(frameon=False)
  fig.savefig(output, dpi=180)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  summaries = {}
  points = {}
  for arm in ARMS:
    summaries[arm], points[arm] = arm_summary(args.run_root, arm)
  directional = (
      summaries["baseline"]["normalized_auc"] > summaries["no_reconstruction"]["normalized_auc"]
      and summaries["no_reward_value"]["normalized_auc"] > summaries["no_reconstruction"]["normalized_auc"])
  secondary = (
      summaries["baseline"]["normalized_auc"] > summaries["no_reward_value"]["normalized_auc"])
  result = {
      "experiment_id": "EXP-0009",
      "evidence_scope": "conceptual_single_task_single_seed_one_tenth_paper_budget",
      "run_root": str(args.run_root),
      "arms": summaries,
      "directional_gate_passed": directional,
      "baseline_exceeds_no_reward_value": secondary,
      "limitations": [
          "One task and one paired seed cannot establish the 14-task Figure 18 aggregate.",
          "The 1M environment-step budget is one tenth of the paper Reacher Hard horizon.",
          "The runtime is the 2024 author reimplementation, not the exact 2023 training artifact.",
      ],
  }
  (args.output_dir / "summary.json").write_text(
      json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  write_plot(args.output_dir / "learning_curves.png", summaries, points)
  with (args.output_dir / "resource_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=(
        "arm", "wall_seconds", "output_bytes", "policy_fps_median",
        "replay_ratio_median", "gpu_memory_metric_max"))
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in summaries.values())
  rows = [
      "# EXP-0009 人工审查入口",
      "",
      "本页只汇总单任务、单 seed、1M environment-step conceptual pilot。",
      "",
      "| Arm | AUC | final 200K | episodes | wall h |",
      "|---|---:|---:|---:|---:|",
  ]
  for arm in ARMS:
    row = summaries[arm]
    rows.append(
        f"| {LABELS[arm]} | {row['normalized_auc']:.2f} | "
        f"{row['final_200k_mean']:.2f} | {row['episodes']} | {row['wall_seconds'] / 3600:.2f} |")
  rows.extend((
      "",
      f"- 主方向门：{'通过' if directional else '未通过'}",
      f"- baseline > no reward/value：{'是' if secondary else '否'}（次指标）",
      "- 曲线：`learning_curves.png`",
      "- 机器结果：`summary.json`",
  ))
  (args.output_dir / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
