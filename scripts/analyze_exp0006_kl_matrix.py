#!/usr/bin/env python3
"""Validate and analyze the frozen EXP-0006 KL mechanism matrix."""

from __future__ import annotations

import argparse
import csv
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


ARMS = ("baseline", "e1", "p4")
SEEDS = (0, 1)
COLORS = {"baseline": "#2f5d8c", "e1": "#c05a3d", "p4": "#2f7d57"}
LABELS = {"baseline": "Baseline", "e1": "E1: no free bits", "p4": "P4 reconstructed"}
ERROR_MARKERS = ("Traceback", "Out of memory", "ResourceExhaustedError")
METRIC_KEYS = (
    "train/kl_raw",
    "train/kl_raw_std",
    "train/kl_below_0p1",
    "train/kl_below_0p5",
    "train/kl_below_1p0",
    "train/dyn_ent",
    "train/rep_ent",
    "train/loss/orientations",
    "train/loss/height",
    "train/loss/velocity",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(path: Path, field: str) -> datetime:
    data = json.loads(path.read_text(encoding="utf-8"))
    return datetime.fromisoformat(data[field].replace("Z", "+00:00"))


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "all_finite": all(math.isfinite(value) for value in values),
    }


def curve_bins(rows: list[dict], width: int = 10_000) -> list[dict]:
    points = np.asarray([
        (float(row["step"]), float(row["episode/score"])) for row in rows
    ])
    output = []
    for end in range(width, 500_001, width):
        values = points[(points[:, 0] > end - width) & (points[:, 0] <= end), 1]
        if len(values):
            output.append({
                "step": end,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return output


def bin_auc(bins: list[dict], lower: int, upper: int) -> float:
    values = [row["mean"] for row in bins if lower < row["step"] <= upper]
    if not values:
        raise ValueError(f"No score bins in ({lower}, {upper}]")
    return statistics.fmean(values)


def metric_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        if not all(key in row for key in METRIC_KEYS):
            continue
        record = {"step": float(row["step"])}
        record.update({key: float(row[key]) for key in METRIC_KEYS})
        record["train/reconstruction"] = sum(
            record[key] for key in (
                "train/loss/orientations",
                "train/loss/height",
                "train/loss/velocity",
            )
        )
        record["train/entropy_gap"] = (
            record["train/dyn_ent"] - record["train/rep_ent"]
        )
        output.append(record)
    return output


def window_metrics(rows: list[dict], lower: int, upper: int) -> dict:
    selected = [row for row in rows if lower < row["step"] <= upper]
    if not selected:
        raise ValueError(f"No mechanism metrics in ({lower}, {upper}]")
    keys = [key for key in selected[0] if key != "step"]
    return {
        key.removeprefix("train/"): statistics.fmean(row[key] for row in selected)
        for key in keys
    } | {"report_count": len(selected)}


def compare_configs(expected: dict, actual: dict, *, eval_config: bool) -> bool:
    if eval_config:
        expected = json.loads(json.dumps(expected))
        expected["run"]["from_checkpoint"] = actual["run"]["from_checkpoint"]
    return expected == actual


def analyze_run(root: Path, configs: Path, arm: str, seed: int) -> tuple[dict, list[dict], list[dict]]:
    run = root / arm / f"s{seed:03d}"
    train_scores = load_jsonl(run / "train" / "scores.jsonl")
    train_metrics = load_jsonl(run / "train" / "metrics.jsonl")
    eval_scores = load_jsonl(run / "eval" / "scores.jsonl")
    eval_metrics = load_jsonl(run / "eval" / "metrics.jsonl")
    train_values = [float(row["episode/score"]) for row in train_scores]
    eval_values = [float(row["episode/score"]) for row in eval_scores]
    final_values = [
        float(row["episode/score"]) for row in train_scores
        if 470_000 < float(row["step"]) <= 500_000
    ]
    bins = curve_bins(train_scores)
    mechanisms = metric_rows(train_metrics)
    early = window_metrics(mechanisms, 0, 250_000)
    late = window_metrics(mechanisms, 400_000, 500_000)

    expected_train = load_yaml(configs / f"exp0006_{arm}_s{seed:03d}_train.yaml")
    expected_eval = load_yaml(configs / f"exp0006_{arm}_s{seed:03d}_eval.yaml")
    actual_train = load_yaml(run / "train" / "config.yaml")
    actual_eval = load_yaml(run / "eval" / "config.yaml")
    checkpoint = Path((run / "checkpoint_path.txt").read_text(encoding="utf-8").strip())
    errors = []
    for logfile in (run / "train_stdout.log", run / "eval_stdout.log"):
        text = logfile.read_text(encoding="utf-8", errors="replace")
        errors.extend(
            f"{logfile.name}:{marker}" for marker in ERROR_MARKERS if marker in text
        )
    relevant_metric_values = [
        value for row in mechanisms for key, value in row.items() if key != "step"
    ]
    eval_train_keys = sorted({
        key for row in eval_metrics for key in row if key.startswith("train/")
    })
    integrity = {
        "root_completed": (run / ".completed").exists(),
        "train_completed": (run / "train.completed").exists(),
        "eval_completed": (run / "eval.completed").exists(),
        "checkpoint_step": int((run / "checkpoint_step.txt").read_text().strip()),
        "checkpoint_agent_sha256": sha256(checkpoint / "agent.pkl"),
        "checkpoint_step_sha256": sha256(checkpoint / "step.pkl"),
        "train_config_match": compare_configs(expected_train, actual_train, eval_config=False),
        "eval_config_match_except_checkpoint": compare_configs(
            expected_eval, actual_eval, eval_config=True
        ),
        "error_markers": errors,
        "train_scores_finite": all(math.isfinite(value) for value in train_values),
        "mechanism_metrics_finite": all(
            math.isfinite(value) for value in relevant_metric_values
        ),
        "eval_scores_finite": all(math.isfinite(value) for value in eval_values),
        "eval_train_metric_keys": eval_train_keys,
    }
    integrity["passed"] = bool(
        integrity["root_completed"]
        and integrity["train_completed"]
        and integrity["eval_completed"]
        and integrity["checkpoint_step"] == 250_000
        and integrity["train_config_match"]
        and integrity["eval_config_match_except_checkpoint"]
        and not integrity["error_markers"]
        and integrity["train_scores_finite"]
        and integrity["mechanism_metrics_finite"]
        and integrity["eval_scores_finite"]
        and not integrity["eval_train_metric_keys"]
        and len(eval_values) >= 64
    )
    started = parse_time(run / ".started", "started_at")
    completed = parse_time(run / ".completed", "completed_at")
    summary = {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run),
        "wall_hours": (completed - started).total_seconds() / 3600,
        "integrity": integrity,
        "train": {
            "episode_count": len(train_values),
            "last_score_step": max(float(row["step"]) for row in train_scores),
            "final_30k": describe(final_values),
            "score_auc_0_250k": bin_auc(bins, 0, 250_000),
            "score_auc_250_500k": bin_auc(bins, 250_000, 500_000),
            "curve_bin_count": len(bins),
        },
        "mechanism": {"early": early, "late": late},
        "eval": {**describe(eval_values), "seed": 10000},
    }
    summary["posterior_collapse_candidate"] = bool(
        late["kl_raw"] < 0.1 and late["kl_below_0p1"] > 0.9
    )
    return summary, bins, mechanisms


def paired_comparisons(runs: list[dict]) -> list[dict]:
    by_key = {(row["arm"], row["seed"]): row for row in runs}
    output = []
    for seed in SEEDS:
        for lhs, rhs, name in (
            ("e1", "baseline", "e1_vs_baseline"),
            ("p4", "e1", "p4_vs_e1"),
        ):
            a = by_key[(lhs, seed)]
            b = by_key[(rhs, seed)]
            a_late, b_late = a["mechanism"]["late"], b["mechanism"]["late"]
            a_final = a["train"]["final_30k"]["mean"]
            b_final = b["train"]["final_30k"]["mean"]
            output.append({
                "comparison": name,
                "seed": seed,
                "late_kl_delta": a_late["kl_raw"] - b_late["kl_raw"],
                "late_kl_ratio": a_late["kl_raw"] / b_late["kl_raw"],
                "late_kl_below_1p0_delta": (
                    a_late["kl_below_1p0"] - b_late["kl_below_1p0"]
                ),
                "late_rep_entropy_delta": a_late["rep_ent"] - b_late["rep_ent"],
                "late_reconstruction_delta": (
                    a_late["reconstruction"] - b_late["reconstruction"]
                ),
                "late_reconstruction_ratio": (
                    a_late["reconstruction"] / b_late["reconstruction"]
                ),
                "final_30k_score_delta": a_final - b_final,
                "final_30k_score_ratio": a_final / b_final,
                "early_auc_delta": (
                    a["train"]["score_auc_0_250k"]
                    - b["train"]["score_auc_0_250k"]
                ),
                "late_auc_delta": (
                    a["train"]["score_auc_250_500k"]
                    - b["train"]["score_auc_250_500k"]
                ),
            })
    return output


def mean_series(series: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(10_000, 500_001, 10_000, dtype=float)
    interpolated = []
    for rows in series:
        xs = np.asarray([row["step"] for row in rows], dtype=float)
        ys = np.asarray([row[key] for row in rows], dtype=float)
        interpolated.append(np.interp(grid, xs, ys))
    return grid, np.mean(interpolated, axis=0)


def plot_timeseries(curves: dict, mechanisms: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "score", "Training episode score"),
        (axes[0, 1], "train/kl_raw", "Raw KL (nat)"),
        (axes[1, 0], "train/kl_below_1p0", "Fraction raw KL < 1 nat"),
        (axes[1, 1], "train/rep_ent", "Posterior entropy"),
    )
    for arm in ARMS:
        color = COLORS[arm]
        source = curves[arm] if panels[0][1] == "score" else mechanisms[arm]
        for rows in curves[arm]:
            axes[0, 0].plot(
                [row["step"] for row in rows], [row["mean"] for row in rows],
                color=color, alpha=0.22, linewidth=1,
            )
        grid, mean = mean_series(curves[arm], "mean")
        axes[0, 0].plot(grid, mean, color=color, linewidth=2.2, label=LABELS[arm])
        for axis, key, _ in panels[1:]:
            for rows in mechanisms[arm]:
                axis.plot(
                    [row["step"] for row in rows], [row[key] for row in rows],
                    color=color, alpha=0.22, linewidth=1,
                )
            grid, mean = mean_series(mechanisms[arm], key)
            axis.plot(grid, mean, color=color, linewidth=2.2, label=LABELS[arm])
    axes[0, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    for axis, _, title in panels:
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_summary(runs: list[dict], output: Path) -> None:
    by_arm = {arm: [row for row in runs if row["arm"] == arm] for arm in ARMS}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    specs = (
        ("Late raw KL", lambda row: row["mechanism"]["late"]["kl_raw"]),
        ("Late reconstruction loss", lambda row: row["mechanism"]["late"]["reconstruction"]),
        ("Final-30K score", lambda row: row["train"]["final_30k"]["mean"]),
    )
    x = np.arange(len(ARMS))
    for axis, (title, getter) in zip(axes, specs):
        means = [statistics.fmean(getter(row) for row in by_arm[arm]) for arm in ARMS]
        axis.bar(x, means, color=[COLORS[arm] for arm in ARMS], alpha=0.72, width=0.62)
        for index, arm in enumerate(ARMS):
            values = [getter(row) for row in by_arm[arm]]
            axis.scatter([index - 0.08, index + 0.08], values, color="#202020", s=25, zorder=3)
        axis.set_xticks(x, ["Baseline", "E1", "P4"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    curves = {arm: [] for arm in ARMS}
    mechanisms = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            summary, score_curve, mechanism_curve = analyze_run(
                args.run_root, args.configs, arm, seed
            )
            runs.append(summary)
            curves[arm].append(score_curve)
            mechanisms[arm].append(mechanism_curve)

    comparisons = paired_comparisons(runs)
    integrity_passed = all(row["integrity"]["passed"] for row in runs)
    e1_direction = all(
        row["late_kl_delta"] < 0
        for row in comparisons if row["comparison"] == "e1_vs_baseline"
    )
    p4_kl_direction = all(
        row["late_kl_delta"] < 0
        for row in comparisons if row["comparison"] == "p4_vs_e1"
    )
    p4_entropy_direction = all(
        row["late_rep_entropy_delta"] < 0
        for row in comparisons if row["comparison"] == "p4_vs_e1"
    )
    summary = {
        "experiment_id": "EXP-0006",
        "scope": "2026 author reimplementation; seeded DMC walker_walk; conceptual mechanism study",
        "runtime_commit": "ad49802bf7051d36be318a9db742a3e1a9255622",
        "integrity_passed": integrity_passed,
        "preregistered_mechanism_gates": {
            "e1_late_raw_kl_lower_than_baseline_both_seeds": e1_direction,
            "p4_late_raw_kl_lower_than_e1_both_seeds": p4_kl_direction,
            "p4_late_posterior_entropy_lower_than_e1_both_seeds": p4_entropy_direction,
            "p4_raw_kl_or_entropy_gate": p4_kl_direction or p4_entropy_direction,
        },
        "runs": runs,
        "paired_comparisons": comparisons,
        "limitations": [
            "Only two paired seeds on one low-dimensional continuous-control task.",
            "P4 is reconstructed from public code semantics; the original ablation config is unavailable.",
            "Environment seeds align random sources but policies still induce different trajectories.",
            "This evidence does not numerically reproduce paper Figure 6 or Figure 17.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    args.review.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flat_runs = []
    for row in runs:
        flat_runs.append({
            "arm": row["arm"],
            "seed": row["seed"],
            "wall_hours": row["wall_hours"],
            "integrity_passed": row["integrity"]["passed"],
            "early_kl_raw": row["mechanism"]["early"]["kl_raw"],
            "late_kl_raw": row["mechanism"]["late"]["kl_raw"],
            "late_kl_below_1p0": row["mechanism"]["late"]["kl_below_1p0"],
            "late_rep_ent": row["mechanism"]["late"]["rep_ent"],
            "late_reconstruction": row["mechanism"]["late"]["reconstruction"],
            "early_score_auc": row["train"]["score_auc_0_250k"],
            "late_score_auc": row["train"]["score_auc_250_500k"],
            "final_30k_mean": row["train"]["final_30k"]["mean"],
            "eval_mean": row["eval"]["mean"],
            "collapse_candidate": row["posterior_collapse_candidate"],
        })
    write_csv(args.output / "per_run_summary.csv", flat_runs)
    write_csv(args.output / "paired_comparisons.csv", comparisons)
    plot_timeseries(curves, mechanisms, args.review / "mechanism_timeseries.png")
    plot_summary(runs, args.review / "representation_and_performance.png")

    result = [
        "# EXP-0006 Result",
        "",
        f"- 完整性门：`{integrity_passed}`",
        f"- E1 两 seed late raw-KL 均低于 baseline：`{e1_direction}`",
        f"- P4 两 seed late raw-KL 均低于 E1：`{p4_kl_direction}`",
        f"- P4 两 seed posterior entropy 均低于 E1：`{p4_entropy_direction}`",
        "",
        "本结果仅适用于 2026 作者重实现、显式 seeded DMC walker_walk 和冻结的 P4 代码语义；",
        "不构成论文 Figure 6/17 的数值复现。详细数字见 `summary.json` 和 CSV。",
    ]
    (args.output / "RESULT.md").write_text("\n".join(result) + "\n", encoding="utf-8")
    (args.review / "README.md").write_text(
        "# EXP-0006 Review\n\n"
        "- `mechanism_timeseries.png`: score、raw KL、低 KL 比例和 posterior entropy。\n"
        "- `representation_and_performance.png`: late KL、reconstruction 和 final score。\n"
        "- 用户确认：`pending`。\n",
        encoding="utf-8",
    )
    if not integrity_passed:
        raise SystemExit("EXP-0006 integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
