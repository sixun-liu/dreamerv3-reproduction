#!/usr/bin/env python3
"""Analyze the frozen EXP-0008 Cheetah Run five-seed replication."""

from __future__ import annotations

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


EXPERIMENT_ID = "EXP-0008"
REFERENCE_SHA256 = "8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4"
TASK = "dmc_cheetah_run"
SEEDS = tuple(range(5))
EXPECTED_CHECKPOINT_STEP = 250_000
FINAL_WINDOW = (470_000, 500_000)
PAPER_SCORE = 614.0


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


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if sha256(path) != REFERENCE_SHA256:
        raise ValueError("Official reference hash drift")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for row in json.load(handle) if row["task"] == TASK]
    if len(rows) != len(SEEDS):
        raise ValueError(f"Expected five official seeds, found {len(rows)}")
    rows.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != list(SEEDS):
        raise ValueError("Official seed identities changed")
    xs = np.asarray(rows[0]["xs"], dtype=float)
    ys = np.asarray([row["ys"] for row in rows], dtype=float)
    if any(not np.array_equal(xs, np.asarray(row["xs"], dtype=float)) for row in rows):
        raise ValueError("Official seed x axes differ")
    if not np.isfinite(ys).all():
        raise ValueError("Official reference contains non-finite values")
    return xs, ys


def aggregate_curve(rows: list[dict], xs: np.ndarray, width: int = 10_000) -> list[dict]:
    points = np.asarray(
        [(float(row["step"]), float(row["episode/score"])) for row in rows],
        dtype=float,
    )
    output = []
    for x in xs:
        values = points[(points[:, 0] > x - width) & (points[:, 0] <= x), 1]
        if len(values):
            output.append(
                {
                    "step": int(x),
                    "count": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return output


def parse_signal(path: Path, field: str) -> datetime:
    value = json.loads(path.read_text(encoding="utf-8"))[field]
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def t95_interval(values: list[float]) -> list[float]:
    # Student t critical value for df=4; the seed count is frozen at five.
    mean = statistics.fmean(values)
    half = 2.776445 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def analyze(args: argparse.Namespace) -> dict:
    ref_x, ref_y = load_reference(args.reference)
    official_seed_finals = [float(np.mean(seed_curve[-3:])) for seed_curve in ref_y]
    official_final = {
        "seed_final_three_curve_point_means": official_seed_finals,
        "mean": statistics.fmean(official_seed_finals),
        "std_population": statistics.pstdev(official_seed_finals),
        "range": [min(official_seed_finals), max(official_seed_finals)],
        "paper_rounded_score": int(PAPER_SCORE),
    }

    per_seed = []
    local_curves: dict[int, list[dict]] = {}
    integrity_passes = []
    for seed in SEEDS:
        run_dir = args.matrix_root / f"s{seed:03d}"
        completed_sibling = Path(str(run_dir) + ".completed")
        if not completed_sibling.is_file() or not (run_dir / ".completed").is_file():
            raise ValueError(f"Seed {seed} completion beacon is missing")
        integrity = json.loads((run_dir / "integrity.json").read_text(encoding="utf-8"))
        integrity_passes.append(bool(integrity.get("passed")))
        rows = load_jsonl(run_dir / "train" / "scores.jsonl")
        scores = [float(row["episode/score"]) for row in rows]
        if not scores or not all(math.isfinite(value) for value in scores):
            raise ValueError(f"Seed {seed} scores are empty or non-finite")
        curve = aggregate_curve(rows, ref_x)
        local_curves[seed] = curve

        final_scores = [
            float(row["episode/score"])
            for row in rows
            if FINAL_WINDOW[0] < float(row["step"]) <= FINAL_WINDOW[1]
        ]
        early_scores = [
            float(row["episode/score"])
            for row in rows
            if 0 < float(row["step"]) <= 250_000
        ]
        late_scores = [
            float(row["episode/score"])
            for row in rows
            if 250_000 < float(row["step"]) <= 500_000
        ]
        if not final_scores or not early_scores or not late_scores:
            raise ValueError(f"Seed {seed} lacks a frozen score window")
        final_stats = describe(final_scores)
        started = parse_signal(Path(str(run_dir) + ".started"), "started_at")
        completed = parse_signal(completed_sibling, "completed_at")
        per_seed.append(
            {
                "seed": seed,
                "episode_count": len(scores),
                "curve_bin_count": len(curve),
                "final_window": final_stats,
                "first_half_episode_mean": statistics.fmean(early_scores),
                "second_half_episode_mean": statistics.fmean(late_scores),
                "second_half_above_first": statistics.fmean(late_scores)
                > statistics.fmean(early_scores),
                "wall_hours": (completed - started).total_seconds() / 3600,
                "checkpoint_sha256": integrity["checkpoint_sha256"],
                "integrity_passed": bool(integrity["passed"]),
            }
        )

    local_seed_finals = [row["final_window"]["mean"] for row in per_seed]
    local_mean = statistics.fmean(local_seed_finals)
    official_range = official_final["range"]
    gates = {
        "matrix_completion_signal": Path(str(args.matrix_root) + ".completed").is_file(),
        "all_five_integrity_passed": all(integrity_passes) and len(integrity_passes) == 5,
        "aggregate_final_mean_inside_official_seed_range": bool(
            official_range[0] <= local_mean <= official_range[1]
        ),
        "same_direction_learning_at_least_four_seeds": sum(
            row["second_half_above_first"] for row in per_seed
        )
        >= 4,
    }
    gates["primary_replication_gate"] = bool(
        gates["matrix_completion_signal"]
        and gates["all_five_integrity_passed"]
        and gates["aggregate_final_mean_inside_official_seed_range"]
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "task": TASK,
        "protocol": {
            "runtime_upstream_commit": "2411f7d136832378c0291c587cdbf2fca6506873",
            "runtime_compatibility_commit": "6642b941f578cd72147bc2be3c3343d5bc72931c",
            "model": "size12m",
            "agent_decisions_per_seed": 250_000,
            "environment_steps_per_seed": 500_000,
            "action_repeat": 2,
            "num_envs": 16,
            "replay_ratio": 512,
            "seeds": list(SEEDS),
            "dmc_environment_seed_controlled": False,
        },
        "aggregation": {
            "local_primary": "per-seed mean of complete training episodes with 470K < step <= 500K, then mean across five seeds",
            "official": "per-seed mean of the final three exported 10K curve points (470K, 480K, 490K), then mean across five seeds",
            "comparability_limit": "The public JSON export pipeline is unavailable; the local and official final summaries are semantically close but not sample-identical.",
        },
        "official": official_final,
        "local": {
            "seed_final_window_means": local_seed_finals,
            "mean": local_mean,
            "std_sample": statistics.stdev(local_seed_finals),
            "std_population": statistics.pstdev(local_seed_finals),
            "ci95_t_df4": t95_interval(local_seed_finals),
            "range": [min(local_seed_finals), max(local_seed_finals)],
            "paper_score_absolute_error": local_mean - PAPER_SCORE,
            "paper_score_relative_error": (local_mean - PAPER_SCORE) / PAPER_SCORE,
        },
        "per_seed": per_seed,
        "gates": gates,
        "limitations": [
            "The runtime is an author reimplementation from 2024, not the exact 2023 training artifact.",
            "The 2411 runtime does not pass the agent seed into DMC environment construction.",
            "The official raw log-to-JSON exporter is unavailable, so local raw final-window episodes do not exactly reconstruct each official curve point.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "seed",
            "episode_count",
            "curve_bin_count",
            "final_window_count",
            "final_window_mean",
            "final_window_median",
            "first_half_episode_mean",
            "second_half_episode_mean",
            "second_half_above_first",
            "wall_hours",
            "integrity_passed",
            "checkpoint_sha256",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_seed:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "episode_count": row["episode_count"],
                    "curve_bin_count": row["curve_bin_count"],
                    "final_window_count": row["final_window"]["count"],
                    "final_window_mean": row["final_window"]["mean"],
                    "final_window_median": row["final_window"]["median"],
                    "first_half_episode_mean": row["first_half_episode_mean"],
                    "second_half_episode_mean": row["second_half_episode_mean"],
                    "second_half_above_first": row["second_half_above_first"],
                    "wall_hours": row["wall_hours"],
                    "integrity_passed": row["integrity_passed"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )

    curve_rows = []
    for seed, curve in local_curves.items():
        for row in curve:
            index = int(np.where(ref_x == row["step"])[0][0])
            curve_rows.append(
                {
                    "seed": seed,
                    **row,
                    "official_mean": float(np.mean(ref_y[:, index])),
                    "official_min": float(np.min(ref_y[:, index])),
                    "official_max": float(np.max(ref_y[:, index])),
                }
            )
    with (args.output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_curve.fill_between(
        ref_x,
        np.min(ref_y, axis=0),
        np.max(ref_y, axis=0),
        color="#d0d0d0",
        alpha=0.55,
        label="Official 5-seed range",
    )
    ax_curve.plot(ref_x, np.mean(ref_y, axis=0), color="#202020", linewidth=2.2, label="Official mean")
    colors = plt.get_cmap("tab10")
    for seed in SEEDS:
        curve = local_curves[seed]
        ax_curve.plot(
            [row["step"] for row in curve],
            [row["median"] for row in curve],
            color=colors(seed),
            linewidth=1.3,
            alpha=0.82,
            label=f"Local seed {seed}",
        )
    ax_curve.set(
        xlabel="Environment steps",
        ylabel="Training episode score",
        title="Cheetah Run learning curves",
        xlim=(0, 500_000),
        ylim=(0, 1000),
    )
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(fontsize=8, loc="lower right")

    positions = np.arange(5)
    ax_final.scatter(positions - 0.08, official_seed_finals, color="#303030", s=45, label="Official seed finals")
    ax_final.scatter(positions + 0.08, local_seed_finals, color="#c23b22", s=45, label="Local seed finals")
    ax_final.axhline(official_final["mean"], color="#303030", linewidth=1.6, linestyle="--", label="Official mean")
    ax_final.axhline(local_mean, color="#c23b22", linewidth=1.6, linestyle="--", label="Local mean")
    ax_final.axhspan(official_range[0], official_range[1], color="#d0d0d0", alpha=0.35)
    ax_final.set(
        xlabel="Seed index",
        ylabel="Final score summary",
        title="Final-window numerical comparison",
        xticks=positions,
        ylim=(0, 1000),
    )
    ax_final.grid(axis="y", alpha=0.2)
    ax_final.legend(fontsize=8, loc="lower right")
    fig.suptitle("EXP-0008 | DreamerV3 2411f7d lineage | DMC Cheetah Run", fontsize=14)
    fig.tight_layout()
    figure = args.review_dir / "cheetah_replication.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = (
        "# EXP-0008 结果\n\n"
        f"- 五 seed 完整性门：`{gates['all_five_integrity_passed']}`\n"
        f"- 主复现门：`{gates['primary_replication_gate']}`\n"
        f"- 本地五 seed final-window：`{local_mean:.2f} +/- {summary['local']['std_population']:.2f}`\n"
        f"- 官方五 seed final-three-points：`{official_final['mean']:.2f} +/- {official_final['std_population']:.2f}`\n"
        f"- 论文 Table 11：`{int(PAPER_SCORE)}`\n\n"
        "主指标使用本地完整 training episodes 的末 30K 窗口；官方值来自公开曲线最后三个点。"
        "由于原始导出流水线未公开，两者不是逐样本同构。\n"
    )
    (args.output_dir / "RESULT.md").write_text(result, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
