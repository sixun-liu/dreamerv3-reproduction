#!/usr/bin/env python3
"""Plot local DMC walker_walk training returns against frozen reference curves."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_reference(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [row for row in json.load(handle) if row["task"] == "dmc_walker_walk"]
    if not rows:
        raise ValueError("No dmc_walker_walk rows in reference")
    xs = np.asarray(rows[0]["xs"], dtype=float)
    ys = np.asarray([row["ys"] for row in rows], dtype=float)
    if any(not np.array_equal(xs, np.asarray(row["xs"], dtype=float)) for row in rows):
        raise ValueError("Reference seeds do not share an x-axis")
    return xs, ys


def load_local(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "episode/score" in row:
                records.append((float(row["step"]), float(row["episode/score"])))
    if not records:
        raise ValueError("No episode/score records in local scores")
    return np.asarray(records, dtype=float)


def aggregate(local, xs, width):
    output = []
    for x in xs:
        values = local[(local[:, 0] > x - width) & (local[:, 0] <= x), 1]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=10_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_x, ref_y = load_reference(args.reference)
    local = load_local(args.scores)
    analysis_x = np.unique(np.append(ref_x, 500_000.0))
    bins = aggregate(local, analysis_x, args.bin_width)
    if not bins:
        raise ValueError("No completed local bin overlaps the reference x-axis")

    for row in bins:
        index = int(np.argmin(np.abs(ref_x - row["step"])))
        values = ref_y[:, index]
        row.update({
            "reference_step": float(ref_x[index]),
            "reference_median": float(np.median(values)),
            "reference_min": float(np.min(values)),
            "reference_max": float(np.max(values)),
            "inside_reference_range": bool(np.min(values) <= row["median"] <= np.max(values)),
        })

    csv_path = args.output_dir / "curve_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=bins[0].keys())
        writer.writeheader()
        writer.writerows(bins)

    summary = {
        "local_episode_count": int(len(local)),
        "local_last_step": float(local[:, 0].max()),
        "bin_width": args.bin_width,
        "completed_bins": len(bins),
        "latest_bin": bins[-1],
        "official_seeds": int(ref_y.shape[0]),
        "official_last_step": float(ref_x[-1]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_ref_x = np.append(ref_x, 500_000.0)
    plot_ref_y = np.column_stack([ref_y, ref_y[:, -1]])
    ref_median = np.median(plot_ref_y, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(plot_ref_x, plot_ref_y.min(0), plot_ref_y.max(0), color="#c7d4e8", alpha=0.65,
                    label="Official seeds 0-4 range")
    ax.plot(plot_ref_x, ref_median, color="#315b8a", linewidth=2.0, label="Official median")
    local_x = np.asarray([row["step"] for row in bins])
    local_median = np.asarray([row["median"] for row in bins])
    local_mean = np.asarray([row["mean"] for row in bins])
    ax.plot(local_x, local_median, color="#b33a3a", linewidth=2.2, marker="o",
            markersize=3.5, label="Local seed 0 bin median")
    ax.plot(local_x, local_mean, color="#d58b2a", linewidth=1.3, alpha=0.9,
            label="Local seed 0 bin mean")
    ax.axhline(936, color="#4f7f52", linestyle=":", linewidth=1.2,
               label="Paper table final: 936")
    ax.set(xlabel="Environment steps", ylabel="Episode score",
           title="DreamerV3 DMC walker_walk reproduction")
    ax.set_xlim(0, 500_000)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(args.output_dir / "curve_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
