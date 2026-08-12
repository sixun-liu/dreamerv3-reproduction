#!/usr/bin/env python3
"""Build a compact human-review artifact for EXP-0016 scaling selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build(run_root: Path, output_dir: Path) -> dict:
    comparison4 = json.loads(
        (run_root / "envs4_vs_envs2.json").read_text(encoding="utf-8")
    )
    comparison8_path = run_root / "envs8_vs_envs4.json"
    comparison8 = (
        json.loads(comparison8_path.read_text(encoding="utf-8"))
        if comparison8_path.is_file()
        else None
    )
    arms = dict(comparison4["arms"])
    if comparison8:
        arms.update(comparison8["arms"])
        selected = comparison8["comparison"]["selected_environment_count"]
    else:
        selected = comparison4["comparison"]["selected_environment_count"]
    ordered = [arms[name] for name in ("envs2", "envs4", "envs8") if name in arms]
    labels = [row["arm"] for row in ordered]
    fps = [row["steady_window"]["policy_fps_mean"] for row in ordered]
    eta = [row["predicted_added_100k_minutes"] for row in ordered]
    cpu = [row["resource"]["cpu_usage_cores_average"] for row in ordered]
    memory = [row["resource"]["max_cgroup_memory_bytes"] / 2**30 for row in ordered]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = output_dir / "recovery_scaling.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ["#2A6F97" if row["environment_count"] == selected else "#8D99AE" for row in ordered]
    panels = (
        (axes[0, 0], fps, "Steady policy FPS", "Higher is better"),
        (axes[0, 1], eta, "Predicted added-100K time (min)", "Lower is better"),
        (axes[1, 0], cpu, "Average CPU cores", "16-core cgroup quota"),
        (axes[1, 1], memory, "Peak cgroup memory (GiB)", "90-GiB cgroup limit"),
    )
    for axis, values, title, subtitle in panels:
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", fontsize=9, color="#555555")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"EXP-0016 Minecraft recovery scaling | selected envs={selected}", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0016",
        "selected_environment_count": selected,
        "envs8_ran": comparison8 is not None,
        "arms": {row["arm"]: row for row in ordered},
        "envs4_vs_envs2": comparison4["comparison"],
        "envs8_vs_envs4": comparison8["comparison"] if comparison8 else None,
        "figure": str(figure),
        "interpretation_boundary": (
            "Selection optimizes predicted training wall time under integrity and "
            "resource gates; GPU utilization is descriptive, not the objective."
        ),
        "human_visual_confirmation": "pending",
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
