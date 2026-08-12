#!/usr/bin/env python3
"""Build the compact EXP-0019 local-scaling review artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build(comparison_path: Path, output_dir: Path) -> dict:
    result = json.loads(comparison_path.read_text(encoding="utf-8"))
    arms = [result["arms"][name] for name in ("envs4", "envs5")]
    selected = result["comparison"]["selected_environment_count"]
    labels = [row["arm"] for row in arms]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = output_dir / "local_scaling.png"
    colors = ["#2A6F97" if row["environment_count"] == selected else "#8D99AE" for row in arms]
    panels = (
        ("Steady policy FPS", [row["steady_window"]["policy_fps_mean"] for row in arms], "Higher is better"),
        ("Predicted added-100K time (min)", [row["predicted_added_100k_minutes"] for row in arms], "Lower is better"),
        ("Average CPU cores", [row["resource"]["cpu_usage_cores_average"] for row in arms], "16-core cgroup quota"),
        ("CPU throttled CPU-sec / wall-sec", [row["resource"]["cpu_throttled_wall_fraction"] for row in arms], "Lower is better"),
        ("Peak cgroup memory (GiB)", [row["resource"]["max_cgroup_memory_bytes"] / 2**30 for row in arms], "72-GiB run gate"),
        ("Mean GPU utilization (%)", [row["resource"]["gpu_util_mean_percent"] for row in arms], "Descriptive only"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for axis, (title, values, subtitle) in zip(axes.flat, panels, strict=True):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title, pad=23, fontsize=10)
        axis.text(0.5, 1.015, subtitle, transform=axis.transAxes, ha="center", fontsize=8, color="#555555")
        axis.set_ylim(0, max(values) * 1.20 if max(values) else 1)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"EXP-0019 Minecraft local scaling | selected envs={selected}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    summary = {
        **result,
        "figure": str(figure),
        "human_visual_confirmation": "pending",
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# EXP-0019 Review\n\n"
        "本探针只选择当前 `script=train` 路线的局部执行配置，不评价策略质量。\n\n"
        f"机器裁决选择 `envs={selected}`；详细数值见 `review_summary.json`，图见 `local_scaling.png`。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.comparison, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
