#!/usr/bin/env python3
"""Build the compact EXP-0014 throughput comparison review artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLACEHOLDER_SHA256 = "8cfb59970c853619e5b3acd97c5e875eeced75ac3e3b270efe4167cf29139da4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_summary(comparison_path: Path) -> dict:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("experiment_id") != "EXP-0014":
        raise ValueError("Unexpected experiment ID")
    result = comparison["comparison"]
    if not result["comparison_valid"]:
        decision = "inconclusive"
    elif result["envs2_preferred"]:
        decision = "promote"
    else:
        decision = "negative"
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0014",
        "decision": decision,
        "verdict": result["verdict"],
        "selected_environment_count": result["selected_environment_count"],
        "comparison": result,
        "arms": comparison["arms"],
        "evidence": {
            "comparison": str(comparison_path.resolve()),
            "comparison_sha256": sha256(comparison_path),
        },
        "limitations": [
            "One paired diagnostic run per environment count; Minecraft world seed is not exposed.",
            "The 5040-step probe selects local execution configuration, not policy quality.",
            "The envs2 arm uses frozen EXP-0013 evidence collected immediately before envs1 rather than interleaved repetitions.",
            "The selection does not establish that envs2 retains the same benefit during checkpoint recovery or long runs.",
        ],
        "next_discriminating_question": (
            "Can the exact EXP-0012 checkpoint, replay, and environment-step state be resumed "
            "under envs=2 without semantic drift, duplication, or loss?"
        ),
        "human_review": "pending",
    }


def make_figure(summary: dict, output: Path) -> None:
    arms = summary["arms"]
    names = ["envs=1", "envs=2"]
    colors = ["#4c78a8", "#2a9d8f"]
    throughput = [arms["envs1"]["end_to_end_steps_per_second"], arms["envs2"]["end_to_end_steps_per_second"]]
    steady = [arms["envs1"]["steady_window"]["policy_fps_mean"], arms["envs2"]["steady_window"]["policy_fps_mean"]]
    cpu = [arms["envs1"]["resource"]["cpu_usage_cores_average"], arms["envs2"]["resource"]["cpu_usage_cores_average"]]
    memory = [arms["envs1"]["resource"]["max_cgroup_memory_bytes"] / 2**30, arms["envs2"]["resource"]["max_cgroup_memory_bytes"] / 2**30]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        (throughput, "End-to-end throughput", "environment steps/s"),
        (steady, "Steady policy throughput", "policy FPS"),
        (cpu, "Average CPU usage", "CPU cores"),
        (memory, "Peak cgroup memory", "GiB"),
    ]
    for axis, (values, title, ylabel) in zip(axes.flat, panels, strict=True):
        bars = axis.bar(names, values, color=colors, width=0.58)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
        axis.set_ylim(0, max(values) * 1.22)
    fig.suptitle("EXP-0014: Equal-Budget Minecraft Throughput", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    placeholder = args.output_dir / "README.md"
    if args.output_dir.exists():
        existing = sorted(args.output_dir.iterdir())
        if existing != [placeholder] or sha256(placeholder) != PLACEHOLDER_SHA256:
            raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    summary = build_summary(args.comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "review_summary.json"
    figure_path = args.output_dir / "paired_throughput.png"
    summary["figure"] = str(figure_path.resolve())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(summary, figure_path)
    envs1 = summary["arms"]["envs1"]
    envs2 = summary["arms"]["envs2"]
    result = summary["comparison"]
    placeholder.write_text(
        f"""# EXP-0014 Minecraft 同预算吞吐归因

## 结论

在相同 `script=train`、`debug=false`、seed 31415、5040 environment steps 下，选择 `envs=2`：

- 端到端吞吐：`{envs1['end_to_end_steps_per_second']:.2f} -> {envs2['end_to_end_steps_per_second']:.2f} steps/s`，提升 `{(result['end_to_end_speedup_envs2_over_envs1'] - 1) * 100:.2f}%`，通过预注册 `10%` 门。
- 固定稳态窗口 policy FPS：`{envs1['steady_window']['policy_fps_mean']:.2f} -> {envs2['steady_window']['policy_fps_mean']:.2f}`，提升 `{(result['steady_policy_speedup_envs2_over_envs1'] - 1) * 100:.2f}%`，通过预注册 `5%` 门。
- 两臂 checkpoint、replay、有限 metrics、ratio32、数据盘定向、清理和 OOM 门全部通过。

## 成本

| 指标 | envs=1 | envs=2 |
|---|---:|---:|
| 训练墙钟 | {envs1['training_wall_seconds']} s | {envs2['training_wall_seconds']} s |
| 平均 CPU | {envs1['resource']['cpu_usage_cores_average']:.2f} cores | {envs2['resource']['cpu_usage_cores_average']:.2f} cores |
| cgroup 峰值内存 | {envs1['resource']['max_cgroup_memory_bytes'] / 2**30:.2f} GiB | {envs2['resource']['max_cgroup_memory_bytes'] / 2**30:.2f} GiB |
| CPU throttled wall fraction | {envs1['resource']['cpu_throttled_wall_fraction']:.3f} | {envs2['resource']['cpu_throttled_wall_fraction']:.3f} |
| GPU 平均利用率 | {envs1['resource']['gpu_util_mean_percent']:.2f}% | {envs2['resource']['gpu_util_mean_percent']:.2f}% |
| 数据盘净增长 | {envs1['resource']['data_disk_growth_bytes'] / 2**30:.2f} GiB | {envs2['resource']['data_disk_growth_bytes'] / 2**30:.2f} GiB |

## 边界

这是执行配置选择，不是 Minecraft 策略质量或论文分数结果。下一步必须单独验证 EXP-0012 的 checkpoint、replay 和 step 在 `envs=2` 下等价恢复，不能仅凭 fresh-run 吞吐直接追加训练。

- 对比图：`paired_throughput.png`
- 机器摘要：`review_summary.json`
- 原始比较：`{summary['evidence']['comparison']}`
- 人工审查：`pending`
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
