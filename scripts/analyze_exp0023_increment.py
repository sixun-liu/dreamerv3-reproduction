#!/usr/bin/env python3
"""Analyze EXP-0023 added replay, evaluation, resources, and 200K comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_exp0012_minecraft import MILESTONE_TIERS, MILESTONES, describe


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay_stepids(root: Path) -> set[bytes]:
    result: set[bytes] = set()
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            for row in chunk["stepid"]:
                key = row.tobytes()
                if key in result:
                    raise ValueError(f"Duplicate replay stepid: {key.hex()}")
                result.add(key)
    return result


def validate_inventory_mapping(inventory_keys: list[str], evaluation: dict) -> dict:
    expected = {
        item: inventory_keys.index(f"inventory/{item}") for item in MILESTONES
    }
    mappings = [episode["milestone_runtime_indices"] for episode in evaluation["episodes"]]
    if len(mappings) != 3 or any(mapping != expected for mapping in mappings):
        raise ValueError(
            "Offline replay inventory schema does not match runtime evaluation mapping"
        )
    return expected


def analyze_added_replay(
    source_replay: Path,
    target_replay: Path,
    inventory_keys: list[str],
) -> dict:
    source_ids = replay_stepids(source_replay)
    indices = {item: inventory_keys.index(f"inventory/{item}") for item in MILESTONES}
    all_target_ids: set[bytes] = set()
    new_ids: set[bytes] = set()
    trajectory_ids: set[bytes] = set()
    new_position = 0
    milestones = {
        item: {
            "observed": False,
            "first_added_replay_position": None,
            "positive_transition_count": 0,
            "successful_trajectory_count": 0,
            "max_inventory_count": 0.0,
        }
        for item in MILESTONES
    }
    positive_trajectories = {item: set() for item in MILESTONES}

    for path in sorted(target_replay.glob("*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            stepids = chunk["stepid"]
            inventory_max = chunk["inventory_max"]
            if len(stepids) != len(inventory_max):
                raise ValueError(f"Replay field length mismatch: {path}")
            if inventory_max.shape[1] != len(inventory_keys):
                raise ValueError(f"Unexpected inventory width: {path}")
            for stepid_row, inventory in zip(stepids, inventory_max, strict=True):
                stepid = stepid_row.tobytes()
                if stepid in all_target_ids:
                    raise ValueError(f"Duplicate target replay stepid: {stepid.hex()}")
                all_target_ids.add(stepid)
                if stepid in source_ids:
                    continue
                new_ids.add(stepid)
                new_position += 1
                trajectory = stepid[:16]
                trajectory_ids.add(trajectory)
                for item, column in indices.items():
                    value = float(inventory[column])
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite inventory value: {item}")
                    milestones[item]["max_inventory_count"] = max(
                        milestones[item]["max_inventory_count"], value
                    )
                    if value > 0:
                        milestones[item]["observed"] = True
                        milestones[item]["positive_transition_count"] += 1
                        positive_trajectories[item].add(trajectory)
                        if milestones[item]["first_added_replay_position"] is None:
                            milestones[item]["first_added_replay_position"] = new_position

    for item in MILESTONES:
        milestones[item]["successful_trajectory_count"] = len(
            positive_trajectories[item]
        )
    return {
        "source_unique_stepids": len(source_ids),
        "target_unique_stepids": len(all_target_ids),
        "source_preserved": source_ids <= all_target_ids,
        "new_unique_stepids": len(new_ids),
        "new_trajectory_prefix_count": len(trajectory_ids),
        "milestones": milestones,
        "tiers": {
            tier: {
                "items": list(items),
                "any_observed": any(milestones[item]["observed"] for item in items),
                "all_observed": all(milestones[item]["observed"] for item in items),
            }
            for tier, items in MILESTONE_TIERS.items()
        },
        "ordering_boundary": (
            "first_added_replay_position follows deterministic file and row scan order; "
            "multi-environment replay is a trajectory forest, so it is not a strict "
            "global environment-time index"
        ),
    }


def summarize_resources(run_dir: Path) -> dict:
    system = load_csv(run_dir / "resource_system.csv")
    gpu = load_csv(run_dir / "resource_gpu.csv")
    completed = json.loads((run_dir / ".completed").read_text(encoding="utf-8"))
    first = system[0]
    last = system[-1]
    sampled_seconds = max(
        float(last["timestamp_epoch"]) - float(first["timestamp_epoch"]), 1e-9
    )
    return {
        "training_wall_seconds": int(completed["training_wall_seconds"]),
        "total_wall_seconds": int(completed["total_wall_seconds"]),
        "output_bytes": int(completed["output_bytes"]),
        "average_cpu_cores": (
            int(last["cpu_usage_usec"]) - int(first["cpu_usage_usec"])
        )
        / 1e6
        / sampled_seconds,
        "cpu_throttled_wall_fraction": (
            int(last["cpu_throttled_usec"])
            - int(first["cpu_throttled_usec"])
        )
        / 1e6
        / sampled_seconds,
        "peak_cgroup_memory_gib": max(
            int(row["cgroup_memory_current_bytes"]) for row in system
        )
        / 2**30,
        "peak_gpu_memory_mib": max(float(row["memory_used_mib"]) for row in gpu),
        "mean_gpu_util_percent": statistics.fmean(
            float(row["gpu_util_percent"]) for row in gpu
        ),
        "peak_gpu_util_percent": max(float(row["gpu_util_percent"]) for row in gpu),
    }


def learning_verdict(added_replay: dict, evaluation: dict) -> dict:
    wooden_replay = added_replay["milestones"]["wooden_pickaxe"]["observed"]
    cobblestone_replay = added_replay["milestones"]["cobblestone"]["observed"]
    cobblestone_eval = evaluation["milestones"]["cobblestone"][
        "successful_episodes"
    ] > 0
    return {
        "maintenance_item": "wooden_pickaxe",
        "maintenance_observed_in_added_replay": wooden_replay,
        "advance_item": "cobblestone",
        "advance_observed_in_added_replay": cobblestone_replay,
        "advance_observed_in_evaluation": cobblestone_eval,
        "advance_observed": cobblestone_replay or cobblestone_eval,
        "hypothesis_supported": wooden_replay and (cobblestone_replay or cobblestone_eval),
    }


def build_figure(
    experiment_id: str,
    baseline_eval: dict,
    evaluation: dict,
    added_replay: dict,
    resources: dict,
    output: Path,
) -> None:
    items = ("log", "planks", "crafting_table", "wooden_pickaxe", "cobblestone")
    x = np.arange(len(items))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#FAFAF7")
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [baseline_eval["milestones"][item]["successful_episodes"] for item in items],
        width,
        label="200K checkpoint",
        color="#8D99AE",
    )
    axes[0, 0].bar(
        x + width / 2,
        [evaluation["milestones"][item]["successful_episodes"] for item in items],
        width,
        label="500K checkpoint",
        color="#2A6F97",
    )
    axes[0, 0].set_xticks(x, items, rotation=20)
    axes[0, 0].set_ylim(0, 3.5)
    axes[0, 0].set_ylabel("Successful episodes out of 3")
    axes[0, 0].set_title("Independent evaluation milestones")
    axes[0, 0].legend(frameon=False)

    baseline_returns = [row["return"] for row in baseline_eval["episodes"]]
    current_returns = [row["return"] for row in evaluation["episodes"]]
    eval_x = np.arange(3)
    axes[0, 1].plot(eval_x, baseline_returns, "o-", label="200K", color="#8D99AE")
    axes[0, 1].plot(eval_x, current_returns, "o-", label="500K", color="#2A6F97")
    axes[0, 1].set_xticks(eval_x)
    axes[0, 1].set_xlabel("Evaluation episode index")
    axes[0, 1].set_ylabel("Raw episode return")
    axes[0, 1].set_title("Uncontrolled-world evaluation returns")
    axes[0, 1].legend(frameon=False)

    trajectories = [
        added_replay["milestones"][item]["successful_trajectory_count"]
        for item in items
    ]
    bars = axes[1, 0].bar(items, trajectories, color="#3A7D44")
    axes[1, 0].set_xticks(x, items, rotation=20)
    axes[1, 0].set_ylabel("Trajectory-prefix count")
    axes[1, 0].set_title("Added 300K replay milestone evidence")
    for bar, value in zip(bars, trajectories, strict=True):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    resource_labels = ("Wall min", "CPU cores", "Peak RAM GiB", "Peak VRAM GiB")
    resource_values = (
        resources["training_wall_seconds"] / 60,
        resources["average_cpu_cores"],
        resources["peak_cgroup_memory_gib"],
        resources["peak_gpu_memory_mib"] / 1024,
    )
    bars = axes[1, 1].bar(resource_labels, resource_values, color="#B07D2B")
    axes[1, 1].set_xticks(np.arange(len(resource_labels)), resource_labels, rotation=15)
    axes[1, 1].set_title("500K increment resource facts")
    for bar, value in zip(bars, resource_values, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{experiment_id} DreamerV3 Minecraft | 200K to 500K increment",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.93,
        "World seeds are uncontrolled; milestone and return comparison is descriptive, not paired causal evidence",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.0)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="EXP-0023")
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--inventory-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    integrity = json.loads((args.train_run / "integrity.json").read_text(encoding="utf-8"))
    evaluation_integrity = json.loads(
        (args.eval_run / "analysis.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (args.eval_run / "evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    baseline_evaluation = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    if (
        not integrity["passed"]
        or not evaluation_integrity["integrity"]["passed"]
        or evaluation["experiment_id"] != args.experiment_id
    ):
        raise ValueError(f"{args.experiment_id} integrity or evaluation gate failed")
    schema = json.loads(args.inventory_schema.read_text(encoding="utf-8"))
    inventory_keys = list(schema["inventory_keys"])
    validate_inventory_mapping(inventory_keys, evaluation)
    added_replay = analyze_added_replay(
        args.source_replay, args.train_run / "train/replay", inventory_keys
    )
    if added_replay["source_unique_stepids"] != 200_000:
        raise ValueError("Unexpected source replay size")
    if added_replay["target_unique_stepids"] != 500_000:
        raise ValueError("Unexpected target replay size")
    if added_replay["new_unique_stepids"] != 300_000:
        raise ValueError("Unexpected added replay size")

    resources = summarize_resources(args.train_run)
    metrics = load_jsonl(args.train_run / "train/metrics.jsonl")
    scores = load_jsonl(args.train_run / "train/scores.jsonl")
    verdict = learning_verdict(added_replay, evaluation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "increment_comparison.png"
    build_figure(
        args.experiment_id,
        baseline_evaluation,
        evaluation,
        added_replay,
        resources,
        figure,
    )
    summary = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "integrity_passed": True,
        "evaluation_integrity": evaluation_integrity,
        "offline_runtime_inventory_mapping_consistent": True,
        "source_step": 200_000,
        "final_step": 500_000,
        "added_replay": added_replay,
        "increment_training_scores": describe(
            [float(row["episode/score"]) for row in scores]
        ),
        "increment_metric_rows": len(metrics),
        "evaluation_200k": baseline_evaluation,
        "evaluation_500k": evaluation,
        "learning_verdict": verdict,
        "resources": resources,
        "figure": str(figure),
        "interpretation_boundary": (
            "Engineering integrity and added-replay membership are exact. Evaluation "
            "comparison is descriptive because Minecraft world seeds are uncontrolled. "
            "This 500K run is still far below the paper-scale Diamond budget."
        ),
        "human_visual_confirmation": "pending",
    }
    summary_path = args.output_dir / "review_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = f"""# {args.experiment_id} Minecraft 500K 增量结果

## 受限裁决

- 工程完整性：`passed`
- 新增 replay 再次出现 wooden_pickaxe：`{str(verdict['maintenance_observed_in_added_replay']).lower()}`
- 新增 replay 出现 cobblestone：`{str(verdict['advance_observed_in_added_replay']).lower()}`
- 终点评测出现 cobblestone：`{str(verdict['advance_observed_in_evaluation']).lower()}`
- 预注册学习假说得到支持：`{str(verdict['hypothesis_supported']).lower()}`

## 资源

- 新增 300K 训练墙钟：`{resources['training_wall_seconds'] / 60:.2f}` 分钟
- 平均 CPU：`{resources['average_cpu_cores']:.2f}` 核
- 峰值内存：`{resources['peak_cgroup_memory_gib']:.2f}` GiB
- 峰值显存：`{resources['peak_gpu_memory_mib']:.0f}` MiB

## 审查入口

- 对比图：`{figure}`
- 机器可读摘要：`{summary_path}`
- 固定 episode-0 视频：`{evaluation['video']}`

## 限制

- 200K 与 500K 评测的 Minecraft 世界 seed 均不可控，因此只作描述性比较。
- 多环境新增 replay 是轨迹森林，首次位置不是严格全局环境时间。
- 500K 仍远低于论文 Diamond 量级，不构成论文钻石结果复现。
"""
    (args.output_dir / "RESULT.md").write_text(result, encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        f"# {args.experiment_id} Review\n\n"
        "- 受限结论：`RESULT.md`\n"
        "- 对比图：`increment_comparison.png`\n"
        "- 机器可读摘要：`review_summary.json`\n"
        "- 人工视觉确认：`pending`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
