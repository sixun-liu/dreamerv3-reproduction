#!/usr/bin/env python3
"""Analyze EXP-0012 training replay, milestones, scores, and resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MILESTONE_TIERS = {
    "L1": ("log", "planks", "crafting_table"),
    "L2": ("wooden_pickaxe", "cobblestone"),
    "L3": ("iron_ore", "iron_pickaxe"),
    "L4": ("diamond",),
}
MILESTONES = tuple(item for values in MILESTONE_TIERS.values() for item in values)


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
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "all_finite": all(math.isfinite(value) for value in values),
    }


def parse_chunk_name(path: Path) -> tuple[str, str, int]:
    parts = path.stem.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected replay chunk name: {path.name}")
    return parts[1], parts[2], int(parts[3])


def validate_chunk_chain(paths: list[Path]) -> None:
    if not paths:
        raise ValueError("Replay contains no chunks")
    parsed = [parse_chunk_name(path) for path in paths]
    for index in range(len(parsed) - 1):
        _, successor, _ = parsed[index]
        next_uuid, _, _ = parsed[index + 1]
        if successor != next_uuid:
            raise ValueError(
                f"Broken replay chain between {paths[index].name} and "
                f"{paths[index + 1].name}"
            )


def load_inventory_schema(path: Path) -> tuple[list[str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = list(data["inventory_keys"])
    expected = {f"inventory/{item}" for item in MILESTONES}
    missing = sorted(expected - set(keys))
    if missing:
        raise ValueError(f"Inventory schema lacks milestones: {missing}")
    return keys, sha256(path)


def analyze_replay(replay_dir: Path, inventory_keys: list[str]) -> dict:
    paths = sorted(replay_dir.glob("*.npz"))
    validate_chunk_chain(paths)
    indices = {item: inventory_keys.index(f"inventory/{item}") for item in MILESTONES}
    segments: list[dict] = []
    current: dict | None = None
    global_step = 0
    stepids: set[bytes] = set()

    def finish(complete: bool, end_reason: str) -> None:
        nonlocal current
        if current is None:
            return
        current["complete"] = complete
        current["end_reason"] = end_reason
        current["return"] = float(current["return"])
        segments.append(current)
        current = None

    for path in paths:
        _, _, declared_length = parse_chunk_name(path)
        with np.load(path, allow_pickle=False) as chunk:
            required = {"inventory_max", "reward", "is_first", "is_last", "stepid"}
            missing = sorted(required - set(chunk.files))
            if missing:
                raise ValueError(f"Replay chunk {path.name} lacks {missing}")
            length = len(chunk["reward"])
            if length != declared_length:
                raise ValueError(f"Replay chunk length mismatch: {path.name}")
            inventory_max = chunk["inventory_max"]
            rewards = chunk["reward"]
            is_first = chunk["is_first"]
            is_last = chunk["is_last"]
            ids = chunk["stepid"]
            if inventory_max.shape != (length, len(inventory_keys)):
                raise ValueError(
                    f"Unexpected inventory shape {inventory_max.shape} in {path.name}"
                )
            if not np.isfinite(inventory_max).all() or not np.isfinite(rewards).all():
                raise ValueError(f"Non-finite replay values in {path.name}")

            for offset in range(length):
                global_step += 1
                stepid = ids[offset].tobytes()
                if stepid in stepids:
                    raise ValueError(f"Duplicate replay stepid at step {global_step}")
                stepids.add(stepid)
                if bool(is_first[offset]):
                    if current is not None:
                        finish(False, "reset_without_is_last")
                    current = {
                        "segment_index": len(segments),
                        "start_step": global_step,
                        "end_step": global_step,
                        "transition_count_including_reset": 0,
                        "return": 0.0,
                        "milestone_max": {item: 0.0 for item in MILESTONES},
                        "milestone_first_step": {item: None for item in MILESTONES},
                    }
                if current is None:
                    raise ValueError(f"Replay did not start with is_first at step {global_step}")
                current["end_step"] = global_step
                current["transition_count_including_reset"] += 1
                current["return"] += float(rewards[offset])
                for item, column in indices.items():
                    value = float(inventory_max[offset, column])
                    current["milestone_max"][item] = max(
                        current["milestone_max"][item], value
                    )
                    if value > 0 and current["milestone_first_step"][item] is None:
                        current["milestone_first_step"][item] = global_step
                if bool(is_last[offset]):
                    finish(True, "is_last")
    finish(False, "budget_boundary")

    milestones = {}
    for item in MILESTONES:
        successful = [
            segment for segment in segments if segment["milestone_max"][item] > 0
        ]
        first_steps = [
            segment["milestone_first_step"][item]
            for segment in successful
            if segment["milestone_first_step"][item] is not None
        ]
        milestones[item] = {
            "successful_segments": len(successful),
            "successful_complete_episodes": sum(
                bool(segment["complete"]) for segment in successful
            ),
            "first_global_step": min(first_steps) if first_steps else None,
            "max_inventory_count": max(
                (segment["milestone_max"][item] for segment in segments), default=0.0
            ),
        }
    tiers = {
        tier: {
            "items": list(items),
            "any_observed": any(
                milestones[item]["first_global_step"] is not None for item in items
            ),
            "all_observed": all(
                milestones[item]["first_global_step"] is not None for item in items
            ),
        }
        for tier, items in MILESTONE_TIERS.items()
    }
    complete = [segment for segment in segments if segment["complete"]]
    return {
        "chunk_count": len(paths),
        "chunk_bytes": sum(path.stat().st_size for path in paths),
        "transition_count": global_step,
        "unique_stepid_count": len(stepids),
        "segment_count": len(segments),
        "complete_episode_count": len(complete),
        "partial_segment_count": len(segments) - len(complete),
        "complete_episode_returns": describe(
            [float(segment["return"]) for segment in complete]
        ),
        "milestones": milestones,
        "tiers": tiers,
        "segments": segments,
    }


def compare_scores(scores: list[dict], replay: dict) -> dict:
    completed = [segment for segment in replay["segments"] if segment["complete"]]
    pairs = list(zip(scores, completed))
    return_matches = [
        math.isclose(
            float(score["episode/score"]),
            float(segment["return"]),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        for score, segment in pairs
    ]
    step_matches = [
        int(score["step"]) == int(segment["end_step"])
        for score, segment in pairs
    ]
    return {
        "score_rows": len(scores),
        "complete_replay_episodes": len(completed),
        "counts_match": len(scores) == len(completed),
        "paired_rows": len(pairs),
        "return_sequence_matches": len(scores) == len(completed) and all(return_matches),
        "terminal_step_sequence_matches": len(scores) == len(completed) and all(step_matches),
    }


def parse_gpu_samples(path: Path) -> dict:
    values = {key: [] for key in ("memory", "utilization", "power", "temperature")}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values["memory"].append(float(row["memory_used_mib"]))
                values["utilization"].append(float(row["gpu_util_percent"]))
                values["power"].append(float(row["power_watts"]))
                values["temperature"].append(float(row["temperature_c"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "sample_count": len(values["memory"]),
        "memory_used_mib_mean": statistics.fmean(values["memory"]),
        "memory_used_mib_max": max(values["memory"]),
        "gpu_util_percent_mean": statistics.fmean(values["utilization"]),
        "gpu_util_percent_max": max(values["utilization"]),
        "power_watts_mean": statistics.fmean(values["power"]),
        "temperature_c_max": max(values["temperature"]),
    }


def plot_training(path: Path, scores: list[dict], replay: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if scores:
        steps = [int(row["step"]) for row in scores]
        returns = [float(row["episode/score"]) for row in scores]
        axes[0].plot(steps, returns, marker="o", markersize=3, linewidth=1.2)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("EXP-0012 Minecraft Diamond, size50m, seed 0")
    axes[0].grid(alpha=0.25)

    y = 0
    labels = []
    positions = []
    colors = {"L1": "#177E89", "L2": "#E07A5F", "L3": "#6A4C93", "L4": "#2A9D8F"}
    for tier, items in MILESTONE_TIERS.items():
        for item in items:
            step = replay["milestones"][item]["first_global_step"]
            labels.append(f"{tier} {item}")
            positions.append(y)
            if step is not None:
                axes[1].scatter(step, y, s=45, color=colors[tier], zorder=3)
                axes[1].hlines(y, 0, step, color=colors[tier], linewidth=1, alpha=0.5)
            y += 1
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlim(0, max(1, replay["transition_count"]))
    axes[1].set_xlabel("Environment step")
    axes[1].set_title("First observed inventory milestone (blank = not observed)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inventory-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    integrity = json.loads((args.run_dir / "integrity_formal.json").read_text())
    completion = json.loads((args.run_dir / ".formal.completed").read_text())
    if not integrity.get("passed") or int(integrity["checkpoint_step"]) != 100000:
        raise ValueError("Formal integrity or exact 100K checkpoint is missing")
    inventory_keys, schema_sha = load_inventory_schema(args.inventory_schema)
    replay = analyze_replay(args.run_dir / "train/replay", inventory_keys)
    if replay["transition_count"] != 100000:
        raise ValueError(f"Expected 100000 replay transitions, got {replay['transition_count']}")
    scores = load_jsonl(args.run_dir / "train/scores.jsonl")
    if not scores or not all(
        math.isfinite(float(row["episode/score"])) for row in scores
    ):
        raise ValueError("Formal scores are empty or non-finite")
    score_replay = compare_scores(scores, replay)
    if not all(
        score_replay[key]
        for key in ("counts_match", "return_sequence_matches", "terminal_step_sequence_matches")
    ):
        raise ValueError(f"Score/replay consistency failed: {score_replay}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    figure = args.output_dir / "training_curve_and_milestones.png"
    plot_training(figure, scores, replay)
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "stage": "formal_training",
        "protocol": {
            "runtime_commit": "5168475b7a4413f9575933b4580e7073caea2114",
            "task": "minecraft_diamond",
            "model": "size50m",
            "agent_seed": 0,
            "world_seed_controlled": False,
            "train_ratio": 32,
            "requested_environment_steps": 100000,
            "paper_scale_comparison_allowed": False,
        },
        "integrity": integrity,
        "completion": completion,
        "inventory_schema": str(args.inventory_schema),
        "inventory_schema_sha256": schema_sha,
        "score_replay_consistency": score_replay,
        "training_scores": describe(
            [float(row["episode/score"]) for row in scores]
        ),
        "replay": replay,
        "resource": {
            "wall_seconds": int(completion["wall_seconds"]),
            "output_bytes": int(completion["output_bytes"]),
            "gpu": parse_gpu_samples(args.run_dir / "resource_formal_gpu.csv"),
        },
        "figure": str(figure),
        "figure_sha256": sha256(figure),
        "interpretation_boundary": (
            "This scaled 100K run tests engineering completeness and early inventory "
            "milestones; it does not reproduce the paper-scale 100M Diamond score."
        ),
    }
    (args.output_dir / "training_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
