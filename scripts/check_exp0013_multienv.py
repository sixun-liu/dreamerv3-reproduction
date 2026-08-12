#!/usr/bin/env python3
"""Exercise multiple real Minecraft environments through the frozen Driver."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image


def make_env(runtime: str, episode_steps: int):
    import sys

    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    from embodied.envs.minecraft import Minecraft

    return Minecraft(
        "diamond",
        repeat=1,
        size=(64, 64),
        break_speed=100.0,
        length=episode_steps,
        logs=False,
    )


def save_contact_sheet(path: Path, images: list[np.ndarray]) -> None:
    height, width = images[0].shape[:2]
    sheet = np.zeros((height, width * len(images), 3), dtype=np.uint8)
    for index, image in enumerate(images):
        sheet[:, index * width : (index + 1) * width] = image
    Image.fromarray(sheet).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--steps-per-env", type=int, default=4)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not (runtime / "embodied/core/driver.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    output.mkdir(parents=True)

    import sys

    sys.path.insert(0, str(runtime))
    import embodied

    transitions: dict[int, list[dict]] = {index: [] for index in range(args.envs)}
    constructors = [
        partial(make_env, str(runtime), max(32, args.steps_per_env + 4))
        for _ in range(args.envs)
    ]
    driver = embodied.Driver(constructors, parallel=True)
    action_count = int(driver.act_space["action"].high)

    def collect(transition: dict, worker: int) -> None:
        image = np.asarray(transition["image"])
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError((worker, image.shape, image.dtype))
        for key in ("reward", "health", "hunger", "breath"):
            if not np.isfinite(float(transition[key])):
                raise ValueError((worker, key, transition[key]))
        transitions[worker].append(
            {
                "is_first": bool(transition["is_first"]),
                "is_last": bool(transition["is_last"]),
                "reward": float(transition["reward"]),
                "image": image.copy(),
            }
        )

    driver.on_step(collect)

    def policy(carry, observation):
        batch = len(observation["is_first"])
        actions = np.arange(batch, dtype=np.int32) % action_count
        return carry, {"action": actions}, {}

    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        driver.reset()
        driver(policy, steps=args.envs * args.steps_per_env)
        temp_dirs_during = sorted(
            str(path.resolve()) for path in temp_root.iterdir() if path.is_dir()
        )
        first_images = [transitions[index][0]["image"] for index in range(args.envs)]
        save_contact_sheet(output / "minecraft_multienv_first_frames.png", first_images)
        pairwise_equal = [
            bool(np.array_equal(first_images[left], first_images[right]))
            for left in range(args.envs)
            for right in range(left + 1, args.envs)
        ]
        summary = {
            "schema_version": 1,
            "experiment_id": "EXP-0013",
            "purpose": "multi-environment reset/step and data-disk temp L0",
            "runtime": str(runtime),
            "envs": args.envs,
            "steps_per_env": args.steps_per_env,
            "action_count": action_count,
            "temp_root": str(temp_root),
            "temp_root_on_data_disk": str(temp_root).startswith("/root/autodl-tmp/"),
            "temp_dirs_during": temp_dirs_during,
            "worker_transition_counts": {
                str(index): len(rows) for index, rows in transitions.items()
            },
            "worker_first_flags": {
                str(index): rows[0]["is_first"] for index, rows in transitions.items()
            },
            "pairwise_first_frame_equal": pairwise_equal,
            "all_workers_observed": all(transitions.values()),
        }
        summary["passed"] = bool(
            summary["temp_root_on_data_disk"]
            and summary["all_workers_observed"]
            and all(summary["worker_first_flags"].values())
            and len(temp_dirs_during) >= args.envs
        )
        (output / "minecraft_multienv_l0.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
        if not summary["passed"]:
            raise SystemExit(1)
    finally:
        driver.close()
        cleanup_started = time.monotonic()
        while time.monotonic() - cleanup_started < 30:
            remaining = sorted(
                str(path.resolve()) for path in temp_root.iterdir() if path.is_dir()
            )
            if not remaining:
                break
            time.sleep(1)
        cleanup = {
            "wait_seconds": time.monotonic() - cleanup_started,
            "remaining": remaining,
            "passed": not remaining,
        }
        (output / "temp_dirs_after_close.json").write_text(
            json.dumps(cleanup, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
