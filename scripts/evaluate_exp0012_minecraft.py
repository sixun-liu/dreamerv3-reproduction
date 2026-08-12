#!/usr/bin/env python3
"""Evaluate the frozen EXP-0012 checkpoint for three Minecraft episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


MILESTONES = (
    "log",
    "planks",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_minecraft_base(env):
    current = env
    for _ in range(10):
        if hasattr(current, "_inv_keys") and hasattr(current, "_action_names"):
            return current
        current = getattr(current, "env")
    raise RuntimeError("Could not locate MinecraftBase through wrapper chain")


class StreamingVideo:

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(1000)
        )
        self.stream.width = 64
        self.stream.height = 64
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}
        self.frame_count = 0
        self.dynamic_pairs = 0
        self.previous: np.ndarray | None = None
        self.closed = False

    def add(self, image: np.ndarray) -> None:
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected video frame: {image.shape} {image.dtype}")
        if self.previous is not None and np.any(self.previous != image):
            self.dynamic_pairs += 1
        self.previous = image.copy()
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--agent-seed", type=int, default=10000)
    parser.add_argument("--episode-length", type=int, default=36000)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.episodes < 1 or args.episode_length < 1 or args.video_stride < 1:
        raise ValueError("episodes, episode_length, and video_stride must be positive")
    if not (runtime / "dreamerv3/main.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    required = tuple(
        checkpoint_path / name for name in ("agent.pkl", "step.pkl", "done")
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Incomplete directory checkpoint: {checkpoint_path}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(runtime))
    import elements  # noqa: PLC0415
    import embodied  # noqa: PLC0415
    from dreamerv3 import main as dv3_main  # noqa: PLC0415
    import ruamel.yaml as yaml  # noqa: PLC0415

    configs = yaml.YAML(typ="safe").load(
        (runtime / "dreamerv3/configs.yaml").read_text(encoding="utf-8")
    )
    config = elements.Config(configs["defaults"])
    for name in ("minecraft", "size50m"):
        config = config.update(configs[name])
    config = config.update(
        task="minecraft_diamond",
        seed=args.agent_seed,
        logdir=str(output),
    )
    config = config.update({"env.minecraft.length": args.episode_length})
    config.save(output / "config.yaml")

    # Reuse this environment's spaces to avoid constructing a second Malmo instance.
    env = dv3_main.make_env(config, 0)
    base = find_minecraft_base(env)
    inventory_keys = list(base._inv_keys)
    milestone_columns = {
        item: inventory_keys.index(f"inventory/{item}") for item in MILESTONES
    }
    from dreamerv3.agent import Agent  # noqa: PLC0415

    notlog = lambda key: not key.startswith("log/")
    obs_space = {key: value for key, value in env.obs_space.items() if notlog(key)}
    act_space = {key: value for key, value in env.act_space.items() if key != "reset"}
    agent = Agent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(str(checkpoint_path), keys=["agent"])

    driver = embodied.Driver([lambda: env], parallel=False)
    video_path = output / "episode_000_preregistered_stride4.mp4"
    first_frame_path = output / "episode_000_first_frame.png"
    video = StreamingVideo(video_path, args.fps)
    completed: list[dict] = []
    current: dict | None = None
    transitions_seen = 0

    def record_transition(transition, worker):
        nonlocal current, transitions_seen
        assert worker == 0
        transitions_seen += 1
        if len(completed) >= args.episodes:
            return
        if bool(transition["is_first"]):
            if current is not None:
                raise RuntimeError("Evaluation world reset without is_last")
            current = {
                "episode_index": len(completed),
                "return": 0.0,
                "transitions_including_reset": 0,
                "milestone_max": {item: 0.0 for item in MILESTONES},
                "milestone_first_transition": {item: None for item in MILESTONES},
            }
        if current is None:
            raise RuntimeError("Evaluation did not start with is_first")
        reward = float(transition["reward"])
        inventory_max = np.asarray(transition["inventory_max"], dtype=np.float32)
        image = np.asarray(transition["image"], dtype=np.uint8)
        if not math.isfinite(reward) or not np.isfinite(inventory_max).all():
            raise ValueError("Non-finite evaluation transition")
        current["return"] += reward
        current["transitions_including_reset"] += 1
        local_transition = current["transitions_including_reset"]
        for item, column in milestone_columns.items():
            value = float(inventory_max[column])
            current["milestone_max"][item] = max(current["milestone_max"][item], value)
            if value > 0 and current["milestone_first_transition"][item] is None:
                current["milestone_first_transition"][item] = local_transition

        if current["episode_index"] == 0:
            record_frame = (
                local_transition == 1
                or (local_transition - 1) % args.video_stride == 0
                or bool(transition["is_last"])
            )
            if local_transition == 1:
                Image.fromarray(image).save(first_frame_path)
            if record_frame:
                video.add(image)
        if bool(transition["is_last"]):
            current["return"] = float(current["return"])
            current["actions_excluding_reset"] = max(0, local_transition - 1)
            current["terminal"] = bool(transition["is_terminal"])
            current["end_reason"] = (
                "terminal" if bool(transition["is_terminal"]) else "time_limit"
            )
            completed.append(current)
            current = None
            if len(completed) == 1:
                video.close()

    policy = lambda *values: agent.policy(*values, mode="eval")
    driver.on_step(record_transition)
    driver.reset(agent.init_policy)
    transition_budget = args.episodes * (args.episode_length + 1) + 10
    try:
        while len(completed) < args.episodes and transitions_seen < transition_budget:
            driver(policy, steps=min(10, transition_budget - transitions_seen))
    finally:
        video.close()
        driver.close()

    if len(completed) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} episodes within {transition_budget} transitions; "
            f"got {len(completed)}"
        )
    returns = [float(episode["return"]) for episode in completed]
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("Evaluation returns contain non-finite values")
    if video.frame_count < 2 or video.dynamic_pairs == 0:
        raise ValueError("Preregistered evaluation video is empty or static")

    milestone_summary = {}
    for item in MILESTONES:
        successful = [episode for episode in completed if episode["milestone_max"][item] > 0]
        milestone_summary[item] = {
            "successful_episodes": len(successful),
            "first_episode_index": successful[0]["episode_index"] if successful else None,
            "max_inventory_count": max(
                episode["milestone_max"][item] for episode in completed
            ),
        }
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "task": "minecraft_diamond",
        "policy_mode": "eval; categorical policy remains sampled with fixed agent RNG seed",
        "agent_seed": args.agent_seed,
        "world_seed_controlled": False,
        "world_seed_limitation": (
            "Runtime DefaultWorldGenerator(force_reset=True) exposes no world seed."
        ),
        "episode_length": args.episode_length,
        "checkpoint": str(checkpoint_path),
        "checkpoint_agent_sha256": sha256(checkpoint_path / "agent.pkl"),
        "checkpoint_step_sha256": sha256(checkpoint_path / "step.pkl"),
        "inventory_key_count": len(inventory_keys),
        "episodes": completed,
        "episode_count": len(completed),
        "return_mean": statistics.fmean(returns),
        "return_median": statistics.median(returns),
        "return_min": min(returns),
        "return_max": max(returns),
        "return_std_population": statistics.pstdev(returns),
        "milestones": milestone_summary,
        "video_selection": "episode index 0 preregistered; not best-of-N",
        "video_stride": args.video_stride,
        "video_fps": args.fps,
        "video": str(video_path),
        "video_frame_count": video.frame_count,
        "video_dynamic_adjacent_pairs": video.dynamic_pairs,
        "first_frame": str(first_frame_path),
        "comparison_boundary": (
            f"{args.episodes} complete terminal-checkpoint episode(s) with uncontrolled "
            "Minecraft worlds; not sample-equivalent to the training curve or "
            "paper-scale result."
        ),
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
