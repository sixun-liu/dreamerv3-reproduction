#!/usr/bin/env python3
"""Evaluate the frozen EXP-0011 terminal checkpoint on seeded ALE episodes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(1000)
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agent-seed", type=int, default=10000)
    parser.add_argument("--environment-seed", type=int, default=20260812)
    parser.add_argument("--max-decisions-per-episode", type=int, default=27010)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not (runtime / "dreamerv3/main.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    required_checkpoint_files = (
        checkpoint_path / "agent.pkl",
        checkpoint_path / "step.pkl",
        checkpoint_path / "done",
    )
    if not all(path.is_file() for path in required_checkpoint_files):
        raise FileNotFoundError(f"Incomplete directory checkpoint: {checkpoint_path}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(runtime))
    import elements  # noqa: PLC0415
    from dreamerv3 import agent as agt  # noqa: PLC0415
    from dreamerv3 import main as dv3_main  # noqa: PLC0415

    config = elements.Config(agt.Agent.configs["defaults"])
    for name in ("atari100k", "size50m"):
        config = config.update(agt.Agent.configs[name])
    config = config.update(
        task="atari100k_breakout",
        seed=args.agent_seed,
        logdir=str(output),
    )
    config.save(output / "config.yaml")

    agent = dv3_main.make_agent(config)
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(str(checkpoint_path), keys=["agent"])

    completed: list[dict] = []
    current_frames: list[np.ndarray] = []
    current_rewards: list[float] = []
    transitions_seen = 0

    def record_transition(transition, worker):
        nonlocal current_frames, current_rewards, transitions_seen
        assert worker == 0
        transitions_seen += 1
        if len(completed) >= args.episodes:
            return
        if bool(transition["is_first"]):
            current_frames = []
            current_rewards = []
        current_frames.append(np.asarray(transition["image"], dtype=np.uint8).copy())
        current_rewards.append(float(transition["reward"]))
        if bool(transition["is_last"]):
            completed.append(
                {
                    "episode_index": len(completed),
                    "return": float(np.sum(current_rewards)),
                    "decisions_including_reset": len(current_rewards),
                    "emulator_frames_upper_bound": max(0, len(current_rewards) - 1) * 4,
                    "frames": current_frames,
                }
            )

    make_env = lambda: dv3_main.make_env(config, 0, seed=args.environment_seed)
    import embodied  # noqa: PLC0415

    driver = embodied.Driver([make_env], parallel=False)
    driver.on_step(record_transition)
    policy = lambda *values: agent.policy(*values, mode="eval")
    driver.reset(agent.init_policy)
    decision_budget = args.episodes * args.max_decisions_per_episode
    while len(completed) < args.episodes and transitions_seen < decision_budget:
        driver(policy, steps=min(100, decision_budget - transitions_seen))

    if len(completed) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} episodes within {decision_budget} decisions; "
            f"got {len(completed)}"
        )
    returns = [item["return"] for item in completed]
    if not np.isfinite(returns).all():
        raise ValueError("Evaluation returns contain non-finite values")
    first_frames = completed[0].pop("frames")
    for item in completed[1:]:
        item.pop("frames")
    if len(first_frames) < 2 or not all(
        frame.shape == (64, 64, 3) for frame in first_frames
    ):
        raise ValueError("First evaluation episode has invalid frames")

    video = output / "episode_000_preregistered.mp4"
    write_mp4(video, first_frames, args.fps)
    Image.fromarray(first_frames[0]).save(output / "episode_000_first_frame.png")
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0011",
        "task": "atari100k_breakout",
        "observation": "64x64 RGB image",
        "policy_mode": "eval (categorical action remains sampled)",
        "agent_seed": args.agent_seed,
        "environment_seed": args.environment_seed,
        "environment_seed_controlled": True,
        "protocol": {
            "sticky": False,
            "minimal_actions": True,
            "repeat": 4,
            "random_noops": [0, 30],
            "raw_reward": True,
        },
        "checkpoint": str(checkpoint_path),
        "episodes": completed,
        "episode_count": len(completed),
        "return_mean": statistics.fmean(returns),
        "return_median": statistics.median(returns),
        "return_min": min(returns),
        "return_max": max(returns),
        "return_std_population": statistics.pstdev(returns),
        "video_selection": "episode index 0, preregistered; not best-of-N",
        "video": str(video),
        "video_frame_count": len(first_frames),
        "frame_shape": list(first_frames[0].shape),
        "pixel_std_first_frame": float(first_frames[0].std()),
        "comparison_boundary": "Independent seeded evaluation; not directly comparable to the public training-curve export.",
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
