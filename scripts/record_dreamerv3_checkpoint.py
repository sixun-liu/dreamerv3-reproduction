#!/usr/bin/env python3
"""Record one complete DreamerV3 checkpoint episode without TensorFlow."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--use-env-seed", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=620)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


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
    args = parse_args()
    args.runtime = args.runtime.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not (args.runtime / "dreamerv3/main.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {args.runtime}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    sys.path.insert(0, str(args.runtime))
    import embodied  # noqa: PLC0415
    from dreamerv3 import agent as agt  # noqa: PLC0415
    from dreamerv3 import main as dv3_main  # noqa: PLC0415

    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in ("dmc_proprio", "size12m"):
        config = config.update(agt.Agent.configs[name])
    config = config.update(
        task=args.task,
        seed=args.seed,
        logdir=str(args.output),
        tensorboard=False,
    )
    config = config.update({"env.dmc.use_seed": args.use_env_seed})
    config.save(args.output / "config.yaml")

    agent = dv3_main.make_agent(config)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(str(args.checkpoint), keys=["agent"])

    frames: list[np.ndarray] = []
    rewards: list[float] = []
    completed = False

    def record_transition(transition, worker):
        nonlocal completed, frames, rewards
        assert worker == 0
        if completed:
            return
        if bool(transition["is_first"]):
            frames = []
            rewards = []
        frames.append(np.asarray(transition["log_image"], dtype=np.uint8).copy())
        rewards.append(float(transition["reward"]))
        if bool(transition["is_last"]):
            completed = True

    make_env = lambda: dv3_main.make_env(config, 0)
    driver = embodied.Driver([make_env], parallel=False)
    driver.on_step(record_transition)
    policy = lambda *values: agent.policy(*values, mode="eval")
    driver.reset(agent.init_policy)
    decisions = 0
    while not completed and decisions < args.max_decisions:
        before = len(frames)
        driver(policy, steps=min(10, args.max_decisions - decisions))
        decisions += max(0, len(frames) - before)

    if not completed:
        raise RuntimeError(
            f"No complete episode within {args.max_decisions} decisions; got {len(frames)} frames"
        )
    if len(frames) < 2 or not all(frame.shape == frames[0].shape for frame in frames):
        raise ValueError("Recorded frames are empty or have inconsistent shapes")
    if not np.isfinite(rewards).all():
        raise ValueError("Recorded rewards contain non-finite values")

    write_mp4(args.output / "policy.mp4", frames, args.fps)
    episode = {
        "schema_version": 1,
        "task": args.task,
        "agent_seed": args.seed,
        "environment_seed_controlled": args.use_env_seed,
        "checkpoint": str(args.checkpoint),
        "frames": len(frames),
        "fps": args.fps,
        "duration_seconds": len(frames) / args.fps,
        "episode_return": float(np.sum(rewards)),
        "reward_count": len(rewards),
        "frame_shape": list(frames[0].shape),
        "pixel_std_middle_frame": float(np.asarray(frames[len(frames) // 2]).std()),
    }
    (args.output / "episode.json").write_text(
        json.dumps(episode, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(episode, ensure_ascii=True))


if __name__ == "__main__":
    main()
