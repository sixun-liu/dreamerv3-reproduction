#!/usr/bin/env python3
"""Evaluate one EXP-0010 checkpoint and record a preregistered episode."""

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
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
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
    parser.add_argument("--max-decisions-per-episode", type=int, default=620)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not (runtime / "dreamerv3/main.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(runtime))
    import embodied  # noqa: PLC0415
    from dreamerv3 import agent as agt  # noqa: PLC0415
    from dreamerv3 import main as dv3_main  # noqa: PLC0415

    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in ("dmc_vision", "size12m"):
        config = config.update(agt.Agent.configs[name])
    config = config.update(
        task="dmc_walker_walk",
        seed=args.agent_seed,
        logdir=str(output),
        tensorboard=False,
    )
    config.save(output / "config.yaml")

    agent = dv3_main.make_agent(config)
    checkpoint = embodied.Checkpoint()
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
        image_key = "image" if "image" in transition else "log_image"
        current_frames.append(np.asarray(transition[image_key], dtype=np.uint8).copy())
        current_rewards.append(float(transition["reward"]))
        if bool(transition["is_last"]):
            completed.append(
                {
                    "episode_index": len(completed),
                    "return": float(np.sum(current_rewards)),
                    "decisions": len(current_rewards),
                    "frames": current_frames,
                }
            )

    make_env = lambda: dv3_main.make_env(config, 0)
    driver = embodied.Driver([make_env], parallel=False)
    driver.on_step(record_transition)
    policy = lambda *values: agent.policy(*values, mode="eval")
    driver.reset(agent.init_policy)
    decision_budget = args.episodes * args.max_decisions_per_episode
    while len(completed) < args.episodes and transitions_seen < decision_budget:
        driver(policy, steps=min(10, decision_budget - transitions_seen))

    if len(completed) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} complete episodes within {decision_budget} decisions; "
            f"got {len(completed)}"
        )
    returns = [item["return"] for item in completed]
    if not np.isfinite(returns).all():
        raise ValueError("Evaluation returns contain non-finite values")
    first_frames = completed[0].pop("frames")
    for item in completed[1:]:
        item.pop("frames")
    if len(first_frames) < 2 or not all(frame.shape == first_frames[0].shape for frame in first_frames):
        raise ValueError("First evaluation episode has invalid frames")

    video_path = output / "episode_000_preregistered.mp4"
    write_mp4(video_path, first_frames, args.fps)
    Image.fromarray(first_frames[0]).save(output / "episode_000_first_frame.png")
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0010",
        "task": "dmc_walker_walk",
        "observation": "64x64 RGB image",
        "policy_mode": "eval (continuous action distribution remains sampled)",
        "agent_seed": args.agent_seed,
        "environment_seed_controlled": False,
        "checkpoint": str(checkpoint_path),
        "episodes": completed,
        "episode_count": len(completed),
        "return_mean": statistics.fmean(returns),
        "return_median": statistics.median(returns),
        "return_min": min(returns),
        "return_max": max(returns),
        "return_std_population": statistics.pstdev(returns),
        "video_selection": "episode index 0, preregistered; not best-of-N",
        "video": str(video_path),
        "frame_shape": list(first_frames[0].shape),
        "pixel_std_first_frame": float(first_frames[0].std()),
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
