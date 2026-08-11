#!/usr/bin/env python3
"""Validate the frozen EXP-0011 ALE protocol without constructing an agent."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


ROM_MD5 = "f34f08e5eb96e500e851a80be3277a56"
ROM_SHA256 = "376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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
    parser.add_argument("--rom-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--max-decisions", type=int, default=160)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    rom = (args.rom_dir / "breakout.bin").resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not (runtime / "embodied/envs/atari.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    if not rom.is_file():
        raise FileNotFoundError(rom)
    rom_md5 = digest(rom, "md5")
    rom_sha256 = digest(rom, "sha256")
    if rom_md5 != ROM_MD5 or rom_sha256 != ROM_SHA256:
        raise ValueError("Breakout ROM fingerprint mismatch")

    output.mkdir(parents=True)
    sys.path.insert(0, str(runtime))
    from embodied.envs.atari import Atari  # noqa: PLC0415

    # The short length is an L0-only intervention to exercise last/terminal/reset.
    env = Atari(
        "breakout",
        repeat=4,
        size=(64, 64),
        gray=False,
        noops=30,
        lives="unused",
        sticky=False,
        actions="needed",
        length=256,
        resize="pillow",
        autostart=False,
        clip_reward=False,
        seed=args.seed,
    )
    first = env.step({"reset": True, "action": 0})
    if not bool(first["is_first"]):
        raise ValueError("Reset did not produce is_first")
    if first["image"].shape != (64, 64, 3) or first["image"].dtype != np.uint8:
        raise ValueError("Unexpected Atari image shape or dtype")

    frames = [np.asarray(first["image"], dtype=np.uint8).copy()]
    rewards = [float(first["reward"])]
    events = [{"decision": 0, "is_first": True, "is_last": False}]
    completed = False
    action_count = int(env.act_space["action"].high)
    for decision in range(1, args.max_decisions + 1):
        obs = env.step({"reset": False, "action": decision % action_count})
        frames.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        rewards.append(float(obs["reward"]))
        if bool(obs["is_last"]):
            events.append(
                {
                    "decision": decision,
                    "is_first": bool(obs["is_first"]),
                    "is_last": True,
                    "is_terminal": bool(obs["is_terminal"]),
                }
            )
            completed = True
            break
    if not completed:
        raise RuntimeError("L0 time-limit episode did not terminate")
    reset = env.step({"reset": True, "action": 0})
    if not bool(reset["is_first"]):
        raise ValueError("Post-terminal reset did not produce is_first")
    if not np.isfinite(rewards).all():
        raise ValueError("Non-finite Atari reward")
    if max(float(frame.std()) for frame in frames) <= 1.0:
        raise ValueError("Atari frames appear blank")

    video = output / "ale_l0_episode.mp4"
    write_mp4(video, frames, 15.0)
    Image.fromarray(frames[0]).save(output / "ale_l0_first_frame.png")
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0011",
        "purpose": "ALE environment L0; length=256 is test-only",
        "runtime": str(runtime),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("ale-py", "jax", "jaxlib", "numpy", "elements")
        },
        "rom": str(rom),
        "rom_md5": rom_md5,
        "rom_sha256": rom_sha256,
        "seed": args.seed,
        "protocol": {
            "task": "breakout",
            "repeat": 4,
            "size": [64, 64],
            "gray": False,
            "sticky": False,
            "actions": "needed",
            "noops_uniform_inclusive": [0, 30],
            "clip_reward": False,
            "autostart": False,
        },
        "action_count": action_count,
        "minimal_action_set": [int(value) for value in env.actionset],
        "decisions": len(frames) - 1,
        "reward_sum": float(np.sum(rewards)),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "frame_shape": list(frames[0].shape),
        "first_frame_std": float(frames[0].std()),
        "max_frame_std": max(float(frame.std()) for frame in frames),
        "terminal_exercised": completed,
        "post_terminal_reset_first": bool(reset["is_first"]),
        "events": events,
        "video": str(video),
        "passed": True,
    }
    (output / "ale_l0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
