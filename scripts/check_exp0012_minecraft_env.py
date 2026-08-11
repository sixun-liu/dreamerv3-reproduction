#!/usr/bin/env python3
"""Validate the EXP-0012 Minecraft environment without constructing an agent."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


REQUIRED_OBSERVATIONS = {
    "image",
    "inventory",
    "inventory_max",
    "equipped",
    "reward",
    "health",
    "hunger",
    "breath",
    "is_first",
    "is_last",
    "is_terminal",
}


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


def find_minecraft_base(env):
    current = env
    for _ in range(8):
        if hasattr(current, "_action_names") and hasattr(current, "_inv_keys"):
            return current
        current = getattr(current, "env")
    raise RuntimeError("Could not locate MinecraftBase through wrapper chain")


def validate_observation(obs: dict, *, expect_first: bool) -> np.ndarray:
    missing = sorted(REQUIRED_OBSERVATIONS - set(obs))
    if missing:
        raise ValueError(f"Missing observations: {missing}")
    if bool(obs["is_first"]) != expect_first:
        raise ValueError(f"Unexpected is_first={obs['is_first']}")
    image = np.asarray(obs["image"])
    if image.shape != (64, 64, 3) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected image: {image.shape} {image.dtype}")
    for key in ("inventory", "inventory_max", "equipped"):
        value = np.asarray(obs[key])
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {key}")
    for key in ("reward", "health", "hunger", "breath"):
        if not np.isfinite(float(obs[key])):
            raise ValueError(f"Non-finite {key}")
    return image.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-steps", type=int, default=32)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not (runtime / "embodied/envs/minecraft_flat.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(runtime))
    from embodied.envs.minecraft import Minecraft  # noqa: PLC0415

    env = None
    try:
        # The short length is an L0-only intervention to exercise last/reset.
        env = Minecraft(
            "diamond",
            repeat=1,
            size=(64, 64),
            break_speed=100.0,
            length=args.episode_steps,
            logs=False,
        )
        base = find_minecraft_base(env)
        action_names = tuple(base._action_names)
        action_count = int(env.act_space["action"].high)
        if action_count != len(action_names) or action_count < 10:
            raise ValueError((action_count, action_names))

        first = env.step({"reset": True, "action": 0})
        frames = [validate_observation(first, expect_first=True)]
        observations = [first]
        desired = (
            "noop",
            "turn_left",
            "forward",
            "jump",
            "attack",
            "turn_right",
        )
        sequence = [action_names.index(name) for name in desired]
        terminal_step = None
        for step in range(1, args.episode_steps + 2):
            obs = env.step(
                {"reset": False, "action": sequence[(step - 1) % len(sequence)]}
            )
            frames.append(validate_observation(obs, expect_first=False))
            observations.append(obs)
            if bool(obs["is_last"]):
                terminal_step = step
                break
        if terminal_step != args.episode_steps:
            raise RuntimeError(
                f"Expected time-limit at {args.episode_steps}, got {terminal_step}"
            )

        reset = env.step({"reset": True, "action": 0})
        reset_frame = validate_observation(reset, expect_first=True)
        frames.append(reset_frame)
        adjacent_dynamic = sum(
            bool(np.any(left != right)) for left, right in zip(frames, frames[1:])
        )
        if max(float(frame.std()) for frame in frames) <= 1.0:
            raise ValueError("Minecraft frames appear blank")
        if adjacent_dynamic == 0:
            raise ValueError("Minecraft video contains no changing adjacent frames")

        inventory = np.stack(
            [np.asarray(obs["inventory"], np.float32) for obs in observations]
        )
        inventory_max = np.stack(
            [np.asarray(obs["inventory_max"], np.float32) for obs in observations]
        )
        if np.any(inventory_max + 1e-6 < inventory):
            raise ValueError("inventory_max is below current inventory")

        video = output / "minecraft_l0_episode.mp4"
        write_mp4(video, frames, 15.0)
        Image.fromarray(frames[0]).save(output / "minecraft_l0_first_frame.png")
        Image.fromarray(frames[-1]).save(output / "minecraft_l0_reset_frame.png")
        java = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, check=True
        )
        rewards = np.asarray([float(obs["reward"]) for obs in observations])
        summary = {
            "schema_version": 1,
            "experiment_id": "EXP-0012",
            "purpose": "Minecraft L0; episode length is test-only",
            "runtime": str(runtime),
            "python": sys.version,
            "java": (java.stderr or java.stdout).splitlines()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "minerl-mirror",
                    "gym",
                    "jax",
                    "jaxlib",
                    "numpy",
                    "opencv-python",
                    "elements",
                )
            },
            "protocol": {
                "task": "diamond",
                "repeat": 1,
                "size": [64, 64],
                "break_speed": 100.0,
                "episode_steps_test_only": args.episode_steps,
            },
            "action_count": action_count,
            "action_names": list(action_names),
            "inventory_keys": list(base._inv_keys),
            "observation_keys": sorted(first),
            "steps": len(observations) - 1,
            "reward_sum": float(rewards.sum()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "inventory_nonzero_count": int(np.count_nonzero(inventory)),
            "inventory_max_nonzero_count": int(np.count_nonzero(inventory_max)),
            "frame_shape": list(frames[0].shape),
            "first_frame_std": float(frames[0].std()),
            "max_frame_std": max(float(frame.std()) for frame in frames),
            "adjacent_dynamic_pairs": adjacent_dynamic,
            "terminal_step": terminal_step,
            "terminal_flag": bool(observations[-1]["is_terminal"]),
            "post_terminal_reset_first": bool(reset["is_first"]),
            "video": str(video),
            "passed": True,
        }
        (output / "minecraft_l0.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
