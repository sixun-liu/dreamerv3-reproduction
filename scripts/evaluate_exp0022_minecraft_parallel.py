#!/usr/bin/env python3
"""Evaluate one frozen Minecraft checkpoint in three parallel worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from fractions import Fraction
from functools import partial
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


def find_runtime_inventory_keys(env) -> tuple[str, ...]:
    """Recover the vector ordering from the environment that created it."""
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        inventory_keys = vars(current).get("_inv_keys")
        if inventory_keys is not None:
            keys = tuple(inventory_keys)
            if not keys or not all(
                isinstance(key, str) and key.startswith("inventory/") for key in keys
            ):
                raise ValueError(f"Invalid runtime inventory keys: {keys!r}")
            if len(set(keys)) != len(keys):
                raise ValueError("Runtime inventory keys contain duplicates")
            return keys
        current = vars(current).get("env")
        if current is None:
            break
    raise ValueError("Could not locate runtime _inv_keys through the wrapper chain")


class RuntimeMilestoneObs:
    """Expose runtime-indexed inventory maxima as log-only observations."""

    def __init__(self, env, elements):
        self.env = env
        self.inventory_keys = find_runtime_inventory_keys(env)
        shape = tuple(env.obs_space["inventory_max"].shape)
        if shape != (len(self.inventory_keys),):
            raise ValueError(
                f"Runtime inventory shape {shape} does not match "
                f"{len(self.inventory_keys)} keys"
            )
        self.milestone_columns = {}
        for item in MILESTONES:
            key = f"inventory/{item}"
            if key not in self.inventory_keys:
                raise ValueError(f"Runtime inventory is missing {key}")
            self.milestone_columns[item] = self.inventory_keys.index(key)
        if len(set(self.milestone_columns.values())) != len(MILESTONES):
            raise ValueError("Runtime milestone indices are not distinct")
        self._obs_space = dict(env.obs_space)
        for item in MILESTONES:
            self._obs_space[f"log/eval_milestone/{item}"] = elements.Space(
                np.float32, (), 0
            )
            self._obs_space[f"log/eval_milestone_index/{item}"] = elements.Space(
                np.int32, (), 0, len(self.inventory_keys)
            )
        self._obs_space["log/eval_inventory_key_count"] = elements.Space(
            np.int32, (), len(self.inventory_keys), len(self.inventory_keys) + 1
        )

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def obs_space(self):
        return self._obs_space

    def step(self, action):
        obs = self.env.step(action)
        inventory_max = np.asarray(obs["inventory_max"], dtype=np.float32)
        if inventory_max.shape != (len(self.inventory_keys),):
            raise ValueError(f"Runtime inventory shape drift: {inventory_max.shape}")
        for item, column in self.milestone_columns.items():
            obs[f"log/eval_milestone/{item}"] = np.float32(inventory_max[column])
            obs[f"log/eval_milestone_index/{item}"] = np.int32(column)
        obs["log/eval_inventory_key_count"] = np.int32(len(self.inventory_keys))
        return obs

    def close(self):
        return self.env.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SingleEpisodeHold:
    """Prevent a completed worker from resetting into an unregistered episode."""

    def __init__(self, env):
        self.env = env
        self._done = False
        self._last = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, action):
        if self._done:
            held = {key: np.asarray(value).copy() for key, value in self._last.items()}
            held["reward"] = np.zeros_like(held["reward"])
            held["is_first"] = np.zeros_like(held["is_first"], dtype=bool)
            held["is_last"] = np.zeros_like(held["is_last"], dtype=bool)
            held["is_terminal"] = np.zeros_like(held["is_terminal"], dtype=bool)
            for key in held:
                if key.startswith("log/"):
                    held[key] = np.zeros_like(held[key])
            return held
        obs = self.env.step(action)
        if bool(obs["is_last"]):
            self._done = True
            self._last = {key: np.asarray(value).copy() for key, value in obs.items()}
        return obs

    def close(self):
        return self.env.close()


def make_single_episode_env(runtime: str, config, worker: int):
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import elements
    from dreamerv3 import main as dv3_main

    env = dv3_main.make_env(config, worker)
    env = RuntimeMilestoneObs(env, elements)
    return SingleEpisodeHold(env)


def driver_obs_space(driver):
    if not driver.parallel:
        return driver.envs[0].obs_space
    driver.pipes[0].send(("obs_space",))
    return driver._receive(driver.pipes[0])


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


def summarize_episodes(completed: list[dict]) -> tuple[list[float], dict]:
    returns = [float(episode["return"]) for episode in completed]
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("Evaluation returns contain non-finite values")
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
    return returns, milestone_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="EXP-0022")
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
    if args.episodes != 3:
        raise ValueError("EXP-0022 is frozen to exactly three workers and episodes")
    if args.episode_length < 1 or args.video_stride < 1:
        raise ValueError("episode_length and video_stride must be positive")
    if not (runtime / "dreamerv3/main.py").is_file():
        raise FileNotFoundError(f"Invalid runtime: {runtime}")
    required = tuple(checkpoint_path / name for name in ("agent.pkl", "step.pkl", "done"))
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Incomplete directory checkpoint: {checkpoint_path}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(runtime))
    import elements
    import embodied
    import ruamel.yaml as yaml
    configs = yaml.YAML(typ="safe").load(
        (runtime / "dreamerv3/configs.yaml").read_text(encoding="utf-8")
    )
    config = elements.Config(configs["defaults"])
    for name in ("minecraft", "size50m"):
        config = config.update(configs[name])
    config = config.update(task="minecraft_diamond", seed=args.agent_seed, logdir=str(output))
    config = config.update({"env.minecraft.length": args.episode_length})
    config.save(output / "config.yaml")

    constructors = [
        partial(make_single_episode_env, str(runtime), config, worker)
        for worker in range(args.episodes)
    ]
    driver = embodied.Driver(constructors, parallel=True)
    obs_space_all = driver_obs_space(driver)
    notlog = lambda key: not key.startswith("log/")
    obs_space = {key: value for key, value in obs_space_all.items() if notlog(key)}
    act_space = {key: value for key, value in driver.act_space.items() if key != "reset"}
    from dreamerv3.agent import Agent

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

    video_path = output / "episode_000_preregistered_stride4.mp4"
    first_frame_path = output / "episode_000_first_frame.png"
    video = StreamingVideo(video_path, args.fps)
    current: dict[int, dict] = {}
    completed_by_worker: dict[int, dict] = {}
    worker_callback_counts = {worker: 0 for worker in range(args.episodes)}
    worker_hold_counts = {worker: 0 for worker in range(args.episodes)}
    synchronized_cycles = 0

    def record_transition(transition, worker):
        worker_callback_counts[worker] += 1
        if worker in completed_by_worker:
            if bool(transition["is_first"]) or bool(transition["is_last"]):
                raise RuntimeError(f"Worker {worker} reset after its registered episode")
            worker_hold_counts[worker] += 1
            return
        if bool(transition["is_first"]):
            if worker in current:
                raise RuntimeError(f"Worker {worker} emitted a second is_first")
            current[worker] = {
                "episode_index": worker,
                "worker": worker,
                "return": 0.0,
                "transitions_including_reset": 0,
                "is_first_count": 0,
                "is_last_count": 0,
                "milestone_max": {item: 0.0 for item in MILESTONES},
                "milestone_first_transition": {item: None for item in MILESTONES},
                "milestone_runtime_indices": None,
                "inventory_key_count": None,
                "reset_milestone_values": None,
            }
        if worker not in current:
            raise RuntimeError(f"Worker {worker} did not start with is_first")
        episode = current[worker]
        reward = float(transition["reward"])
        inventory_max = np.asarray(transition["inventory_max"], dtype=np.float32)
        image = np.asarray(transition["image"], dtype=np.uint8)
        if not math.isfinite(reward) or not np.isfinite(inventory_max).all():
            raise ValueError(f"Non-finite evaluation transition from worker {worker}")
        episode["return"] += reward
        episode["transitions_including_reset"] += 1
        episode["is_first_count"] += int(bool(transition["is_first"]))
        episode["is_last_count"] += int(bool(transition["is_last"]))
        local_transition = episode["transitions_including_reset"]
        runtime_indices = {
            item: int(transition[f"log/eval_milestone_index/{item}"])
            for item in MILESTONES
        }
        inventory_key_count = int(transition["log/eval_inventory_key_count"])
        milestone_values = {
            item: float(transition[f"log/eval_milestone/{item}"])
            for item in MILESTONES
        }
        if not all(math.isfinite(value) for value in milestone_values.values()):
            raise ValueError(f"Non-finite milestone from worker {worker}")
        if local_transition == 1:
            episode["milestone_runtime_indices"] = runtime_indices
            episode["inventory_key_count"] = inventory_key_count
            episode["reset_milestone_values"] = milestone_values
        elif (
            runtime_indices != episode["milestone_runtime_indices"]
            or inventory_key_count != episode["inventory_key_count"]
        ):
            raise RuntimeError(f"Runtime milestone mapping drift from worker {worker}")
        for item, value in milestone_values.items():
            episode["milestone_max"][item] = max(episode["milestone_max"][item], value)
            if value > 0 and episode["milestone_first_transition"][item] is None:
                episode["milestone_first_transition"][item] = local_transition

        if worker == 0:
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
            episode["return"] = float(episode["return"])
            episode["actions_excluding_reset"] = max(0, local_transition - 1)
            episode["terminal"] = bool(transition["is_terminal"])
            episode["end_reason"] = (
                "terminal" if bool(transition["is_terminal"]) else "time_limit"
            )
            completed_by_worker[worker] = episode
            del current[worker]
            if worker == 0:
                video.close()

    policy = lambda *values: agent.policy(*values, mode="eval")
    driver.on_step(record_transition)
    driver.reset(agent.init_policy)
    cycle_budget = args.episode_length + 11
    try:
        while len(completed_by_worker) < args.episodes and synchronized_cycles < cycle_budget:
            driver(policy, steps=args.episodes)
            synchronized_cycles += 1
    finally:
        video.close()
        driver.close()

    if len(completed_by_worker) != args.episodes:
        raise RuntimeError(
            f"Expected one episode from each worker within {cycle_budget} cycles; "
            f"got {sorted(completed_by_worker)}"
        )
    completed = [completed_by_worker[worker] for worker in range(args.episodes)]
    for episode in completed:
        if episode["is_first_count"] != 1 or episode["is_last_count"] != 1:
            raise RuntimeError(f"Episode boundary drift: {episode}")
        if episode["actions_excluding_reset"] > args.episode_length:
            raise RuntimeError(f"Episode length drift: {episode}")
    if video.frame_count < 2 or video.dynamic_pairs == 0:
        raise ValueError("Preregistered worker-0 video is empty or static")
    returns, milestone_summary = summarize_episodes(completed)
    active_transitions = sum(ep["transitions_including_reset"] for ep in completed)
    summary = {
        "schema_version": 2,
        "experiment_id": args.experiment_id,
        "task": "minecraft_diamond",
        "evaluation_scheduler": "three synchronous process environments; one episode per worker",
        "policy_mode": "eval; shared categorical policy with fixed agent RNG seed",
        "agent_seed": args.agent_seed,
        "world_seed_controlled": False,
        "world_seed_limitation": "Runtime DefaultWorldGenerator(force_reset=True) exposes no world seed.",
        "environment_count": args.episodes,
        "episode_length": args.episode_length,
        "checkpoint": str(checkpoint_path),
        "checkpoint_agent_sha256": sha256(checkpoint_path / "agent.pkl"),
        "checkpoint_step_sha256": sha256(checkpoint_path / "step.pkl"),
        "milestone_mapping_source": "runtime _MinecraftBase._inv_keys per worker",
        "episodes": completed,
        "episode_count": len(completed),
        "active_transitions_including_reset": active_transitions,
        "active_actions": sum(ep["actions_excluding_reset"] for ep in completed),
        "synchronized_cycles": synchronized_cycles,
        "worker_callback_counts": worker_callback_counts,
        "worker_hold_counts": worker_hold_counts,
        "extra_episode_count": 0,
        "return_mean": statistics.fmean(returns),
        "return_median": statistics.median(returns),
        "return_min": min(returns),
        "return_max": max(returns),
        "return_std_population": statistics.pstdev(returns),
        "milestones": milestone_summary,
        "video_selection": "worker 0 / episode index 0 preregistered; not best-of-N",
        "video_worker": 0,
        "video_stride": args.video_stride,
        "video_fps": args.fps,
        "video": str(video_path),
        "video_frame_count": video.frame_count,
        "video_dynamic_adjacent_pairs": video.dynamic_pairs,
        "first_frame": str(first_frame_path),
        "comparison_boundary": (
            "Scheduling throughput probe with uncontrolled Minecraft worlds. Scores and "
            "episode lengths are not paired to the serial baseline and do not select the evaluator."
        ),
    }
    (output / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
