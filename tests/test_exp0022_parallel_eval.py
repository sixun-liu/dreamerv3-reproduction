from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_exp0022_minecraft_parallel",
    ROOT / "scripts/evaluate_exp0022_minecraft_parallel.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_exp0022_parallel_eval import analyze  # noqa: E402


class FakeEnv:

    def __init__(self):
        self.calls = 0
        self.closed = False

    def step(self, action):
        self.calls += 1
        last = self.calls == 3
        return {
            "image": np.full((2, 2, 3), self.calls, np.uint8),
            "reward": np.float32(self.calls),
            "is_first": np.asarray(self.calls == 1),
            "is_last": np.asarray(last),
            "is_terminal": np.asarray(last),
            "log/item": np.int64(self.calls),
        }

    def close(self):
        self.closed = True


class FakeSpace:

    def __init__(self, dtype, shape=(), low=None, high=None):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape) if isinstance(shape, tuple) else shape
        self.low = low
        self.high = high


class FakeElements:
    Space = FakeSpace


def test_single_episode_hold_never_resets_underlying_env():
    base = FakeEnv()
    env = MODULE.SingleEpisodeHold(base)
    first = env.step({"reset": True})
    env.step({"reset": False})
    last = env.step({"reset": False})
    held1 = env.step({"reset": True})
    held2 = env.step({"reset": False})

    assert bool(first["is_first"])
    assert bool(last["is_last"])
    assert base.calls == 3
    for held in (held1, held2):
        assert not bool(held["is_first"])
        assert not bool(held["is_last"])
        assert not bool(held["is_terminal"])
        assert float(held["reward"]) == 0.0
        assert int(held["log/item"]) == 0
        assert np.array_equal(held["image"], last["image"])

    env.close()
    assert base.closed


def test_find_runtime_inventory_keys_through_wrapper_chain():
    keys = tuple(f"inventory/{item}" for item in MODULE.MILESTONES)

    class Base:
        def __init__(self):
            self._inv_keys = keys

    class Wrapper:
        def __init__(self, env):
            self.env = env

    assert MODULE.find_runtime_inventory_keys(Wrapper(Wrapper(Base()))) == keys


def test_runtime_milestone_obs_uses_environment_order():
    runtime_order = tuple(
        f"inventory/{item}" for item in reversed(MODULE.MILESTONES)
    )

    class Base:
        def __init__(self):
            self._inv_keys = runtime_order
            self.obs_space = {
                "inventory_max": FakeSpace(np.float32, (len(runtime_order),))
            }

        def step(self, action):
            values = np.arange(len(runtime_order), dtype=np.float32)
            return {"inventory_max": values}

    wrapped = MODULE.RuntimeMilestoneObs(Base(), FakeElements)
    obs = wrapped.step({})
    for item in MODULE.MILESTONES:
        expected = runtime_order.index(f"inventory/{item}")
        assert int(obs[f"log/eval_milestone_index/{item}"]) == expected
        assert float(obs[f"log/eval_milestone/{item}"]) == float(expected)


def test_episode_summary_preserves_worker_order_and_milestones():
    episodes = []
    for worker, score in enumerate((1.0, 3.0, 2.0)):
        maxima = {item: 0.0 for item in MODULE.MILESTONES}
        maxima["log"] = float(worker > 0)
        episodes.append(
            {
                "episode_index": worker,
                "return": score,
                "milestone_max": maxima,
            }
        )
    returns, milestones = MODULE.summarize_episodes(episodes)
    assert returns == [1.0, 3.0, 2.0]
    assert milestones["log"]["successful_episodes"] == 2
    assert milestones["log"]["first_episode_index"] == 1


def test_analyzer_requires_integrity_and_speed_gate():
    episodes = []
    for worker, actions in enumerate((100, 80, 60)):
        episodes.append(
            {
                "episode_index": worker,
                "worker": worker,
                "return": float(worker),
                "transitions_including_reset": actions + 1,
                "actions_excluding_reset": actions,
                "is_first_count": 1,
                "is_last_count": 1,
                "milestone_runtime_indices": {
                    item: index for index, item in enumerate(MODULE.MILESTONES)
                },
                "inventory_key_count": len(MODULE.MILESTONES),
                "reset_milestone_values": {
                    item: 0.0 for item in MODULE.MILESTONES
                },
            }
        )
    evaluation = {
        "experiment_id": "EXP-0022",
        "agent_seed": 10000,
        "environment_count": 3,
        "episode_count": 3,
        "episode_length": 128,
        "video_worker": 0,
        "video_stride": 4,
        "video_selection": "worker 0 / episode index 0 preregistered; not best-of-N",
        "milestone_mapping_source": "runtime _MinecraftBase._inv_keys per worker",
        "checkpoint_agent_sha256": "agent",
        "checkpoint_step_sha256": "step",
        "episodes": episodes,
        "extra_episode_count": 0,
        "active_actions": 240,
        "synchronized_cycles": 101,
        "worker_hold_counts": {"0": 0, "1": 20, "2": 40},
        "worker_callback_counts": {"0": 101, "1": 101, "2": 101},
    }
    system = [
        {
            "timestamp_epoch": str(t),
            "cpu_usage_usec": str(t * 4_000_000),
            "cpu_throttled_usec": "0",
            "cgroup_memory_current_bytes": "1000",
            "tree_rss_bytes": "500",
            "java_count": "3",
            "temp_dir_count": "3",
            "system_disk_used_bytes": "100",
            "data_disk_used_bytes": "100",
            "memory_events_oom": "0",
            "memory_events_oom_kill": "0",
        }
        for t in (1, 2)
    ]
    gpu = [{"gpu_util_percent": "1", "memory_used_mib": "100"}]
    kwargs = dict(
        expected_checkpoint_agent_sha256="agent",
        expected_checkpoint_step_sha256="step",
        expected_episode_length=128,
        serial_wall_seconds=300.0,
        serial_active_actions=240,
        minimum_wall_speedup=1.5,
        maximum_memory_bytes=2000,
    )
    promoted = analyze(
        evaluation,
        {"total_wall_seconds": 100},
        system,
        gpu,
        {"passed": True},
        {"passed": True},
        **kwargs,
    )
    assert promoted["integrity"]["passed"]
    assert promoted["parallel_evaluator_promoted"]
    retained = analyze(
        evaluation,
        {"total_wall_seconds": 250},
        system,
        gpu,
        {"passed": True},
        {"passed": True},
        **kwargs,
    )
    assert retained["integrity"]["passed"]
    assert not retained["parallel_evaluator_promoted"]

    evaluation["episodes"][0]["reset_milestone_values"]["cobblestone"] = 365.0
    invalid = analyze(
        evaluation,
        {"total_wall_seconds": 100},
        system,
        gpu,
        {"passed": True},
        {"passed": True},
        **kwargs,
    )
    assert not invalid["integrity"]["reset_milestones_zero"]
    assert not invalid["integrity"]["passed"]
