from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0012_minecraft import MILESTONES  # noqa: E402
from analyze_exp0023_increment import (  # noqa: E402
    analyze_added_replay,
    learning_verdict,
    validate_inventory_mapping,
)
from analyze_exp0023_parallel_eval import analyze as analyze_parallel  # noqa: E402
from analyze_exp0023_parallel_eval import MILESTONES as EVAL_MILESTONES  # noqa: E402
from generate_exp0023_config import FINAL_STEP, render  # noqa: E402
from verify_exp0015_recovery import (  # noqa: E402
    ALLOWED_CONFIG_CHANGES,
    REQUIRED_RECOVERY_CONFIG_CHANGES,
    changed_paths,
)


BASELINE = ROOT / "docs/reproduction/configs/exp0020_minecraft_s000_200k_env.yaml"


def write_chunk(path: Path, prefix: int, values: list[float], keys: list[str]) -> None:
    count = len(values)
    stepids = np.zeros((count, 20), dtype=np.uint8)
    stepids[:, 0] = prefix
    stepids[:, -1] = np.arange(1, count + 1)
    inventory = np.zeros((count, len(keys)), dtype=np.float32)
    inventory[:, keys.index("inventory/wooden_pickaxe")] = values
    np.savez_compressed(path, stepid=stepids, inventory_max=inventory)


class Exp0023IncrementTest(unittest.TestCase):

    def test_offline_and_runtime_milestones_match(self) -> None:
        self.assertEqual(MILESTONES, EVAL_MILESTONES)

    def test_config_changes_only_declared_recovery_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/exp0023")))
        self.assertEqual(generated["run"]["envs"], 4)
        self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
        self.assertEqual(generated["run"]["train_ratio"], 32.0)
        differences = changed_paths(source, generated)
        self.assertLessEqual(differences, ALLOWED_CONFIG_CHANGES)
        self.assertLessEqual({"logdir", "run.steps"}, differences)

    def test_added_replay_isolated_by_stepid_and_tracks_milestone(self) -> None:
        keys = [f"inventory/{item}" for item in MILESTONES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            write_chunk(source / "source.npz", 1, [0, 0], keys)
            write_chunk(target / "source.npz", 1, [0, 0], keys)
            write_chunk(target / "new.npz", 2, [0, 1, 1], keys)
            result = analyze_added_replay(source, target, keys)
            self.assertEqual(result["source_unique_stepids"], 2)
            self.assertEqual(result["target_unique_stepids"], 5)
            self.assertEqual(result["new_unique_stepids"], 3)
            wooden = result["milestones"]["wooden_pickaxe"]
            self.assertTrue(wooden["observed"])
            self.assertEqual(wooden["successful_trajectory_count"], 1)
            self.assertEqual(wooden["first_added_replay_position"], 2)

    def test_learning_verdict_requires_wooden_maintenance_and_cobblestone(self) -> None:
        added = {
            "milestones": {
                "wooden_pickaxe": {"observed": True},
                "cobblestone": {"observed": True},
            }
        }
        evaluation = {
            "milestones": {
                "cobblestone": {"successful_episodes": 0},
            }
        }
        result = learning_verdict(added, evaluation)
        self.assertTrue(result["hypothesis_supported"])
        added["milestones"]["wooden_pickaxe"]["observed"] = False
        self.assertFalse(learning_verdict(added, evaluation)["hypothesis_supported"])

    def test_offline_inventory_schema_must_match_all_runtime_workers(self) -> None:
        keys = [f"inventory/{item}" for item in MILESTONES]
        mapping = {item: index for index, item in enumerate(MILESTONES)}
        evaluation = {
            "episodes": [
                {"milestone_runtime_indices": dict(mapping)} for _ in range(3)
            ]
        }
        self.assertEqual(validate_inventory_mapping(keys, evaluation), mapping)
        evaluation["episodes"][2]["milestone_runtime_indices"]["cobblestone"] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_inventory_mapping(keys, evaluation)

    def test_parallel_evaluation_requires_exp0023_protocol(self) -> None:
        mapping = {item: index for index, item in enumerate(MILESTONES)}
        episodes = [
            {
                "episode_index": worker,
                "worker": worker,
                "return": float(worker),
                "transitions_including_reset": actions + 1,
                "actions_excluding_reset": actions,
                "is_first_count": 1,
                "is_last_count": 1,
                "milestone_runtime_indices": mapping,
                "inventory_key_count": len(MILESTONES),
                "reset_milestone_values": {item: 0.0 for item in MILESTONES},
            }
            for worker, actions in enumerate((100, 80, 60))
        ]
        evaluation = {
            "experiment_id": "EXP-0023",
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
        kwargs = dict(
            expected_experiment_id="EXP-0023",
            expected_checkpoint_agent_sha256="agent",
            expected_checkpoint_step_sha256="step",
            expected_episode_length=128,
            serial_wall_seconds=300.0,
            serial_active_actions=240,
            minimum_wall_speedup=1.5,
            maximum_memory_bytes=2000,
        )
        result = analyze_parallel(
            evaluation,
            {"total_wall_seconds": 100},
            system,
            [{"gpu_util_percent": "1", "memory_used_mib": "100"}],
            {"passed": True},
            {"passed": True},
            **kwargs,
        )
        self.assertTrue(result["integrity"]["passed"])
        self.assertEqual(result["experiment_id"], "EXP-0023")
        evaluation["experiment_id"] = "EXP-0022"
        result = analyze_parallel(
            evaluation,
            {"total_wall_seconds": 100},
            system,
            [{"gpu_util_percent": "1", "memory_used_mib": "100"}],
            {"passed": True},
            {"passed": True},
            **kwargs,
        )
        self.assertFalse(result["integrity"]["protocol_ok"])


if __name__ == "__main__":
    unittest.main()
