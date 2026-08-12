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
from analyze_exp0017_increment import analyze_added_replay, learning_verdict  # noqa: E402
from generate_exp0017_config import FINAL_STEP, render  # noqa: E402
from verify_exp0015_recovery import (  # noqa: E402
    ALLOWED_CONFIG_CHANGES,
    REQUIRED_RECOVERY_CONFIG_CHANGES,
    changed_paths,
)


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml"


def write_chunk(path: Path, prefix: int, values: list[float], keys: list[str]) -> None:
    count = len(values)
    stepids = np.zeros((count, 20), dtype=np.uint8)
    stepids[:, 0] = prefix
    stepids[:, -1] = np.arange(1, count + 1)
    inventory = np.zeros((count, len(keys)), dtype=np.float32)
    inventory[:, keys.index("inventory/wooden_pickaxe")] = values
    np.savez_compressed(path, stepid=stepids, inventory_max=inventory)


class Exp0017IncrementTest(unittest.TestCase):

    def test_config_changes_only_declared_recovery_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/exp0017")))
        self.assertEqual(generated["run"]["envs"], 4)
        self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
        self.assertEqual(generated["run"]["train_ratio"], 32.0)
        differences = changed_paths(source, generated)
        self.assertLessEqual(differences, ALLOWED_CONFIG_CHANGES)
        self.assertLessEqual(REQUIRED_RECOVERY_CONFIG_CHANGES, differences)

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

    def test_learning_verdict_requires_l1_and_advance(self) -> None:
        added = {"milestones": {"wooden_pickaxe": {"observed": True}}}
        evaluation = {
            "milestones": {
                "crafting_table": {"successful_episodes": 2},
                "wooden_pickaxe": {"successful_episodes": 0},
            }
        }
        result = learning_verdict(added, evaluation)
        self.assertTrue(result["hypothesis_supported"])
        evaluation["milestones"]["crafting_table"]["successful_episodes"] = 1
        self.assertFalse(learning_verdict(added, evaluation)["hypothesis_supported"])


if __name__ == "__main__":
    unittest.main()
