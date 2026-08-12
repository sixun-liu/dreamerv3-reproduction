from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0012_minecraft import (  # noqa: E402
    MILESTONES,
    analyze_replay,
    compare_scores,
)
from analyze_exp0012_smoke import compute_smoke_gate  # noqa: E402
from verify_exp0012_run import natural_stop_step  # noqa: E402


class Exp0012AnalysisTest(unittest.TestCase):

    def test_driver_quantum_rounds_smoke_to_natural_boundary(self) -> None:
        self.assertEqual(natural_stop_step(4096, 10), 4100)
        self.assertEqual(natural_stop_step(100000, 10), 100000)

    def test_driver_quantum_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            natural_stop_step(4096, 0)
        with self.assertRaises(ValueError):
            natural_stop_step(-1, 10)

    def test_smoke_gate_requires_integrity_resources_time_and_disk(self) -> None:
        result = compute_smoke_gate(
            {"passed": True},
            memory=[24000.0, 25000.0],
            utilization=[50.0, 70.0],
            wall_seconds=342,
            output_bytes=600 * 1024**2,
            disk_free_bytes=80 * 1024**3,
        )
        self.assertTrue(result["formal_gate"])
        self.assertEqual(result["peak_vram_mib"], 25000.0)

    def test_replay_analysis_recovers_complete_and_partial_milestones(self) -> None:
        keys = [f"inventory/{item}" for item in MILESTONES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = np.zeros((6, len(keys)), np.float32)
            inventory[1:, keys.index("inventory/log")] = 1
            inventory[4:, keys.index("inventory/planks")] = 4
            stepids = np.zeros((6, 20), np.uint8)
            stepids[:, -1] = np.arange(1, 7)
            common = {
                "inventory_max": inventory,
                "reward": np.asarray([0, 1, 0, 0, 1, 0], np.float32),
                "is_first": np.asarray([1, 0, 0, 1, 0, 0], bool),
                "is_last": np.asarray([0, 0, 1, 0, 0, 0], bool),
                "stepid": stepids,
            }
            np.savez_compressed(
                root / "20260101T000000F000000-aaaa-bbbb-3.npz",
                **{key: value[:3] for key, value in common.items()},
            )
            np.savez_compressed(
                root / "20260101T000001F000000-bbbb-cccc-3.npz",
                **{key: value[3:] for key, value in common.items()},
            )
            replay = analyze_replay(root, keys)
            self.assertEqual(replay["transition_count"], 6)
            self.assertEqual(replay["complete_episode_count"], 1)
            self.assertEqual(replay["partial_segment_count"], 1)
            self.assertEqual(replay["milestones"]["log"]["first_global_step"], 2)
            self.assertEqual(replay["milestones"]["planks"]["first_global_step"], 5)
            comparison = compare_scores(
                [{"step": 3, "episode/score": 1.0}], replay
            )
            self.assertTrue(comparison["counts_match"])
            self.assertTrue(comparison["return_sequence_matches"])
            self.assertTrue(comparison["terminal_step_sequence_matches"])


if __name__ == "__main__":
    unittest.main()
