from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0010_dmcvision import (  # noqa: E402
    compute_continuation_gate,
    fixed_bins,
)


class Exp0010AnalysisTest(unittest.TestCase):

    def test_fixed_bins_are_left_open_right_closed(self) -> None:
        rows = [
            {"step": 1, "episode/score": 10.0},
            {"step": 10_000, "episode/score": 30.0},
            {"step": 10_001, "episode/score": 50.0},
            {"step": 20_000, "episode/score": 70.0},
        ]
        bins = fixed_bins(rows, 20_000)
        self.assertEqual([row["count"] for row in bins], [2, 2])
        self.assertEqual([row["mean"] for row in bins], [20.0, 60.0])

    def test_continuation_gate_requires_every_resource_and_score_gate(self) -> None:
        passed = compute_continuation_gate(
            early_mean=50,
            tail_mean=250,
            integrity_passed=True,
            wall_seconds=1800,
            output_bytes=1024**3,
            free_bytes=30 * 1024**3,
        )
        failed = compute_continuation_gate(
            early_mean=180,
            tail_mean=250,
            integrity_passed=True,
            wall_seconds=1800,
            output_bytes=1024**3,
            free_bytes=30 * 1024**3,
        )
        self.assertTrue(passed["continuation_gate"])
        self.assertFalse(failed["continuation_gate"])
        self.assertFalse(failed["improvement_at_least_100"])


if __name__ == "__main__":
    unittest.main()
