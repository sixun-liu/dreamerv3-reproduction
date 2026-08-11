from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0011_atari import compute_smoke_gate, fixed_bins  # noqa: E402
from verify_exp0011_run import unexpected_nonfinite_metrics  # noqa: E402


class Exp0011AnalysisTest(unittest.TestCase):

    def test_fixed_bins_use_emulator_frame_axis(self) -> None:
        rows = [
            {"step": 4, "episode/score": 1.0},
            {"step": 10_000, "episode/score": 3.0},
            {"step": 10_004, "episode/score": 5.0},
            {"step": 20_000, "episode/score": 7.0},
        ]
        bins = fixed_bins(rows, 20_000)
        self.assertEqual([item["count"] for item in bins], [2, 2])
        self.assertEqual([item["mean"] for item in bins], [2.0, 6.0])

    def test_smoke_gate_requires_integrity_eta_and_disk(self) -> None:
        passed = compute_smoke_gate(
            True,
            [10.0, 12.0, 14.0],
            smoke_steps=2048,
            formal_steps=100_000,
            output_bytes=100 * 1024**2,
            free_bytes=100 * 1024**3,
        )
        slow = compute_smoke_gate(
            True,
            [1.0, 1.0],
            smoke_steps=2048,
            formal_steps=100_000,
            output_bytes=100 * 1024**2,
            free_bytes=100 * 1024**3,
        )
        self.assertTrue(passed["formal_gate"])
        self.assertFalse(slow["formal_gate"])
        self.assertFalse(slow["eta_at_most_12_hours"])

    def test_conditional_stat_nan_allowed_but_training_loss_nan_rejected(self) -> None:
        rows = [
            {"train/constats/neg_loss": math.nan},
            {"train/loss/dyn": math.nan},
        ]
        unexpected = unexpected_nonfinite_metrics(rows)
        self.assertEqual(len(unexpected), 1)
        self.assertEqual(unexpected[0]["key"], "train/loss/dyn")


if __name__ == "__main__":
    unittest.main()
