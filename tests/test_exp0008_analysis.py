from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0008_cheetah import aggregate_curve, describe, t95_interval  # noqa: E402


class Exp0008AnalysisTest(unittest.TestCase):

    def test_aggregate_curve_uses_right_closed_10k_bins(self) -> None:
        rows = [
            {"step": 1, "episode/score": 10.0},
            {"step": 10_000, "episode/score": 30.0},
            {"step": 10_001, "episode/score": 50.0},
            {"step": 20_000, "episode/score": 70.0},
        ]
        curve = aggregate_curve(rows, np.asarray([10_000, 20_000], dtype=float))
        self.assertEqual([row["count"] for row in curve], [2, 2])
        self.assertEqual([row["median"] for row in curve], [20.0, 60.0])

    def test_describe_population_std(self) -> None:
        summary = describe([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertAlmostEqual(summary["std_population"], np.sqrt(2.0 / 3.0))

    def test_t_interval_is_symmetric_for_five_seeds(self) -> None:
        interval = t95_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual((interval[0] + interval[1]) / 2, 3.0)
        self.assertLess(interval[0], 1.1)
        self.assertGreater(interval[1], 4.9)


if __name__ == "__main__":
    unittest.main()
