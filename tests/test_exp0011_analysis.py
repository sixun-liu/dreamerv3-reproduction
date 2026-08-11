from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0011_atari import compute_smoke_gate, fixed_bins  # noqa: E402
from verify_exp0011_run import resolve_latest_checkpoint, unexpected_nonfinite_metrics  # noqa: E402


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
            smoke_steps=4090,
            formal_steps=100_000,
            output_bytes=100 * 1024**2,
            free_bytes=100 * 1024**3,
        )
        slow = compute_smoke_gate(
            True,
            [1.0, 1.0],
            smoke_steps=4090,
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

    def test_resolve_latest_directory_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory)
            checkpoint = train / "ckpt" / "20260812T043021F558700"
            checkpoint.mkdir(parents=True)
            (train / "ckpt/latest").write_text(checkpoint.name, encoding="utf-8")
            (checkpoint / "agent.pkl").write_bytes(b"agent")
            (checkpoint / "step.pkl").write_bytes(b"step")
            (checkpoint / "done").touch()
            self.assertEqual(resolve_latest_checkpoint(train), checkpoint)

    def test_resolve_latest_rejects_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory)
            (train / "ckpt").mkdir(parents=True)
            (train / "ckpt/latest").write_text("missing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete directory checkpoint"):
                resolve_latest_checkpoint(train)


if __name__ == "__main__":
    unittest.main()
