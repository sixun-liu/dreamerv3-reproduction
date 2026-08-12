from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

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


if __name__ == "__main__":
    unittest.main()
