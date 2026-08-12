from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0016_recovery_scaling import compare  # noqa: E402
from generate_exp0016_configs import FINAL_STEP, render  # noqa: E402
from verify_exp0015_recovery import ALLOWED_CONFIG_CHANGES, changed_paths  # noqa: E402


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml"


def arm(envs: int, fps: float, eta: float, memory: int = 100) -> dict:
    return {
        "arm": f"envs{envs}",
        "environment_count": envs,
        "integrity_passed": True,
        "added_environment_steps": 5040,
        "predicted_added_100k_seconds": eta,
        "steady_window": {"sample_count": 9, "policy_fps_mean": fps},
        "resource": {
            "max_cgroup_memory_bytes": memory,
            "max_java_count": envs,
            "max_temp_dir_count": envs,
        },
    }


class Exp0016RecoveryScalingTest(unittest.TestCase):

    def test_configs_only_change_declared_runtime_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        for envs in (4, 8):
            generated = yaml.safe_load(render(source, Path("/data/run"), envs))
            self.assertEqual(generated["seed"], 0)
            self.assertEqual(generated["run"]["envs"], envs)
            self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
            self.assertEqual(generated["run"]["train_ratio"], 32.0)
            self.assertEqual(changed_paths(source, generated), ALLOWED_CONFIG_CHANGES)

    def test_promotion_requires_speed_eta_and_resource_gates(self) -> None:
        incumbent = arm(2, 40.0, 2600.0)
        candidate = arm(4, 45.0, 2300.0)
        promoted = compare(incumbent, candidate, 1.10, 0.08, 5, 1000)
        self.assertTrue(promoted["candidate_promoted"])
        weak_eta = compare(incumbent, arm(4, 45.0, 2500.0), 1.10, 0.08, 5, 1000)
        self.assertFalse(weak_eta["candidate_promoted"])
        unsafe = compare(
            incumbent, arm(4, 45.0, 2300.0, memory=1001), 1.10, 0.08, 5, 1000
        )
        self.assertFalse(unsafe["candidate_promoted"])
        self.assertFalse(unsafe["resource_safe"])

    def test_invalid_sample_count_is_inconclusive(self) -> None:
        incumbent = arm(2, 40.0, 2600.0)
        candidate = arm(4, 50.0, 2200.0)
        candidate["steady_window"]["sample_count"] = 4
        result = compare(incumbent, candidate, 1.10, 0.08, 5, 1000)
        self.assertFalse(result["comparison_valid"])
        self.assertEqual(result["verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
