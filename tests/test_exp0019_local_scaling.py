from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0016_recovery_scaling import compare  # noqa: E402
from generate_exp0019_configs import (  # noqa: E402
    ADDED_STEPS,
    ENVIRONMENT_COUNTS,
    FINAL_STEP,
    render,
)
from verify_exp0015_recovery import ALLOWED_CONFIG_CHANGES, changed_paths  # noqa: E402


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml"


def arm(envs: int, fps: float, eta: float) -> dict:
    return {
        "arm": f"envs{envs}",
        "environment_count": envs,
        "integrity_passed": True,
        "added_environment_steps": ADDED_STEPS,
        "predicted_added_100k_seconds": eta,
        "steady_window": {"sample_count": 8, "policy_fps_mean": fps},
        "resource": {
            "max_cgroup_memory_bytes": 100,
            "max_java_count": envs,
            "max_temp_dir_count": envs,
        },
    }


class Exp0019LocalScalingTest(unittest.TestCase):

    def test_configs_are_reachable_and_change_only_runtime_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        for envs in ENVIRONMENT_COUNTS:
            self.assertEqual(ADDED_STEPS % envs, 0)
            generated = yaml.safe_load(render(source, Path("/data/run"), envs))
            self.assertEqual(generated["run"]["envs"], envs)
            self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
            self.assertEqual(generated["run"]["train_ratio"], 32.0)
            self.assertEqual(changed_paths(source, generated), ALLOWED_CONFIG_CHANGES)

    def test_envs5_requires_both_five_percent_gates(self) -> None:
        incumbent = arm(4, 60.0, 2000.0)
        candidate = arm(5, 63.1, 1890.0)
        self.assertTrue(compare(incumbent, candidate, 1.05, 0.05, 5, 1000)["candidate_promoted"])
        self.assertFalse(compare(incumbent, arm(5, 62.9, 1890.0), 1.05, 0.05, 5, 1000)["candidate_promoted"])
        self.assertFalse(compare(incumbent, arm(5, 63.1, 1910.0), 1.05, 0.05, 5, 1000)["candidate_promoted"])


if __name__ == "__main__":
    unittest.main()
