from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0014_paired_throughput import compare, summarize_arm  # noqa: E402
from generate_exp0014_config import render  # noqa: E402
from minecraft_temp_cleanup import validate_temp_root  # noqa: E402


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_smoke_s31415_4096_env.yaml"


def metric(step: int, fps: float, ratio: float) -> dict:
    return {"step": step, "fps/policy": fps, "replay/replay_ratio": ratio}


def system_rows() -> list[dict[str, str]]:
    return [
        {
            "timestamp_epoch": "1",
            "cpu_usage_usec": "1000000",
            "cpu_throttled_usec": "0",
            "cgroup_memory_current_bytes": "100",
            "java_count": "1",
            "system_disk_used_bytes": "1000",
            "data_disk_used_bytes": "2000",
        },
        {
            "timestamp_epoch": "11",
            "cpu_usage_usec": "21000000",
            "cpu_throttled_usec": "1000000",
            "cgroup_memory_current_bytes": "200",
            "java_count": "1",
            "system_disk_used_bytes": "1000",
            "data_disk_used_bytes": "3000",
        },
    ]


class Exp0014PairedThroughputTest(unittest.TestCase):

    def test_config_changes_only_declared_runtime_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/run")))
        self.assertEqual(generated["run"]["envs"], 1)
        self.assertFalse(generated["run"]["debug"])
        self.assertEqual(generated["run"]["steps"], 5040.0)
        self.assertEqual(generated["agent"], source["agent"])
        self.assertEqual(generated["env"], source["env"])

    def test_cleanup_rejects_wrong_experiment_or_non_temp_path(self) -> None:
        good = Path(
            "/root/autodl-tmp/Runs/EXP-0014__paired__tag/envs1/work/tmp"
        )
        self.assertEqual(validate_temp_root(good, "EXP-0014"), good)
        with self.assertRaises(ValueError):
            validate_temp_root(good, "EXP-0013")
        with self.assertRaises(ValueError):
            validate_temp_root(good.parent, "EXP-0014")
        with self.assertRaises(ValueError):
            validate_temp_root(Path("/tmp"), "EXP-0014")

    def test_steady_window_excludes_startup_and_shutdown(self) -> None:
        metrics = [
            metric(1000, 1.0, 32.0),
            metric(2100, 20.0, 32.0),
            metric(3000, 30.0, 32.0),
            metric(4200, 40.0, 32.0),
            metric(4300, 99.0, 32.0),
            metric(3500, 99.0, 20.0),
        ]
        gpu = [
            {"gpu_util_percent": "10", "memory_used_mib": "20"},
            {"gpu_util_percent": "30", "memory_used_mib": "40"},
        ]
        result = summarize_arm(
            metrics,
            system_rows(),
            gpu,
            {"arm": "envs1", "training_wall_seconds": 300},
            {"passed": True},
            5040,
        )
        self.assertEqual(result["steady_window"]["sample_count"], 3)
        self.assertEqual(result["steady_window"]["policy_fps_mean"], 30.0)

    def test_comparison_requires_both_preregistered_gates(self) -> None:
        base = {
            "integrity_passed": True,
            "requested_environment_steps": 5040,
            "end_to_end_steps_per_second": 20.0,
            "steady_window": {"sample_count": 4, "policy_fps_mean": 30.0},
        }
        faster = {
            **base,
            "end_to_end_steps_per_second": 22.0,
            "steady_window": {"sample_count": 4, "policy_fps_mean": 31.5},
        }
        self.assertTrue(compare(base, faster)["envs2_preferred"])
        weak_steady = {
            **faster,
            "steady_window": {"sample_count": 4, "policy_fps_mean": 31.4},
        }
        self.assertFalse(compare(base, weak_steady)["envs2_preferred"])
        self.assertEqual(compare(base, weak_steady)["selected_environment_count"], 1)


if __name__ == "__main__":
    unittest.main()
