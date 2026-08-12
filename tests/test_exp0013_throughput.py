from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cloudpickle
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cleanup_exp0013_temp import process_references  # noqa: E402
from generate_exp0013_configs import render  # noqa: E402
from sample_exp0013_resources import descendants  # noqa: E402
from verify_exp0013_throughput import natural_stop_step, verify  # noqa: E402


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_smoke_s31415_4096_env.yaml"


class Exp0013ThroughputTest(unittest.TestCase):

    def test_driver_quantum_exact_for_all_arms(self) -> None:
        for quantum in (10, 12, 16):
            self.assertEqual(natural_stop_step(5040, quantum), 5040)

    def test_generated_configs_only_change_declared_runtime_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/run"), 4))
        self.assertEqual(generated["run"]["envs"], 4)
        self.assertFalse(generated["run"]["debug"])
        self.assertEqual(generated["run"]["steps"], 5040.0)
        self.assertEqual(generated["run"]["train_ratio"], 32.0)
        self.assertEqual(generated["agent"], source["agent"])
        self.assertEqual(generated["env"], source["env"])

    def test_descendants_recovers_complete_process_tree(self) -> None:
        table = {
            10: {"ppid": 1, "comm": "time", "rss_bytes": 1},
            11: {"ppid": 10, "comm": "python", "rss_bytes": 1},
            12: {"ppid": 11, "comm": "java", "rss_bytes": 1},
            13: {"ppid": 1, "comm": "other", "rss_bytes": 1},
        }
        self.assertEqual(descendants(table, 10), {10, 11, 12})

    def test_cleanup_reference_scan_does_not_match_command_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(process_references(Path(directory).resolve()), [])

    def test_verifier_passes_valid_synthetic_run_and_rejects_temp_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            train = run_dir / "train"
            checkpoint = train / "ckpt/cp"
            checkpoint.mkdir(parents=True)
            (train / "ckpt/latest").write_text("cp", encoding="utf-8")
            (checkpoint / "agent.pkl").write_bytes(b"agent")
            (checkpoint / "step.pkl").write_bytes(cloudpickle.dumps(5040))
            (checkpoint / "done").touch()
            replay = train / "replay"
            replay.mkdir()
            (replay / "chunk.npz").write_bytes(b"replay")
            config = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
            config["logdir"] = str(train)
            config["script"] = "train"
            config["seed"] = 31415
            config["run"].update(
                debug=False,
                envs=2,
                steps=5040.0,
                train_ratio=32.0,
                log_every=10,
                report_every=120,
                save_every=300,
                save_at_end=True,
                from_checkpoint="",
            )
            frozen = run_dir / "frozen.yaml"
            text = yaml.safe_dump(config, sort_keys=False)
            frozen.write_text(text, encoding="utf-8")
            (train / "config.yaml").write_text(text, encoding="utf-8")
            (train / "scores.jsonl").write_text("", encoding="utf-8")
            (train / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "step": 5040,
                        "train/loss/image": 1.0,
                        "fps/policy": 40.0,
                        "replay/replay_ratio": 32.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout = run_dir / "stdout.log"
            stdout.write_text("Start training loop\n", encoding="utf-8")
            temp_root = run_dir / "work/tmp"
            (run_dir / "work/malmo/logs").mkdir(parents=True)
            temp_root.mkdir(parents=True)
            (run_dir / "temp_cleanup_postprocess.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            for index in range(2):
                (run_dir / f"work/malmo/logs/mc_{index}.log").touch()
            resource = run_dir / "resource_system.csv"
            fields = [
                "timestamp_epoch",
                "system_disk_used_bytes",
                "memory_events_oom",
                "memory_events_oom_kill",
                "temp_dir_count",
                "java_count",
            ]
            with resource.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp_epoch": 1,
                            "system_disk_used_bytes": 100,
                            "memory_events_oom": 0,
                            "memory_events_oom_kill": 0,
                            "temp_dir_count": 2,
                            "java_count": 2,
                        },
                        {
                            "timestamp_epoch": 2,
                            "system_disk_used_bytes": 100,
                            "memory_events_oom": 0,
                            "memory_events_oom_kill": 0,
                            "temp_dir_count": 2,
                            "java_count": 2,
                        },
                    ]
                )
            args = Namespace(
                run_dir=run_dir,
                frozen_config=frozen,
                expected_step=5040,
                expected_envs=2,
                driver_step_quantum=10,
                stdout_log=stdout,
                resource_system=resource,
                temp_root=temp_root,
                max_system_disk_loss=536870912,
            )
            self.assertTrue(verify(args)["passed"])
            (temp_root / "leaked-instance").mkdir()
            leaked = verify(args)
            self.assertFalse(leaked["passed"])
            self.assertFalse(leaked["temp_instances_cleaned"])


if __name__ == "__main__":
    unittest.main()
