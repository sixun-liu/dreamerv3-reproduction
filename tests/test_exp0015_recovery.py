from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cloudpickle
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_exp0015_config import FINAL_STEP, render  # noqa: E402
from prepare_exp0015_recovery import prepare, tree_manifest  # noqa: E402
from verify_exp0015_recovery import ALLOWED_CONFIG_CHANGES, verify  # noqa: E402


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml"


def write_checkpoint(train: Path, name: str, step: int, agent: bytes) -> None:
    checkpoint = train / "ckpt" / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "agent.pkl").write_bytes(agent)
    (checkpoint / "step.pkl").write_bytes(cloudpickle.dumps(step))
    (checkpoint / "replay.pkl").write_bytes(cloudpickle.dumps(None))
    (checkpoint / "done").touch()
    (train / "ckpt/latest").write_text(name, encoding="utf-8")


def write_chunk(path: Path, prefix: int, count: int, first: bool = False) -> None:
    stepids = np.zeros((count, 20), dtype=np.uint8)
    stepids[:, 0] = prefix
    stepids[:, -1] = np.arange(count, dtype=np.uint8)
    is_first = np.zeros(count, dtype=bool)
    is_first[0] = first
    np.savez(path, stepid=stepids, is_first=is_first)


class Exp0015RecoveryTest(unittest.TestCase):

    def test_generated_config_changes_only_recovery_runtime_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/recovery")))
        self.assertEqual(generated["seed"], 0)
        self.assertEqual(generated["run"]["envs"], 2)
        self.assertFalse(generated["run"]["debug"])
        self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
        self.assertEqual(generated["run"]["train_ratio"], 32.0)
        self.assertEqual(generated["run"]["from_checkpoint"], "")
        from verify_exp0015_recovery import changed_paths

        self.assertEqual(changed_paths(source, generated), ALLOWED_CONFIG_CHANGES)

    def test_prepare_creates_content_equal_distinct_inodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "replay").mkdir(parents=True)
            write_checkpoint(source, "source", 4, b"agent")
            write_chunk(source / "replay/chunk.npz", 1, 4, first=True)
            result = prepare(source, root / "target", 4)
            self.assertTrue(result["passed"])
            self.assertTrue(result["comparisons"]["checkpoint"]["distinct_inodes"])
            self.assertTrue(result["comparisons"]["replay"]["content_equal"])

    def _fixture(self, root: Path, expected_envs: int = 2) -> Namespace:
        source = root / "source"
        target = root / "target"
        (source / "replay").mkdir(parents=True)
        write_checkpoint(source, "source", 4, b"source-agent")
        write_chunk(source / "replay/source.npz", 1, 4, first=True)
        clone = prepare(source, target / "train", 4)
        clone_manifest = target / "clone.json"
        clone_manifest.write_text(json.dumps(clone), encoding="utf-8")

        final_step = 4 + 2 * expected_envs
        write_checkpoint(target / "train", "final", final_step, b"updated-agent")
        for worker in range(expected_envs):
            write_chunk(
                target / f"train/replay/new-worker-{worker}.npz",
                2 + worker,
                2,
                first=True,
            )

        source_config = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        source_config["logdir"] = str(source)
        source_config["run"]["steps"] = 4.0
        frozen = yaml.safe_load(yaml.safe_dump(source_config))
        frozen["logdir"] = str(target / "train")
        frozen["run"].update(
            debug=False,
            envs=expected_envs,
            steps=float(final_step),
            log_every=10,
            report_every=120,
            save_every=300,
        )
        source_config_path = root / "source_config.yaml"
        frozen_path = root / "frozen.yaml"
        source_config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")
        frozen_path.write_text(yaml.safe_dump(frozen), encoding="utf-8")
        (target / "train/config.yaml").write_text(
            yaml.safe_dump(frozen), encoding="utf-8"
        )
        (target / "train/metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": final_step,
                    "train/loss/image": 1.0,
                    "replay/replay_ratio": 32.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (target / "train/scores.jsonl").write_text("", encoding="utf-8")
        stdout = target / "stdout.log"
        stdout.write_text(
            "Found existing checkpoint.\nLoading checkpoint: cp\n"
            "Loaded checkpoint.\nStart training loop\n",
            encoding="utf-8",
        )
        resource = target / "resource.csv"
        fields = [
            "cgroup_memory_current_bytes",
            "memory_events_oom",
            "memory_events_oom_kill",
            "java_count",
            "temp_dir_count",
            "system_disk_used_bytes",
        ]
        with resource.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for _ in range(2):
                writer.writerow(
                    {
                        "cgroup_memory_current_bytes": 100,
                        "memory_events_oom": 0,
                        "memory_events_oom_kill": 0,
                        "java_count": expected_envs,
                        "temp_dir_count": expected_envs,
                        "system_disk_used_bytes": 100,
                    }
                )
        temp_root = target / "work/tmp"
        temp_root.mkdir(parents=True)
        (target / "temp_cleanup_postprocess.json").write_text(
            json.dumps({"passed": True}), encoding="utf-8"
        )
        return Namespace(
            run_dir=target,
            source_train=source,
            source_config=source_config_path,
            clone_manifest=clone_manifest,
            frozen_config=frozen_path,
            source_step=4,
            final_step=final_step,
            stdout_log=stdout,
            resource_system=resource,
            temp_root=temp_root,
            max_system_disk_loss=536870912,
            skip_live_process_check=True,
            experiment_id="EXP-TEST",
            expected_envs=expected_envs,
        )

    def test_verifier_accepts_replay_forest_and_rejects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            result = verify(args)
            self.assertTrue(result["passed"])
            self.assertEqual(result["replay"]["new_unique_stepids"], 4)
            self.assertEqual(result["replay"]["new_is_first_chunk_prefix_count"], 2)
            with (args.source_train / "replay/source.npz").open("ab") as handle:
                handle.write(b"drift")
            drifted = verify(args)
            self.assertFalse(drifted["passed"])
            self.assertFalse(drifted["source_unchanged"]["replay"])

    def test_verifier_requires_every_declared_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory), expected_envs=4)
            result = verify(args)
            self.assertTrue(result["passed"])
            self.assertEqual(result["replay"]["new_is_first_chunk_prefix_count"], 4)
            args.expected_envs = 8
            insufficient = verify(args)
            self.assertFalse(insufficient["passed"])
            self.assertFalse(insufficient["replay"]["new_worker_starts_observed"])


if __name__ == "__main__":
    unittest.main()
