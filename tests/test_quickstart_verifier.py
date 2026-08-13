from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cloudpickle
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_quickstart import natural_stop_step, verify  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_args(run_dir: Path, task: str, expected_step: int, quantum: int = 1):
    return argparse.Namespace(
        task=task,
        run_dir=run_dir,
        expected_step=expected_step,
        driver_step_quantum=quantum,
        stdout_log=run_dir / "train_stdout.log",
        output=run_dir / "integrity.json",
    )


def test_natural_stop_step() -> None:
    assert natural_stop_step(4096, 10) == 4100
    assert natural_stop_step(100000, 10) == 100000


def test_verify_legacy_dmc_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "dmc"
    train = run_dir / "train"
    train.mkdir(parents=True)
    config = {
        "task": "dmc_walker_walk",
        "env": {"dmc": {"image": True, "repeat": 2}},
        "dyn": {"rssm": {"deter": 2048}},
        "run": {"train_ratio": 512},
    }
    (train / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (train / "checkpoint.ckpt").write_bytes(cloudpickle.dumps({"step": 8192}))
    (train / "replay").mkdir()
    (train / "replay" / "chunk.npz").write_bytes(b"replay")
    write_jsonl(train / "metrics.jsonl", [{"train/dyn_loss": 1.0}])
    write_jsonl(train / "scores.jsonl", [{"step": 10.0, "score": 1.0}])
    (run_dir / "train_stdout.log").write_text("completed\n", encoding="utf-8")

    result = verify(make_args(run_dir, "dmc-vision", 8192))

    assert result["passed"] is True
    assert result["checkpoint_step"] == 8192


def test_verify_directory_minecraft_checkpoint_with_driver_quantum(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "minecraft"
    train = run_dir / "train"
    checkpoint = train / "ckpt" / "20260813T000000"
    checkpoint.mkdir(parents=True)
    config = {
        "task": "minecraft_diamond",
        "env": {"minecraft": {"size": [64, 64], "length": 36000}},
        "agent": {"dyn": {"rssm": {"deter": 4096}}},
        "run": {"train_ratio": 32},
    }
    (train / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (train / "ckpt" / "latest").write_text(checkpoint.name, encoding="utf-8")
    (checkpoint / "agent.pkl").write_bytes(cloudpickle.dumps({"agent": True}))
    (checkpoint / "step.pkl").write_bytes(cloudpickle.dumps(4100))
    (checkpoint / "done").write_text("", encoding="utf-8")
    (train / "replay").mkdir()
    (train / "replay" / "chunk.npz").write_bytes(b"replay")
    write_jsonl(train / "metrics.jsonl", [{"train/loss/dyn": 1.0}])
    write_jsonl(train / "scores.jsonl", [])
    (run_dir / "train_stdout.log").write_text("completed\n", encoding="utf-8")

    result = verify(make_args(run_dir, "minecraft", 4096, quantum=10))

    assert result["passed"] is True
    assert result["expected_checkpoint_step"] == 4100


def test_verify_directory_breakout_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "breakout"
    train = run_dir / "train"
    checkpoint = train / "ckpt" / "20260813T000000"
    checkpoint.mkdir(parents=True)
    config = {
        "task": "atari100k_breakout",
        "env": {
            "atari100k": {
                "size": [64, 64],
                "gray": False,
                "repeat": 4,
                "clip_reward": False,
            }
        },
        "agent": {"dyn": {"rssm": {"deter": 4096}}},
        "run": {"train_ratio": 256},
    }
    (train / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (train / "ckpt" / "latest").write_text(checkpoint.name, encoding="utf-8")
    (checkpoint / "agent.pkl").write_bytes(cloudpickle.dumps({"agent": True}))
    (checkpoint / "step.pkl").write_bytes(cloudpickle.dumps(4090))
    (checkpoint / "done").write_text("", encoding="utf-8")
    (train / "replay").mkdir()
    (train / "replay" / "chunk.npz").write_bytes(b"replay")
    write_jsonl(train / "metrics.jsonl", [{"train/loss/dyn": 1.0}])
    write_jsonl(train / "scores.jsonl", [{"step": 4090.0, "score": 2.0}])
    (run_dir / "train_stdout.log").write_text("completed\n", encoding="utf-8")

    result = verify(make_args(run_dir, "breakout", 4090))

    assert result["passed"] is True
    assert all(result["config_checks"].values())
