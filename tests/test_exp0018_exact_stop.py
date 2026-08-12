from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_exp0018_config import (  # noqa: E402
    ENVIRONMENT_COUNT,
    FINAL_STEP,
    SOURCE_STEP,
    render,
)
from verify_exp0015_recovery import (  # noqa: E402
    ALLOWED_CONFIG_CHANGES,
    REQUIRED_RECOVERY_CONFIG_CHANGES,
    changed_paths,
)


BASELINE = ROOT / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml"


class Exp0018ExactStopTest(unittest.TestCase):

    def test_config_is_reachable_and_only_changes_recovery_fields(self) -> None:
        source = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        generated = yaml.safe_load(render(source, Path("/data/exp0018")))
        self.assertEqual((FINAL_STEP - SOURCE_STEP) % ENVIRONMENT_COUNT, 0)
        self.assertEqual(generated["run"]["envs"], ENVIRONMENT_COUNT)
        self.assertEqual(generated["run"]["steps"], float(FINAL_STEP))
        differences = changed_paths(source, generated)
        self.assertLessEqual(differences, ALLOWED_CONFIG_CHANGES)
        self.assertLessEqual(REQUIRED_RECOVERY_CONFIG_CHANGES, differences)

    def test_old_fixed_driver_request_would_overshoot(self) -> None:
        current = SOURCE_STEP
        while current < FINAL_STEP:
            current += 12  # ceil(10 / 4) * 4 synchronous transitions.
        self.assertEqual(current, 100_048)


if __name__ == "__main__":
    unittest.main()
