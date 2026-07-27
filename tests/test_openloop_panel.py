from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openloop_panel import (  # noqa: E402
    ARMS,
    SEEDS,
    build_panel,
    sha256,
    verify_panel,
    write_deterministic_npz,
)


def make_chunk(path: Path, source_index: int, length: int = 196) -> None:
    time = np.arange(length, dtype=np.float32)
    arrays = {
        "action": np.stack([time + source_index] * 6, axis=-1),
        "height": time + 10 * source_index,
        "is_first": np.zeros(length, bool),
        "is_last": np.zeros(length, bool),
        "is_terminal": np.zeros(length, bool),
        "orientations": np.stack([time + source_index] * 14, axis=-1),
        "reward": time / 10 + source_index,
        "stepid": np.tile(np.arange(20, dtype=np.uint8), (length, 1)),
        "velocity": np.stack([time - source_index] * 9, axis=-1),
        "dyn/deter": np.ones((length, 4), np.float32),
        "dyn/stoch": np.ones((length, 2, 2), np.float32),
    }
    arrays["is_first"][70] = True
    arrays["is_last"][69] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class OpenloopPanelTest(unittest.TestCase):

    def test_deterministic_npz_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {
                "z": np.arange(7, dtype=np.int32),
                "a": np.linspace(-1, 1, 9, dtype=np.float32),
            }
            first, second = root / "first.npz", root / "second.npz"
            write_deterministic_npz(first, arrays)
            write_deterministic_npz(second, arrays)
            self.assertEqual(sha256(first), sha256(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_balanced_reproducible_panel_excludes_cached_latents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            source_index = 0
            for arm in ARMS:
                for seed in SEEDS:
                    replay = matrix / arm / f"s{seed:03d}" / "train" / "replay"
                    make_chunk(replay / "chunk-1024.npz", source_index)
                    source_index += 1

            first = root / "first.npz"
            second = root / "second.npz"
            first_manifest = root / "first.json"
            second_manifest = root / "second.json"
            kwargs = dict(
                context=4, horizon=3, windows_per_source=5, seed=20260727
            )
            build_panel(matrix, first, first_manifest, **kwargs)
            build_panel(matrix, second, second_manifest, **kwargs)

            self.assertEqual(sha256(first), sha256(second))
            status = verify_panel(
                first, first_manifest, verify_sources=True, verify_hashes=True
            )
            self.assertEqual(status["window_count"], 30)
            with np.load(first, allow_pickle=False) as panel:
                self.assertNotIn("dyn/deter", panel.files)
                self.assertNotIn("dyn/stoch", panel.files)
                self.assertEqual(panel["orientations"].shape, (30, 7, 14))
                self.assertEqual(
                    Counter(panel["source_index"].tolist()),
                    Counter({index: 5 for index in range(6)}),
                )
                boundary = (
                    panel["is_first"] | panel["is_last"] | panel["is_terminal"]
                )
                self.assertFalse(boundary.any())


if __name__ == "__main__":
    unittest.main()
