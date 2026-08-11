from __future__ import annotations

import math
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_exp0007_openloop import (  # noqa: E402
    analyze,
    reduce_by_window,
    summarize_errors,
)
from openloop_panel import (  # noqa: E402
    ARMS,
    SEEDS,
    build_panel,
    sha256,
    write_deterministic_npz,
)
from test_openloop_panel import make_chunk  # noqa: E402


class Exp0007AnalysisTest(unittest.TestCase):

    def test_reduce_draws_inside_each_window(self) -> None:
        ids = np.array([3, 3, 7, 7])
        unique, values = reduce_by_window(np.array([1.0, 3.0, 10.0, 14.0]), ids)
        np.testing.assert_array_equal(unique, [3, 7])
        np.testing.assert_allclose(values, [2.0, 12.0])

    def test_known_constant_normalized_error(self) -> None:
        # Two latent draws for each of two replay windows. A unit normalized
        # error must produce pooled NRMSE=1 after within-window aggregation.
        ids = np.array([0, 0, 1, 1])
        summary = summarize_errors(
            norm_sq=np.ones(4),
            raw_sq=np.full(4, 4.0),
            raw_abs=np.full(4, 2.0),
            decoder_loss=np.array([2.0, 4.0, 6.0, 8.0]),
            window_ids=ids,
        )
        self.assertEqual(summary["n_windows"], 2)
        self.assertEqual(summary["n_rows"], 4)
        self.assertAlmostEqual(summary["nrmse"], 1.0)
        self.assertAlmostEqual(summary["rmse"], 2.0)
        self.assertAlmostEqual(summary["mae"], 2.0)
        self.assertAlmostEqual(summary["decoder_loss_mean"], 5.0)

    def test_worst_window_uses_draw_averaged_squared_error(self) -> None:
        ids = np.array([4, 4, 9, 9])
        summary = summarize_errors(
            norm_sq=np.array([1.0, 9.0, 16.0, 16.0]),
            raw_sq=np.array([1.0, 9.0, 16.0, 16.0]),
            raw_abs=np.array([1.0, 3.0, 4.0, 4.0]),
            decoder_loss=np.zeros(4),
            window_ids=ids,
        )
        self.assertEqual(summary["worst_window_index"], 9)
        self.assertAlmostEqual(summary["max_window_nrmse"], 4.0)
        self.assertAlmostEqual(summary["nrmse"], math.sqrt(10.5))

    def test_synthetic_end_to_end_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            source_index = 0
            for arm in ARMS:
                for seed in SEEDS:
                    replay = matrix / arm / f"s{seed:03d}" / "train" / "replay"
                    make_chunk(replay / "chunk-1024.npz", source_index)
                    source_index += 1
            panel_path, manifest_path = root / "panel.npz", root / "panel.json"
            manifest = build_panel(
                matrix,
                panel_path,
                manifest_path,
                context=4,
                horizon=15,
                windows_per_source=2,
                seed=13,
            )
            with np.load(panel_path, allow_pickle=False) as panel_file:
                panel = {key: np.array(panel_file[key], copy=True) for key in panel_file.files}
            count, draws, horizon = manifest["window_count"], 2, 15
            window_ids = np.repeat(np.arange(count, dtype=np.int32), draws)
            draw_ids = np.tile(np.arange(draws, dtype=np.int16), count)
            results = root / "results"
            offsets = {"baseline": 0.2, "e1": 0.3, "p4": 0.6}
            kls = {"baseline": 2.0, "e1": 1.0, "p4": 0.1}
            for arm in ARMS:
                for seed in SEEDS:
                    output_dir = results / arm / f"s{seed:03d}"
                    output_dir.mkdir(parents=True)
                    arrays = {
                        "window_index": window_ids,
                        "draw_index": draw_ids,
                    }
                    for key in ("orientations", "height", "velocity"):
                        target = panel[key][window_ids, 4:4 + horizon]
                        arrays[f"prior_pred__{key}"] = target + offsets[arm]
                        arrays[f"posterior_pred__{key}"] = target + offsets[arm] / 2
                        shape = (len(window_ids), horizon)
                        arrays[f"prior_decoder_loss__{key}"] = np.full(shape, offsets[arm])
                        arrays[f"posterior_decoder_loss__{key}"] = np.full(shape, offsets[arm] / 2)
                    reward = panel["reward"][window_ids, 4:4 + horizon]
                    arrays["prior_reward_pred"] = reward + offsets[arm]
                    arrays["posterior_reward_pred"] = reward + offsets[arm] / 2
                    arrays["prior_reward_loss"] = np.full(reward.shape, offsets[arm])
                    arrays["posterior_reward_loss"] = np.full(reward.shape, offsets[arm] / 2)
                    arrays["posterior_prior_kl"] = np.full(reward.shape, kls[arm])
                    arrays["prior_entropy"] = np.full(reward.shape, 3.0)
                    arrays["posterior_entropy"] = np.full(reward.shape, 2.0)
                    output = output_dir / "predictions.npz"
                    write_deterministic_npz(output, arrays)
                    metadata = {
                        "panel_sha256": manifest["panel_sha256"],
                        "output_sha256": sha256(output),
                        "checkpoint_step": 250000,
                        "draws_per_window": draws,
                        "runtime_commit": "synthetic-runtime",
                        "runtime_tracked_status": "",
                        "finite": True,
                        "checkpoint_agent_sha256": f"synthetic-{arm}-{seed}",
                        "row_count": len(window_ids),
                    }
                    (output_dir / "metadata.json").write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )

            artifacts, review = root / "artifacts", root / "review"
            summary = analyze(argparse.Namespace(
                panel=panel_path,
                manifest=manifest_path,
                results_root=results,
                artifacts=artifacts,
                review=review,
            ))
            gates = summary["primary"]["preregistered_gates"]
            self.assertTrue(gates["p4_prior_nrmse_worse_than_e1_both_seeds"])
            self.assertTrue(gates["p4_teacher_nrmse_worse_than_e1_both_seeds"])
            self.assertTrue(gates["p4_kl_lower_than_e1_both_seeds"])
            self.assertTrue((artifacts / "summary.json").is_file())
            self.assertTrue((review / "openloop_prediction_diagnostics.png").is_file())


if __name__ == "__main__":
    unittest.main()
