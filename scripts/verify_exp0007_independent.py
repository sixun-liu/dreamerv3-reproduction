#!/usr/bin/env python3
"""Independently recompute the decisive EXP-0007 horizon-15 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("baseline", "e1", "p4")
SEEDS = (0, 1)
OBS_KEYS = ("orientations", "height", "velocity")
OBS_DIMS = {"orientations": 14, "height": 1, "velocity": 9}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {path}:{key}")
    return arrays


def target_scales(panel: dict[str, np.ndarray], context: int, horizon: int) -> dict:
    scales = {}
    for key in (*OBS_KEYS, "reward"):
        target = np.asarray(panel[key][:, context:context + horizon], np.float64)
        width = 1 if target.ndim == 2 else target.shape[-1]
        std = np.std(target.reshape(-1, width), axis=0)
        floor = max(float(np.max(std)) * 1e-6, 1e-8)
        scales[key] = np.maximum(std, floor)
    return scales


def source_arms(panel: dict[str, np.ndarray], manifest: dict) -> np.ndarray:
    mapping = {
        int(source["source_index"]): source["arm"]
        for source in manifest["sources"]
    }
    return np.asarray([mapping[int(index)] for index in panel["source_index"]])


def window_mean(values: np.ndarray, windows: np.ndarray, count: int) -> np.ndarray:
    sums = np.bincount(windows, weights=values, minlength=count)
    rows = np.bincount(windows, minlength=count)
    if np.any(rows == 0):
        raise ValueError("At least one panel window has no latent draw")
    return sums / rows


def all_proprio_window_sq(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    scales: dict,
    mode: str,
    context: int,
    horizon_index: int,
) -> np.ndarray:
    windows = arrays["window_index"].astype(np.int64)
    per_dimension = []
    for key in OBS_KEYS:
        pred = np.asarray(arrays[f"{mode}_pred__{key}"], np.float64)
        target = np.asarray(
            panel[key][windows, context:context + pred.shape[1]], np.float64
        )
        pred = pred.reshape(pred.shape[0], pred.shape[1], OBS_DIMS[key])
        target = target.reshape(target.shape[0], target.shape[1], OBS_DIMS[key])
        scale = scales[key].reshape(1, 1, -1)
        per_dimension.append(np.square((pred - target) / scale))
    row_sq = np.concatenate(per_dimension, axis=-1).mean(axis=-1)
    return window_mean(
        row_sq[:, horizon_index], windows, len(panel["window_index"])
    )


def key_window_sq(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    scales: dict,
    mode: str,
    key: str,
    context: int,
    horizon_index: int,
) -> np.ndarray:
    windows = arrays["window_index"].astype(np.int64)
    pred = np.asarray(arrays[f"{mode}_pred__{key}"], np.float64)
    target = np.asarray(
        panel[key][windows, context:context + pred.shape[1]], np.float64
    )
    pred = pred.reshape(pred.shape[0], pred.shape[1], OBS_DIMS[key])
    target = target.reshape(target.shape[0], target.shape[1], OBS_DIMS[key])
    row_sq = np.square(
        (pred - target) / scales[key].reshape(1, 1, -1)
    ).mean(axis=-1)
    return window_mean(
        row_sq[:, horizon_index], windows, len(panel["window_index"])
    )


def reward_window_sq(
    arrays: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    scales: dict,
    mode: str,
    context: int,
    horizon_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = arrays["window_index"].astype(np.int64)
    pred = np.asarray(arrays[f"{mode}_reward_pred"], np.float64)
    target = np.asarray(
        panel["reward"][windows, context:context + pred.shape[1]], np.float64
    )
    raw_sq = np.square(pred[:, horizon_index] - target[:, horizon_index])
    raw = window_mean(raw_sq, windows, len(panel["window_index"]))
    norm = raw / float(scales["reward"][0] ** 2)
    return norm, raw


def latent_window_mean(
    arrays: dict[str, np.ndarray], key: str, horizon_index: int, count: int
) -> np.ndarray:
    windows = arrays["window_index"].astype(np.int64)
    values = np.asarray(arrays[key], np.float64)[:, horizon_index]
    return window_mean(values, windows, count)


def aggregate(window_sq: np.ndarray, mask: np.ndarray) -> dict:
    values = np.sqrt(window_sq[mask])
    return {
        "n_windows": int(mask.sum()),
        "nrmse": float(np.sqrt(np.mean(window_sq[mask]))),
        "median_window_nrmse": float(np.median(values)),
        "p95_window_nrmse": float(np.quantile(values, 0.95)),
        "max_window_nrmse": float(np.max(values)),
        "worst_window_index": int(np.flatnonzero(mask)[int(np.argmax(values))]),
    }


def csv_lookup(path: Path) -> dict[tuple, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["model_arm"],
            int(row["model_seed"]),
            row["data_arm"],
            row["prediction_mode"],
            row["target_key"],
            int(row["horizon"]),
        ): row
        for row in rows
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if sha256(args.panel) != manifest["panel_sha256"]:
        raise ValueError("Panel SHA256 does not match its manifest")
    panel = load_npz(args.panel)
    if any(key.startswith("dyn/") for key in panel):
        raise ValueError("Panel contains forbidden cached latent state")
    scales = target_scales(panel, args.context, args.horizon)
    data_arms = source_arms(panel, manifest)
    count = len(panel["window_index"])
    horizon_index = args.horizon - 1
    existing_summary = read_json(args.summary)
    summary_primary = {
        (row["arm"], int(row["seed"])): row
        for row in existing_summary["primary"]["per_seed"]
    }
    existing_aggregate = csv_lookup(args.aggregate)

    rows: list[dict] = []
    window_cache: dict[tuple[str, int, str], np.ndarray] = {}
    kl_cache: dict[tuple[str, int], np.ndarray] = {}
    mismatches: list[dict] = []
    output_hashes = []
    comparison_count = 0

    for arm in ARMS:
        for seed in SEEDS:
            directory = args.run_root / arm / f"s{seed:03d}"
            metadata = read_json(directory / "metadata.json")
            prediction_path = directory / "predictions.npz"
            prediction_sha = sha256(prediction_path)
            if prediction_sha != metadata["output_sha256"]:
                raise ValueError(f"Output SHA256 mismatch: {prediction_path}")
            if metadata["panel_sha256"] != manifest["panel_sha256"]:
                raise ValueError(f"Panel mismatch: {prediction_path}")
            arrays = load_npz(prediction_path)
            if len(arrays["window_index"]) != count * metadata["draws_per_window"]:
                raise ValueError(f"Unexpected row count: {prediction_path}")
            output_hashes.append({"arm": arm, "seed": seed, "sha256": prediction_sha})

            kl = latent_window_mean(
                arrays, "posterior_prior_kl", horizon_index, count
            )
            kl_cache[(arm, seed)] = kl
            for mode in ("prior", "posterior"):
                all_sq = all_proprio_window_sq(
                    arrays, panel, scales, mode, args.context, horizon_index
                )
                window_cache[(arm, seed, mode)] = all_sq
                key_sq = {
                    key: key_window_sq(
                        arrays,
                        panel,
                        scales,
                        mode,
                        key,
                        args.context,
                        horizon_index,
                    )
                    for key in OBS_KEYS
                }
                reward_norm_sq, reward_raw_sq = reward_window_sq(
                    arrays, panel, scales, mode, args.context, horizon_index
                )
                for data_arm in ("all", *ARMS):
                    mask = (
                        np.ones(count, dtype=bool)
                        if data_arm == "all"
                        else data_arms == data_arm
                    )
                    stats = aggregate(all_sq, mask)
                    row = {
                        "model_arm": arm,
                        "model_seed": seed,
                        "data_arm": data_arm,
                        "prediction_mode": mode,
                        "horizon": args.horizon,
                        **stats,
                        "orientations_nrmse": float(
                            np.sqrt(np.mean(key_sq["orientations"][mask]))
                        ),
                        "height_nrmse": float(
                            np.sqrt(np.mean(key_sq["height"][mask]))
                        ),
                        "velocity_nrmse": float(
                            np.sqrt(np.mean(key_sq["velocity"][mask]))
                        ),
                        "reward_nrmse": float(
                            np.sqrt(np.mean(reward_norm_sq[mask]))
                        ),
                        "reward_rmse": float(
                            np.sqrt(np.mean(reward_raw_sq[mask]))
                        ),
                        "posterior_prior_kl": float(np.mean(kl[mask])),
                    }
                    rows.append(row)

                    reference = existing_aggregate[
                        (arm, seed, data_arm, mode, "all_proprio", args.horizon)
                    ]
                    for field in (
                        "nrmse",
                        "median_window_nrmse",
                        "p95_window_nrmse",
                        "max_window_nrmse",
                    ):
                        comparison_count += 1
                        delta = abs(stats[field] - float(reference[field]))
                        if delta > args.tolerance:
                            mismatches.append({
                                "kind": "aggregate_csv",
                                "arm": arm,
                                "seed": seed,
                                "data_arm": data_arm,
                                "mode": mode,
                                "target_key": "all_proprio",
                                "field": field,
                                "delta": delta,
                            })
                    comparison_count += 1
                    if stats["worst_window_index"] != int(
                        reference["worst_window_index"]
                    ):
                        mismatches.append({
                            "kind": "aggregate_csv",
                            "arm": arm,
                            "seed": seed,
                            "data_arm": data_arm,
                            "mode": mode,
                            "target_key": "all_proprio",
                            "field": "worst_window_index",
                            "computed": stats["worst_window_index"],
                            "reference": int(reference["worst_window_index"]),
                        })
                    for key in OBS_KEYS:
                        comparison_count += 1
                        key_reference = existing_aggregate[
                            (arm, seed, data_arm, mode, key, args.horizon)
                        ]
                        key_value = float(np.sqrt(np.mean(key_sq[key][mask])))
                        delta = abs(key_value - float(key_reference["nrmse"]))
                        if delta > args.tolerance:
                            mismatches.append({
                                "kind": "aggregate_csv",
                                "arm": arm,
                                "seed": seed,
                                "data_arm": data_arm,
                                "mode": mode,
                                "target_key": key,
                                "field": "nrmse",
                                "delta": delta,
                            })
                    comparison_count += 1
                    reward_reference = existing_aggregate[
                        (arm, seed, data_arm, mode, "reward", args.horizon)
                    ]
                    reward_value = float(np.sqrt(np.mean(reward_norm_sq[mask])))
                    delta = abs(reward_value - float(reward_reference["nrmse"]))
                    if delta > args.tolerance:
                        mismatches.append({
                            "kind": "aggregate_csv",
                            "arm": arm,
                            "seed": seed,
                            "data_arm": data_arm,
                            "mode": mode,
                            "target_key": "reward",
                            "field": "nrmse",
                            "delta": delta,
                        })

            primary = summary_primary[(arm, seed)]
            for mode, field in (
                ("prior", "prior_nrmse_h15"),
                ("posterior", "posterior_nrmse_h15"),
            ):
                value = float(np.sqrt(np.mean(window_cache[(arm, seed, mode)])))
                comparison_count += 1
                delta = abs(value - float(primary[field]))
                if delta > args.tolerance:
                    mismatches.append({
                        "kind": "summary_json",
                        "arm": arm,
                        "seed": seed,
                        "field": field,
                        "delta": delta,
                    })
            comparison_count += 1
            kl_delta = abs(float(np.mean(kl)) - float(primary["posterior_prior_kl_h15"]))
            if kl_delta > args.tolerance:
                mismatches.append({
                    "kind": "summary_json",
                    "arm": arm,
                    "seed": seed,
                    "field": "posterior_prior_kl_h15",
                    "delta": kl_delta,
                })

    paired = []
    for seed in SEEDS:
        for mode in ("prior", "posterior"):
            p4 = np.sqrt(window_cache[("p4", seed, mode)])
            e1 = np.sqrt(window_cache[("e1", seed, mode)])
            delta = p4 - e1
            paired.append({
                "seed": seed,
                "prediction_mode": mode,
                "mean_window_delta_p4_minus_e1": float(np.mean(delta)),
                "median_window_delta_p4_minus_e1": float(np.median(delta)),
                "p4_worse_window_fraction": float(np.mean(delta > 0)),
                "trimmed_mean_delta_5pct": float(
                    np.mean(np.sort(delta)[int(0.05 * count):int(0.95 * count)])
                ),
                "top_5pct_abs_contribution_fraction": float(
                    np.sum(np.sort(np.abs(delta))[-max(1, int(0.05 * count)):])
                    / np.sum(np.abs(delta))
                ),
            })

    gates = {
        "p4_prior_nrmse_worse_than_e1_both_seeds": all(
            np.sqrt(np.mean(window_cache[("p4", seed, "prior")]))
            > np.sqrt(np.mean(window_cache[("e1", seed, "prior")]))
            for seed in SEEDS
        ),
        "p4_teacher_nrmse_worse_than_e1_both_seeds": all(
            np.sqrt(np.mean(window_cache[("p4", seed, "posterior")]))
            > np.sqrt(np.mean(window_cache[("e1", seed, "posterior")]))
            for seed in SEEDS
        ),
        "p4_kl_lower_than_e1_both_seeds": all(
            np.mean(kl_cache[("p4", seed)]) < np.mean(kl_cache[("e1", seed)])
            for seed in SEEDS
        ),
    }
    if gates != existing_summary["primary"]["preregistered_gates"]:
        mismatches.append({
            "kind": "preregistered_gates",
            "computed": gates,
            "reference": existing_summary["primary"]["preregistered_gates"],
        })
    comparison_count += len(gates)

    payload = {
        "schema_version": 1,
        "experiment_id": "EXP-0007",
        "scope": "Independent direct recomputation from frozen panel and predictions",
        "implementation_independence": (
            "Does not import analyze_exp0007_openloop.py or its helper functions."
        ),
        "panel_sha256": manifest["panel_sha256"],
        "prediction_sha256": output_hashes,
        "horizon": args.horizon,
        "comparison_tolerance": args.tolerance,
        "comparison_count": comparison_count,
        "comparison_passed": not mismatches,
        "mismatches": mismatches,
        "preregistered_gates": gates,
        "paired_window_robustness": paired,
        "rows_csv": str(args.output_csv),
    }
    write_csv(args.output_csv, rows)
    write_json(args.output_json, payload)
    if mismatches:
        raise SystemExit(f"Independent recomputation found {len(mismatches)} mismatches")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
