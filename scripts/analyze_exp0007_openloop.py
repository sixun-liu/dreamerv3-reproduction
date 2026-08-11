#!/usr/bin/env python3
"""Validate, summarize, and plot the EXP-0007 open-loop diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openloop_panel import sha256, verify_panel, write_json


ARMS = ("baseline", "e1", "p4")
SEEDS = (0, 1)
OBS_KEYS = ("orientations", "height", "velocity")
HORIZONS = (1, 3, 5, 10, 15)
COLORS = {"baseline": "#2f5d8c", "e1": "#c05a3d", "p4": "#2f7d57"}
LABELS = {"baseline": "Baseline", "e1": "E1: no free bits", "p4": "P4: KL weight 1"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reduce_by_window(values: np.ndarray, window_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, np.float64)
    window_ids = np.asarray(window_ids, np.int64)
    unique, inverse = np.unique(window_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique, sums / counts


def summarize_errors(
    norm_sq: np.ndarray,
    raw_sq: np.ndarray,
    raw_abs: np.ndarray,
    decoder_loss: np.ndarray,
    window_ids: np.ndarray,
) -> dict:
    unique, window_norm_sq = reduce_by_window(norm_sq, window_ids)
    _, window_raw_sq = reduce_by_window(raw_sq, window_ids)
    _, window_raw_abs = reduce_by_window(raw_abs, window_ids)
    _, window_decoder = reduce_by_window(decoder_loss, window_ids)
    window_nrmse = np.sqrt(window_norm_sq)
    worst = int(unique[int(np.argmax(window_nrmse))])
    return {
        "n_windows": len(unique),
        "n_rows": len(window_ids),
        "nrmse": float(np.sqrt(np.mean(window_norm_sq))),
        "median_window_nrmse": float(np.median(window_nrmse)),
        "p95_window_nrmse": float(np.quantile(window_nrmse, 0.95)),
        "max_window_nrmse": float(np.max(window_nrmse)),
        "worst_window_index": worst,
        "rmse": float(np.sqrt(np.mean(window_raw_sq))),
        "mae": float(np.mean(window_raw_abs)),
        "decoder_loss_mean": float(np.mean(window_decoder)),
    }


def summarize_values(values: np.ndarray, window_ids: np.ndarray) -> dict:
    unique, window_values = reduce_by_window(values, window_ids)
    return {
        "n_windows": len(unique),
        "mean": float(np.mean(window_values)),
        "median": float(np.median(window_values)),
        "p95": float(np.quantile(window_values, 0.95)),
        "max": float(np.max(window_values)),
    }


def target_scales(panel: dict[str, np.ndarray], context: int, horizon: int) -> dict:
    scales = {}
    for key in (*OBS_KEYS, "reward"):
        target = np.asarray(panel[key][:, context:context + horizon], np.float64)
        flat = target.reshape((-1, 1 if target.ndim == 2 else target.shape[-1]))
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        floor = max(float(np.max(std)) * 1e-6, 1e-8)
        scales[key] = {
            "mean": mean,
            "std": std,
            "scale": np.maximum(std, floor),
            "floor": floor,
        }
    return scales


def load_result(path: Path, metadata_path: Path, panel_sha: str) -> tuple[dict, dict]:
    metadata = read_json(metadata_path)
    if metadata["panel_sha256"] != panel_sha:
        raise ValueError(f"Panel mismatch: {metadata_path}")
    if metadata["output_sha256"] != sha256(path):
        raise ValueError(f"Output SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    nonfinite = {
        key: int((~np.isfinite(value)).sum())
        for key, value in arrays.items()
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
    }
    if nonfinite:
        raise ValueError(f"Non-finite output in {path}: {nonfinite}")
    return arrays, metadata


def error_components(
    arrays: dict,
    panel: dict,
    scales: dict,
    key: str,
    mode: str,
    context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = arrays["window_index"].astype(int)
    pred = np.asarray(arrays[f"{mode}_pred__{key}"], np.float64)
    target = np.asarray(panel[key][windows, context:context + pred.shape[1]], np.float64)
    pred = pred.reshape((*pred.shape[:2], -1))
    target = target.reshape((*target.shape[:2], -1))
    error = pred - target
    scale = np.asarray(scales[key]["scale"], np.float64).reshape((1, 1, -1))
    return (
        np.mean(np.square(error / scale), axis=-1),
        np.mean(np.square(error), axis=-1),
        np.mean(np.abs(error), axis=-1),
        np.asarray(arrays[f"{mode}_decoder_loss__{key}"], np.float64),
    )


def reward_components(
    arrays: dict,
    panel: dict,
    scales: dict,
    mode: str,
    context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = arrays["window_index"].astype(int)
    pred = np.asarray(arrays[f"{mode}_reward_pred"], np.float64)
    target = np.asarray(panel["reward"][windows, context:context + pred.shape[1]], np.float64)
    error = pred - target
    scale = float(np.asarray(scales["reward"]["scale"]).reshape(-1)[0])
    return (
        np.square(error / scale),
        np.square(error),
        np.abs(error),
        np.asarray(arrays[f"{mode}_reward_loss"], np.float64),
    )


def source_labels(manifest: dict, panel: dict) -> tuple[np.ndarray, np.ndarray]:
    arms = np.empty(len(panel["window_index"]), object)
    seeds = np.empty(len(panel["window_index"]), np.int16)
    sources = {row["source_index"]: row for row in manifest["sources"]}
    for index, source_index in enumerate(panel["source_index"].astype(int)):
        arms[index] = sources[source_index]["arm"]
        seeds[index] = sources[source_index]["seed"]
    return arms, seeds


def select_rows(
    row_windows: np.ndarray,
    panel_arms: np.ndarray,
    data_arm: str,
) -> np.ndarray:
    if data_arm == "all":
        return np.ones(len(row_windows), bool)
    return panel_arms[row_windows.astype(int)] == data_arm


def make_aggregate_rows(
    model_results: dict,
    panel: dict,
    manifest: dict,
    scales: dict,
    context: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    aggregate_rows: list[dict] = []
    value_rows: list[dict] = []
    per_window_rows: list[dict] = []
    panel_arms, panel_seeds = source_labels(manifest, panel)

    for (model_arm, model_seed), (arrays, metadata) in model_results.items():
        windows = arrays["window_index"].astype(int)
        draws = metadata["draws_per_window"]
        components = {}
        for mode in ("prior", "posterior"):
            for key in OBS_KEYS:
                components[(mode, key)] = error_components(
                    arrays, panel, scales, key, mode, context
                )
            components[(mode, "reward")] = reward_components(
                arrays, panel, scales, mode, context
            )
            parts = [components[(mode, key)] for key in OBS_KEYS]
            norm_sq = np.concatenate([
                np.repeat(part[0][..., None], panel[key].shape[-1] if panel[key].ndim == 3 else 1, axis=-1)
                for part, key in zip(parts, OBS_KEYS)
            ], axis=-1).mean(-1)
            raw_sq = np.concatenate([
                np.repeat(part[1][..., None], panel[key].shape[-1] if panel[key].ndim == 3 else 1, axis=-1)
                for part, key in zip(parts, OBS_KEYS)
            ], axis=-1).mean(-1)
            raw_abs = np.concatenate([
                np.repeat(part[2][..., None], panel[key].shape[-1] if panel[key].ndim == 3 else 1, axis=-1)
                for part, key in zip(parts, OBS_KEYS)
            ], axis=-1).mean(-1)
            decoder = sum(part[3] for part in parts)
            components[(mode, "all_proprio")] = (norm_sq, raw_sq, raw_abs, decoder)

        for data_arm in ("all", *ARMS):
            mask = select_rows(windows, panel_arms, data_arm)
            for mode in ("prior", "posterior"):
                for key in (*OBS_KEYS, "all_proprio", "reward"):
                    norm_sq, raw_sq, raw_abs, decoder = components[(mode, key)]
                    for horizon in HORIZONS:
                        summary = summarize_errors(
                            norm_sq[mask, horizon - 1],
                            raw_sq[mask, horizon - 1],
                            raw_abs[mask, horizon - 1],
                            decoder[mask, horizon - 1],
                            windows[mask],
                        )
                        aggregate_rows.append({
                            "model_arm": model_arm,
                            "model_seed": model_seed,
                            "data_arm": data_arm,
                            "prediction_mode": mode,
                            "target_key": key,
                            "horizon": horizon,
                            "draws_per_window": draws,
                            **summary,
                        })
            for key, array_key in (
                ("posterior_prior_kl", "posterior_prior_kl"),
                ("prior_entropy", "prior_entropy"),
                ("posterior_entropy", "posterior_entropy"),
            ):
                values = np.asarray(arrays[array_key], np.float64)
                for horizon in HORIZONS:
                    summary = summarize_values(
                        values[mask, horizon - 1], windows[mask]
                    )
                    value_rows.append({
                        "model_arm": model_arm,
                        "model_seed": model_seed,
                        "data_arm": data_arm,
                        "metric": key,
                        "horizon": horizon,
                        "draws_per_window": draws,
                        **summary,
                    })

        # Tail artifact: average stochastic draws inside each replay window.
        for mode in ("prior", "posterior"):
            norm_sq, _, _, decoder = components[(mode, "all_proprio")]
            reward_sq = components[(mode, "reward")][1]
            kl = np.asarray(arrays["posterior_prior_kl"], np.float64)
            for horizon in HORIZONS:
                unique, window_norm_sq = reduce_by_window(
                    norm_sq[:, horizon - 1], windows
                )
                _, window_reward_sq = reduce_by_window(
                    reward_sq[:, horizon - 1], windows
                )
                _, window_decoder = reduce_by_window(
                    decoder[:, horizon - 1], windows
                )
                _, window_kl = reduce_by_window(kl[:, horizon - 1], windows)
                for index, window in enumerate(unique.astype(int)):
                    per_window_rows.append({
                        "model_arm": model_arm,
                        "model_seed": model_seed,
                        "data_arm": panel_arms[window],
                        "data_seed": int(panel_seeds[window]),
                        "window_index": window,
                        "prediction_mode": mode,
                        "horizon": horizon,
                        "proprio_nrmse": math.sqrt(window_norm_sq[index]),
                        "reward_rmse": math.sqrt(window_reward_sq[index]),
                        "decoder_loss": window_decoder[index],
                        "posterior_prior_kl": window_kl[index],
                    })
    return aggregate_rows, value_rows, per_window_rows


def matching(rows: list[dict], **conditions) -> list[dict]:
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def paired_rows(aggregate: list[dict], values: list[dict]) -> list[dict]:
    output = []
    for seed in SEEDS:
        for lhs, rhs, name in (
            ("e1", "baseline", "e1_vs_baseline"),
            ("p4", "e1", "p4_vs_e1"),
        ):
            for horizon in HORIZONS:
                fields = {}
                for mode in ("prior", "posterior"):
                    left = matching(
                        aggregate,
                        model_arm=lhs,
                        model_seed=seed,
                        data_arm="all",
                        prediction_mode=mode,
                        target_key="all_proprio",
                        horizon=horizon,
                    )[0]["nrmse"]
                    right = matching(
                        aggregate,
                        model_arm=rhs,
                        model_seed=seed,
                        data_arm="all",
                        prediction_mode=mode,
                        target_key="all_proprio",
                        horizon=horizon,
                    )[0]["nrmse"]
                    fields[f"{mode}_nrmse_lhs"] = left
                    fields[f"{mode}_nrmse_rhs"] = right
                    fields[f"{mode}_nrmse_delta"] = left - right
                    fields[f"{mode}_nrmse_ratio"] = left / right
                left_kl = matching(
                    values,
                    model_arm=lhs,
                    model_seed=seed,
                    data_arm="all",
                    metric="posterior_prior_kl",
                    horizon=horizon,
                )[0]["mean"]
                right_kl = matching(
                    values,
                    model_arm=rhs,
                    model_seed=seed,
                    data_arm="all",
                    metric="posterior_prior_kl",
                    horizon=horizon,
                )[0]["mean"]
                output.append({
                    "comparison": name,
                    "seed": seed,
                    "horizon": horizon,
                    **fields,
                    "kl_lhs": left_kl,
                    "kl_rhs": right_kl,
                    "kl_delta": left_kl - right_kl,
                    "kl_ratio": left_kl / right_kl,
                })
    return output


def plot_main(
    aggregate: list[dict], values: list[dict], output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for ax, mode, title in (
        (axes[0, 0], "prior", "Open-loop proprio prediction"),
        (axes[0, 1], "posterior", "Teacher-forced reconstruction"),
    ):
        for arm in ARMS:
            seed_values = []
            for seed in SEEDS:
                rows = matching(
                    aggregate,
                    model_arm=arm,
                    model_seed=seed,
                    data_arm="all",
                    prediction_mode=mode,
                    target_key="all_proprio",
                )
                rows = sorted(rows, key=lambda row: row["horizon"])
                seed_values.append([row["nrmse"] for row in rows])
                ax.plot(
                    HORIZONS,
                    seed_values[-1],
                    color=COLORS[arm],
                    alpha=0.28,
                    linewidth=1.2,
                )
            seed_values = np.asarray(seed_values)
            ax.plot(
                HORIZONS,
                np.mean(seed_values, axis=0),
                marker="o",
                color=COLORS[arm],
                linewidth=2.4,
                label=LABELS[arm],
            )
            ax.fill_between(
                HORIZONS,
                np.min(seed_values, axis=0),
                np.max(seed_values, axis=0),
                color=COLORS[arm],
                alpha=0.12,
            )
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("Normalized RMSE")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)

    ax = axes[1, 0]
    for arm in ARMS:
        seed_values = []
        for seed in SEEDS:
            rows = matching(
                values,
                model_arm=arm,
                model_seed=seed,
                data_arm="all",
                metric="posterior_prior_kl",
            )
            rows = sorted(rows, key=lambda row: row["horizon"])
            seed_values.append([row["mean"] for row in rows])
            ax.plot(HORIZONS, seed_values[-1], color=COLORS[arm], alpha=0.28)
        seed_values = np.asarray(seed_values)
        ax.plot(
            HORIZONS,
            np.mean(seed_values, axis=0),
            marker="o",
            color=COLORS[arm],
            linewidth=2.4,
            label=LABELS[arm],
        )
        ax.fill_between(
            HORIZONS,
            np.min(seed_values, axis=0),
            np.max(seed_values, axis=0),
            color=COLORS[arm],
            alpha=0.12,
        )
    ax.set_title("Posterior || prior KL on common observations")
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("KL (nats)")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.22)

    matrix = np.zeros((len(ARMS), len(ARMS)), np.float64)
    for row_index, model_arm in enumerate(ARMS):
        for col_index, data_arm in enumerate(ARMS):
            rows = matching(
                aggregate,
                model_arm=model_arm,
                data_arm=data_arm,
                prediction_mode="prior",
                target_key="all_proprio",
                horizon=15,
            )
            matrix[row_index, col_index] = statistics.fmean(
                row["nrmse"] for row in rows
            )
    ax = axes[1, 1]
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for row in range(len(ARMS)):
        for col in range(len(ARMS)):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > np.median(matrix) else "black",
                fontsize=10,
            )
    ax.set_xticks(range(len(ARMS)), ["Baseline", "E1", "P4"])
    ax.set_yticks(range(len(ARMS)), ["Baseline", "E1", "P4"])
    ax.set_xlabel("Replay source arm")
    ax.set_ylabel("Checkpoint arm")
    ax.set_title("Horizon-15 open-loop NRMSE")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "EXP-0007: finite-context prediction on one shared replay panel",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def primary_summary(aggregate: list[dict], values: list[dict]) -> dict:
    per_seed = []
    for arm in ARMS:
        for seed in SEEDS:
            prior = matching(
                aggregate,
                model_arm=arm,
                model_seed=seed,
                data_arm="all",
                prediction_mode="prior",
                target_key="all_proprio",
                horizon=15,
            )[0]
            posterior = matching(
                aggregate,
                model_arm=arm,
                model_seed=seed,
                data_arm="all",
                prediction_mode="posterior",
                target_key="all_proprio",
                horizon=15,
            )[0]
            kl = matching(
                values,
                model_arm=arm,
                model_seed=seed,
                data_arm="all",
                metric="posterior_prior_kl",
                horizon=15,
            )[0]
            per_seed.append({
                "arm": arm,
                "seed": seed,
                "prior_nrmse_h15": prior["nrmse"],
                "posterior_nrmse_h15": posterior["nrmse"],
                "prior_minus_posterior_h15": prior["nrmse"] - posterior["nrmse"],
                "posterior_prior_kl_h15": kl["mean"],
                "prior_p95_window_nrmse_h15": prior["p95_window_nrmse"],
                "prior_worst_window_h15": prior["worst_window_index"],
            })
    lookup = {(row["arm"], row["seed"]): row for row in per_seed}
    gates = {
        "p4_prior_nrmse_worse_than_e1_both_seeds": all(
            lookup[("p4", seed)]["prior_nrmse_h15"]
            > lookup[("e1", seed)]["prior_nrmse_h15"]
            for seed in SEEDS
        ),
        "p4_teacher_nrmse_worse_than_e1_both_seeds": all(
            lookup[("p4", seed)]["posterior_nrmse_h15"]
            > lookup[("e1", seed)]["posterior_nrmse_h15"]
            for seed in SEEDS
        ),
        "p4_kl_lower_than_e1_both_seeds": all(
            lookup[("p4", seed)]["posterior_prior_kl_h15"]
            < lookup[("e1", seed)]["posterior_prior_kl_h15"]
            for seed in SEEDS
        ),
    }
    return {"per_seed": per_seed, "preregistered_gates": gates}


def write_result(path: Path, summary: dict) -> None:
    rows = {(row["arm"], row["seed"]): row for row in summary["primary"]["per_seed"]}
    gates = summary["primary"]["preregistered_gates"]
    lines = [
        "# EXP-0007 共同数据面板 open-loop 诊断",
        "",
        "## 事实",
        "",
        f"- 共同 panel：{summary['panel']['window_count']} 个无边界窗口，六个 replay 来源各等量；每窗先观察 16 步，再预测 15 步。",
        "- 六个冻结 checkpoint 全部在同一 panel、同一真实 action 序列和固定评估随机种子下完成；每窗 4 个 latent draw，统计时先在窗内平均。",
        "- replay 中缓存的 `dyn/deter` 与 `dyn/stoch` 未进入 panel 或推理输入。",
        "",
        "## Horizon 15 主指标",
        "",
        "| Arm | Seed | Prior NRMSE | Teacher NRMSE | Prior-Teacher | KL(post||prior) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for seed in SEEDS:
            row = rows[(arm, seed)]
            lines.append(
                f"| {LABELS[arm]} | {seed} | {row['prior_nrmse_h15']:.4f} | "
                f"{row['posterior_nrmse_h15']:.4f} | "
                f"{row['prior_minus_posterior_h15']:.4f} | "
                f"{row['posterior_prior_kl_h15']:.4f} |"
            )
    lines += [
        "",
        "## 受限解释",
        "",
        f"- P4 相对 E1 的 H15 prior NRMSE 两 seed 同向恶化：`{gates['p4_prior_nrmse_worse_than_e1_both_seeds']}`。",
        f"- P4 相对 E1 的 teacher-forced NRMSE 两 seed 同向恶化：`{gates['p4_teacher_nrmse_worse_than_e1_both_seeds']}`。",
        f"- P4 相对 E1 的 posterior-prior KL 两 seed 更低：`{gates['p4_kl_lower_than_e1_both_seeds']}`。",
        "- prior 与 teacher 同时变差时，证据更符合表征/decoder 可用信息下降；只有 prior 额外变差时，才更特异地指向 imagination drift。两者必须分开讲。",
        "- 这是固定 walker replay 上的离线有限上下文诊断，不是闭环控制反事实、跨任务结论、论文 Figure 6/17 数值复现，也不能从两个训练 seed 估计总体效应大小。",
        "",
        "## 可复算入口",
        "",
        "- `panel_manifest.json`：窗口来源、offset、chunk SHA256 与动作对齐契约。",
        "- `per_model_source_horizon.csv`：模型 seed × replay 来源 × horizon 的完整聚合。",
        "- `per_window_metrics.csv`：尾部与 worst-window 审计。",
        "- 六个 `predictions.npz`：逐 window、逐 latent draw 的模型原始输出。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict:
    panel_status = verify_panel(
        args.panel, args.manifest, verify_sources=True, verify_hashes=True
    )
    manifest = read_json(args.manifest)
    with np.load(args.panel, allow_pickle=False) as source:
        panel = {key: np.array(source[key], copy=True) for key in source.files}
    scales = target_scales(
        panel, manifest["context_steps"], manifest["prediction_steps"]
    )

    model_results = {}
    integrity = []
    for arm in ARMS:
        for seed in SEEDS:
            directory = args.results_root / arm / f"s{seed:03d}"
            arrays, metadata = load_result(
                directory / "predictions.npz",
                directory / "metadata.json",
                manifest["panel_sha256"],
            )
            if metadata["checkpoint_step"] != 250_000:
                raise ValueError((arm, seed, metadata["checkpoint_step"]))
            expected_rows = manifest["window_count"] * metadata["draws_per_window"]
            if len(arrays["window_index"]) != expected_rows:
                raise ValueError((arm, seed, len(arrays["window_index"]), expected_rows))
            model_results[(arm, seed)] = (arrays, metadata)
            integrity.append({
                "arm": arm,
                "seed": seed,
                "output_sha256": metadata["output_sha256"],
                "checkpoint_agent_sha256": metadata["checkpoint_agent_sha256"],
                "runtime_commit": metadata["runtime_commit"],
                "runtime_clean": not bool(metadata["runtime_tracked_status"]),
                "finite": metadata["finite"],
                "row_count": metadata["row_count"],
            })
    if len({row["runtime_commit"] for row in integrity}) != 1:
        raise ValueError("Runtime commit differs across checkpoint evaluations")
    if not all(row["runtime_clean"] and row["finite"] for row in integrity):
        raise ValueError(f"Integrity failure: {integrity}")

    aggregate, values, per_window = make_aggregate_rows(
        model_results,
        panel,
        manifest,
        scales,
        manifest["context_steps"],
    )
    paired = paired_rows(aggregate, values)
    primary = primary_summary(aggregate, values)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.review.mkdir(parents=True, exist_ok=True)
    write_csv(args.artifacts / "per_model_source_horizon.csv", aggregate)
    write_csv(args.artifacts / "latent_metrics.csv", values)
    write_csv(args.artifacts / "paired_model_comparisons.csv", paired)
    write_csv(args.artifacts / "per_window_metrics.csv", per_window)
    normalization = {
        key: {
            "mean": np.asarray(value["mean"]).tolist(),
            "std": np.asarray(value["std"]).tolist(),
            "scale": np.asarray(value["scale"]).tolist(),
            "floor": value["floor"],
        }
        for key, value in scales.items()
    }
    write_json(args.artifacts / "normalization.json", normalization)
    figure = args.review / "openloop_prediction_diagnostics.png"
    plot_main(aggregate, values, figure)
    summary = {
        "experiment_id": "EXP-0007",
        "scope": "Offline finite-context prediction on frozen EXP-0006 replay",
        "panel": {**panel_status, "path": str(args.panel)},
        "protocol": {
            "context_steps": manifest["context_steps"],
            "horizons": list(HORIZONS),
            "sampling_unit": "replay window",
            "latent_draw_aggregation": "mean within window before across-window statistics",
            "normalization": "per raw observation dimension using shared panel target std",
            "cached_latents_used": False,
        },
        "integrity": integrity,
        "primary": primary,
        "artifacts": {
            "aggregate": str(args.artifacts / "per_model_source_horizon.csv"),
            "latent": str(args.artifacts / "latent_metrics.csv"),
            "paired": str(args.artifacts / "paired_model_comparisons.csv"),
            "per_window": str(args.artifacts / "per_window_metrics.csv"),
            "normalization": str(args.artifacts / "normalization.json"),
            "main_figure": str(figure),
        },
        "limitations": [
            "Only two training seeds and one DMC proprio task.",
            "Finite 16-step context starts from zero latent state.",
            "Offline real-action replay does not establish closed-loop control causality.",
            "P4 is reconstructed from code semantics, not the paper's original ablation artifact.",
        ],
    }
    write_json(args.artifacts / "summary.json", summary)
    write_result(args.artifacts / "RESULT.md", summary)
    (args.review / "README.md").write_text(
        "# EXP-0007 Review\n\n"
        "主图：`openloop_prediction_diagnostics.png`。实线为两个模型 seed 的均值，"
        "浅色范围为两个 seed 的 min--max；这不是置信区间。\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({
        "panel": summary["panel"],
        "gates": summary["primary"]["preregistered_gates"],
        "artifacts": summary["artifacts"],
    }, indent=2))


if __name__ == "__main__":
    main()
