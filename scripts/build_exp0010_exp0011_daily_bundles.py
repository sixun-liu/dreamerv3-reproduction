#!/usr/bin/env python3
"""Build daily-report bundles for EXP-0010 and EXP-0011."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import av
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/root/autodl-tmp")
CONTROL = ROOT / "Code/DreamerV3/dreamerv3-reproduction"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
COLORS = {
    "ink": "#18212B",
    "muted": "#5D6873",
    "paper": "#F5F7F8",
    "green": "#2F7D59",
    "teal": "#167D8D",
    "gold": "#C58A16",
    "red": "#B44747",
    "gray": "#AAB2BA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-date", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    return {
        "title": ImageFont.truetype(str(FONT_BOLD), 27),
        "label": ImageFont.truetype(str(FONT_BOLD), 18),
        "body": ImageFont.truetype(str(FONT), 16),
        "small": ImageFont.truetype(str(FONT), 13),
    }


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def read_video(path: Path) -> tuple[list[Image.Image], float]:
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames.extend(frame.to_image().convert("RGB") for frame in container.decode(stream))
    if not frames:
        raise ValueError(f"No video frames: {path}")
    return frames, fps


def labeled_frame(
    source: Image.Image,
    title: str,
    subtitle: str,
    index: int,
    count: int,
    fps: float,
) -> Image.Image:
    fs = fonts()
    canvas = Image.new("RGB", (448, 510), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    centered_text(draw, (0, 8, 448, 50), title, fs["title"], COLORS["ink"])
    centered_text(draw, (0, 48, 448, 78), subtitle, fs["small"], COLORS["muted"])
    image = source.resize((416, 416), Image.Resampling.NEAREST)
    canvas.paste(image, (16, 82))
    centered_text(
        draw,
        (0, 481, 448, 508),
        f"episode0 | t={index / fps:05.1f}s / {(count - 1) / fps:05.1f}s | realtime",
        fs["small"],
        COLORS["muted"],
    )
    return canvas


def write_realtime_gif(
    path: Path,
    source_frames: list[Image.Image],
    source_fps: float,
    stride: int,
    title: str,
    subtitle: str,
) -> dict:
    indices = list(range(0, len(source_frames), stride))
    frames = [
        labeled_frame(
            source_frames[index], title, subtitle, index, len(source_frames), source_fps
        )
        for index in indices
    ]
    gif_fps = source_fps / stride
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / gif_fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "source_frames": len(source_frames),
        "source_fps": source_fps,
        "sample_stride": stride,
        "gif_frames": len(frames),
        "gif_fps": gif_fps,
        "source_duration_seconds": len(source_frames) / source_fps,
        "gif_duration_seconds": len(frames) / gif_fps,
        "speed_policy": "sample stride and GIF frame rate reduced by the same factor",
    }


def build_contact_sheet(
    path: Path,
    source_frames: list[Image.Image],
    source_fps: float,
    title: str,
) -> None:
    fs = fonts()
    indices = np.linspace(0, len(source_frames) - 1, 6, dtype=int).tolist()
    canvas = Image.new("RGB", (1180, 870), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    centered_text(draw, (0, 12, 1180, 60), title, fs["title"], COLORS["ink"])
    for position, index in enumerate(indices):
        row, column = divmod(position, 3)
        x, y = 28 + column * 384, 82 + row * 370
        image = source_frames[index].resize((352, 352), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        centered_text(
            draw,
            (x, y + 326, x + 352, y + 352),
            f"{100 * index / (len(source_frames) - 1):.0f}% | {index / source_fps:.1f}s",
            fs["small"],
            "white",
        )
    centered_text(
        draw,
        (0, 824, 1180, 862),
        "Six uniform moments from preregistered evaluation episode 0; never best-of-N.",
        fs["body"],
        COLORS["muted"],
    )
    canvas.save(path)


def verdict_cards(ax: plt.Axes, cards: list[tuple[str, str, str, str]]) -> None:
    ax.axis("off")
    ax.set_title("Bounded verdict", loc="left", fontsize=15, fontweight="bold", pad=8)
    width = 0.225
    for index, (header, verdict, detail, color) in enumerate(cards):
        x = 0.01 + index * 0.247
        ax.add_patch(
            plt.Rectangle(
                (x, 0.12), width, 0.70, transform=ax.transAxes,
                facecolor="white", edgecolor="#D7DCE0",
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x, 0.73), width, 0.09, transform=ax.transAxes,
                facecolor=color, edgecolor=color,
            )
        )
        ax.text(x + 0.012, 0.755, header, color="white", fontsize=9.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, 0.60, verdict, color=color, fontsize=15, fontweight="bold", transform=ax.transAxes)
        words, lines, line = detail.split(), [], []
        for word in words:
            if len(" ".join(line + [word])) > 31:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        ax.text(x + 0.012, 0.49, "\n".join(lines), color=COLORS["muted"], fontsize=10, va="top", linespacing=1.45, transform=ax.transAxes)


def resource_panel(ax: plt.Axes, rows: list[tuple[str, str]]) -> None:
    ax.axis("off")
    ax.set_title("Run and resource facts", loc="left", fontweight="bold", pad=12)
    for row, (label, value) in enumerate(rows):
        y = 0.88 - row * 0.17
        ax.text(0.02, y, label, color=COLORS["muted"], fontsize=11, transform=ax.transAxes)
        ax.text(0.98, y, value, ha="right", color=COLORS["ink"], fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.plot([0.02, 0.98], [y - 0.055, y - 0.055], color="#D7DCE0", transform=ax.transAxes)


def build_exp0010_overview(summary: dict, auc: dict, evaluation: dict, path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.82), hspace=0.38, wspace=0.32)
    fig.suptitle("DreamerV3 DMC Vision Walker | EXP-0010", fontsize=25, fontweight="bold", color=COLORS["ink"], y=0.97)
    fig.text(0.5, 0.925, "1M environment steps | pixel-only control | single-seed author-reimplementation alignment", ha="center", fontsize=13, color=COLORS["muted"])

    ax = fig.add_subplot(grid[0, 0])
    local = summary["final_envelope"]["local_tail_mean"]
    official = summary["official"]["tail"]
    ax.errorbar(
        [0], [official["mean"]],
        yerr=[[official["mean"] - official["range"][0]], [official["range"][1] - official["mean"]]],
        fmt="o", markersize=10, capsize=8, color=COLORS["ink"], label="Official 10-seed range",
    )
    ax.scatter([1], [local], s=120, color=COLORS["green"], label="Local seed 0")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(945, 970)
    ax.set_xticks([0, 1], ["Official", "Local"])
    ax.set_ylabel("Final 100K mean return")
    ax.set_title("Terminal numeric alignment", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", frameon=False)
    ax.text(1, local + 1.0, f"{local:.2f}", ha="center", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    values = [episode["return"] for episode in evaluation["episodes"]]
    ax.scatter(range(1, 11), values, color=COLORS["teal"], s=65)
    ax.axhline(evaluation["return_mean"], color=COLORS["gold"], linewidth=2, label=f"mean {evaluation['return_mean']:.2f}")
    ax.set_xlabel("Independent evaluation episode")
    ax.set_ylabel("Return")
    ax.set_title("Terminal checkpoint evaluation", loc="left", fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 2])
    gpu = summary["resource"]["gpu_samples"]
    resource_panel(
        ax,
        [
            ("Training wall time", f"{summary['resource']['wall_seconds'] / 3600:.2f} h"),
            ("Peak VRAM", f"{gpu['memory_used_mib_max'] / 1024:.1f} GiB"),
            ("Mean sampled GPU util.", f"{gpu['gpu_util_percent_mean']:.1f}%"),
            ("Local normalized AUC", f"{auc['normalized_auc']:.2f}"),
            ("Eval return range", f"{evaluation['return_min']:.1f} - {evaluation['return_max']:.1f}"),
        ],
    )
    verdict_cards(
        fig.add_subplot(grid[1, :]),
        [
            ("ENGINEERING", "COMPLETE", "Pixel input, exact checkpoint, replay, finite metrics and independent evaluation all passed.", COLORS["teal"]),
            ("TERMINAL SCORE", "ALIGNED", "Local final-100K mean 959.43 lies inside the public 10-seed range.", COLORS["green"]),
            ("LEARNING", "AUC IN RANGE", "Local fixed-bin AUC 849.43 is within the descriptive official per-seed range.", COLORS["gold"]),
            ("BOUNDARY", "SINGLE SEED", "This is not cross-seed, exact-2023-artifact or full DMC Vision replication.", COLORS["red"]),
        ],
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_exp0011_overview(summary: dict, evaluation: dict, path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.82), hspace=0.38, wspace=0.32)
    fig.suptitle("DreamerV3 Atari100K Breakout | EXP-0011", fontsize=25, fontweight="bold", color=COLORS["ink"], y=0.97)
    fig.text(0.5, 0.925, "100K agent decisions | size50m / ratio256 | scaled single-seed learning instance", ha="center", fontsize=13, color=COLORS["muted"])

    ax = fig.add_subplot(grid[0, 0])
    early = summary["local"]["early_0_40k_frames"]["mean"]
    tail = summary["local"]["tail_360_400k_frames"]["mean"]
    bars = ax.bar(["Early 0-40K", "Tail 360-400K"], [early, tail], color=[COLORS["gray"], COLORS["green"]], width=0.62)
    ax.axhline(3.0, color=COLORS["gold"], linestyle="--", label="Frozen tail threshold")
    ax.set_ylabel("Training raw score mean")
    ax.set_title("Frozen learning trend gate", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for bar, value in zip(bars, (early, tail)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    values = [episode["return"] for episode in evaluation["episodes"]]
    ax.scatter(range(1, 11), values, color=COLORS["teal"], s=65)
    ax.axhline(evaluation["return_mean"], color=COLORS["gold"], linewidth=2, label=f"mean {evaluation['return_mean']:.1f}")
    ax.set_xlabel("Independent evaluation episode")
    ax.set_ylabel("Raw return")
    ax.set_title("Terminal checkpoint evaluation", loc="left", fontweight="bold")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 2])
    gpu = summary["resource"]["gpu_samples"]
    resource_panel(
        ax,
        [
            ("Training wall time", f"{summary['resource']['wall_seconds'] / 60:.1f} min"),
            ("Peak VRAM", f"{gpu['memory_used_mib_max'] / 1024:.1f} GiB"),
            ("Mean sampled GPU util.", f"{gpu['gpu_util_percent_mean']:.1f}%"),
            ("Training episodes", str(summary["local"]["episode_count"])),
            ("Eval return range", f"{evaluation['return_min']:.0f} - {evaluation['return_max']:.0f}"),
        ],
    )
    verdict_cards(
        fig.add_subplot(grid[1, :]),
        [
            ("ENGINEERING", "COMPLETE", "ALE L0, exact checkpoint, replay, finite metrics and fixed evaluation all passed.", COLORS["teal"]),
            ("LEARNING TREND", "PASSED", "Tail mean 7.54 exceeds early mean 1.29 by 6.26 and clears the frozen gate.", COLORS["green"]),
            ("TERMINAL POLICY", "USABLE", "Independent 10-episode return mean is 9.1 with range 5 to 16.", COLORS["gold"]),
            ("BOUNDARY", "SCALED ONLY", "Size50m/ratio256 differs from paper 200M/ratio128; no direct DQN ranking is valid.", COLORS["red"]),
        ],
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def experiment_paths(experiment_id: str) -> dict:
    if experiment_id == "EXP-0010":
        artifact = ROOT / "Artifacts/dreamerv3/EXP-0010"
        review = ROOT / "Artifacts/dreamerv3/review/EXP-0010-dmc-vision-walker-staged"
        train = ROOT / "Runs/EXP-0010__walker-vision__s000__staged-1m-env__20260811T151208Z"
        evaluation = ROOT / "Runs/EXP-0010__walker-vision__eval-s10000-10eps__20260811T195043Z"
        return {
            "slug": "dreamerv3-dmcvision-walker",
            "zip": "DreamerV3_EXP-0010_DMC_Vision_Walker_Daily_2026-08-12.zip",
            "runtime": ROOT / "Code/DreamerV3/dreamerv3-runtime-2411-crossdomain",
            "artifact": artifact,
            "review": review,
            "train": train,
            "eval": evaluation,
            "summary": artifact / "summary_full.json",
            "extra": artifact / "auc_supplement.json",
            "evaluation": evaluation / "evaluation/evaluation.json",
            "video": evaluation / "evaluation/episode_000_preregistered.mp4",
            "curve": review / "dmcvision_walker_full.png",
            "result": artifact / "RESULT_full.md",
            "protocol": CONTROL / "docs/reproduction/EXP0010_DMC_VISION_PROTOCOL.md",
            "config": CONTROL / "docs/reproduction/configs/exp0010_dmcvision_s000_1m_env.yaml",
            "closure": "EVT-0067",
            "artifacts": [f"ART-{value:04d}" for value in range(54, 63)],
            "gif_stride": 2,
        }
    artifact = ROOT / "Artifacts/dreamerv3/EXP-0011"
    review = ROOT / "Artifacts/dreamerv3/review/EXP-0011-atari100k-breakout"
    train = ROOT / "Runs/EXP-0011__breakout__s000__100k-dec__20260811T201000Z"
    evaluation = ROOT / "Runs/EXP-0011__breakout__eval-s10000-env20260812-10eps__20260811T231600Z"
    return {
        "slug": "dreamerv3-atari100k-breakout",
        "zip": "DreamerV3_EXP-0011_Atari100K_Breakout_Daily_2026-08-12.zip",
        "runtime": ROOT / "Code/DreamerV3/dreamerv3-runtime-2026-crossdomain",
        "artifact": artifact,
        "review": review,
        "train": train,
        "eval": evaluation,
        "summary": artifact / "summary_formal.json",
        "extra": CONTROL / "docs/reproduction/EXP0011_DQN_PROTOCOL_COMPARISON.md",
        "evaluation": artifact / "evaluation/evaluation.json",
        "video": artifact / "evaluation/episode_000_preregistered.mp4",
        "curve": review / "atari100k_breakout_formal.png",
        "result": artifact / "RESULT.md",
        "protocol": CONTROL / "docs/reproduction/EXP0011_ATARI100K_PROTOCOL.md",
        "config": CONTROL / "docs/reproduction/configs/exp0011_atari100k_s000_100k_dec.yaml",
        "closure": "EVT-0080",
        "artifacts": [f"ART-{value:04d}" for value in range(63, 71)],
        "gif_stride": 3,
    }


def source_copies(experiment_id: str, paths: dict) -> dict[str, Path]:
    common = {
        "data/summary.json": paths["summary"],
        "data/evaluation.json": paths["evaluation"],
        "figures/learning_curve.png": paths["curve"],
        "demo/episode_000_full_realtime.mp4": paths["video"],
        "protocol/RESULT.md": paths["result"],
        "protocol/PROTOCOL.md": paths["protocol"],
        "protocol/config.yaml": paths["config"],
        "provenance/formal.freeze.json": paths["train"] / ".freeze",
        "provenance/formal.completed.json": paths["train"] / ".completed",
        "provenance/formal.integrity.json": paths["train"] / ("integrity_full.json" if experiment_id == "EXP-0010" else "integrity_formal.json"),
        "provenance/evaluation.freeze.json": paths["eval"] / ".freeze",
        "provenance/evaluation.completed.json": paths["eval"] / ".completed",
        "provenance/scripts/bundle_builder.py": Path(__file__).resolve(),
    }
    if experiment_id == "EXP-0010":
        common["data/auc_supplement.json"] = paths["extra"]
        common["data/curve.csv"] = paths["artifact"] / "curve_full.csv"
        common["provenance/scripts/analysis.py"] = CONTROL / "scripts/analyze_exp0010_dmcvision.py"
        common["provenance/scripts/evaluation.py"] = CONTROL / "scripts/evaluate_exp0010_checkpoint.py"
    else:
        common["protocol/DQN_PROTOCOL_COMPARISON.md"] = paths["extra"]
        common["data/curve.csv"] = paths["artifact"] / "curve_formal.csv"
        common["provenance/scripts/analysis.py"] = CONTROL / "scripts/analyze_exp0011_atari.py"
        common["provenance/scripts/evaluation.py"] = CONTROL / "scripts/evaluate_exp0011_atari.py"
    return common


def copy_sources(output: Path, experiment_id: str, paths: dict) -> list[dict]:
    records = []
    for destination, source in source_copies(experiment_id, paths).items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "destination": destination,
            "source_path": str(source),
            "source_sha256": sha256(source),
            "bytes": source.stat().st_size,
        })
    return records


def write_exp0010_docs(output: Path, report_date: str, summary: dict, auc: dict, evaluation: dict) -> None:
    output.joinpath("README.md").write_text(
        f"""# DreamerV3 DMC Vision Walker 日报材料包（{report_date}）

本目录由 `EXP-0010` 已登记证据派生，没有重新训练或挑选最好回合。固定 episode0 的 MP4 和 GIF
均保持原时间尺度；GIF 仅隔帧压缩，同时按相同比例降低播放帧率。

## 推荐顺序

1. `01_日报事实卡.md`
2. `figures/result_overview.png`
3. `figures/learning_curve.png`
4. `demo/episode_000_full_realtime.gif`
5. `figures/episode_000_contact_sheet.png`
6. `03_日报草稿.md`

完整 MP4、原始 JSON/CSV、协议、冻结信标、来源 manifest 与 `SHA256SUMS` 均在本包中。
科学结论仍以 `EVT-0067`、`ART-0054`--`ART-0062` 为准，人工审查状态为 `pending`。
""",
        encoding="utf-8",
    )
    output.joinpath("01_日报事实卡.md").write_text(
        f"""# EXP-0010 日报事实卡

- 任务：DMC Vision Walker Walk，64x64 RGB 像素输入。
- runtime：作者 2024 重实现谱系 `6642b94`；size12m、repeat2、ratio512、seed0。
- staged 预算：100K gate 通过后，同一 checkpoint/replay 续至 1M environment steps。
- 完整性：精确 checkpoint step500000、992 个训练回合、736 个 replay 文件，有限性检查通过。
- 训练末 100K 均值：`{summary['local']['tail']['mean']:.2f}`；官方 10-seed 范围：
  `{summary['official']['tail']['range'][0]:.2f}--{summary['official']['tail']['range'][1]:.2f}`。
- 本地 fixed-bin AUC：`{auc['normalized_auc']:.2f}`；官方逐 seed 范围：
  `{auc['official_normalized_auc_range'][0]:.2f}--{auc['official_normalized_auc_range'][1]:.2f}`。
- 独立 10 局：均值 `{evaluation['return_mean']:.2f}`，中位数 `{evaluation['return_median']:.2f}`，
  范围 `{evaluation['return_min']:.2f}--{evaluation['return_max']:.2f}`。
- 训练墙钟：`{summary['resource']['wall_seconds'] / 3600:.2f}h`；峰值显存
  `{summary['resource']['gpu_samples']['memory_used_mib_max']:.0f}MiB`；平均 GPU 利用率
  `{summary['resource']['gpu_samples']['gpu_util_percent_mean']:.2f}%`。

结论：形成单 seed 的 DMC Vision 数值与行为对齐实例；不是跨 seed、整个 DMC Vision 域或 exact
2023 artifact 的完整复现。公开平滑曲线与本地完整 episode 分箱也不是逐样本同构。
""",
        encoding="utf-8",
    )
    output.joinpath("02_实验过程与结果.md").write_text(
        """# 实验过程与结果

本轮先用 8192-decision smoke 验证像素观察、模型训练、checkpoint/replay 和单回合行为录制。随后
运行精确 50K decisions 的 100K environment-step pilot；末段均值和相对早期改善通过预注册 gate，
ETA 与磁盘门也通过，因此按冻结规则在同一 logdir、replay 和 checkpoint 上续训至 500K decisions。

正式训练完成后，本地末 100K 均值进入公开十个 seed 的末段范围，fixed-bin AUC 也进入公开逐 seed
范围。独立终点 checkpoint 评测十局均取得高回报，固定第 0 局视频显示 Walker 从 64x64 图像输入
形成稳定行走行为。

该结果最有价值的地方是补齐了从 DMC proprio 到纯像素控制的证据，但单 agent seed 与未受控的 DMC
environment RNG 仍限制稳定性结论。公开视频曲线是平滑导出，本地曲线是完整训练回合的固定分箱，
因此数值范围用于描述性对齐，不把每个点解释为样本等价。
""",
        encoding="utf-8",
    )
    output.joinpath("03_日报草稿.md").write_text(
        f"""# DreamerV3 复现日报草稿（{report_date}）

今天整理了 DreamerV3 在 DMC Vision Walker Walk 上的像素控制复现。实验使用作者 2024 重实现
谱系，配置为 64x64 RGB、size12m、action repeat 2、replay ratio 512 和 agent seed 0。先运行
100K environment-step gate；通过学习趋势、完整性、时间和磁盘门后，在同一 checkpoint/replay 上
续训至 1M environment steps。

完整训练自然结束在 checkpoint step500000，共记录 992 个有限训练回合。末 100K 回报均值为
{summary['local']['tail']['mean']:.2f}，进入公开十个 seed 的末段范围
{summary['official']['tail']['range'][0]:.2f}--{summary['official']['tail']['range'][1]:.2f}。本地固定分箱
AUC 为 {auc['normalized_auc']:.2f}，也处在公开逐 seed 范围内。终点 checkpoint 独立评测十局均值
为 {evaluation['return_mean']:.2f}，范围 {evaluation['return_min']:.2f}--{evaluation['return_max']:.2f}。

正式训练耗时约 {summary['resource']['wall_seconds'] / 3600:.2f} 小时，峰值显存
{summary['resource']['gpu_samples']['memory_used_mib_max']:.0f} MiB，平均采样 GPU 利用率
{summary['resource']['gpu_samples']['gpu_util_percent_mean']:.1f}%。本次同时整理了论文曲线对比、结果
总览、固定 episode0 的正常速度 GIF/MP4 和六时刻行为图。

准确结论是：作者重实现在单 seed 上形成了 DMC Vision Walker 的数值与行为对齐实例。由于只有一个
agent seed、DMC 环境随机数未由 runtime 控制，且不是 exact 2023 artifact，不能称为跨 seed 或整个
DMC Vision 域的完整复现。
""",
        encoding="utf-8",
    )
    output.joinpath("04_口径与常见问答.md").write_text(
        """# 口径与常见问答

## 末段均值进入官方范围是否等于完整复现？

不等于。这是单 agent seed 的作者重实现结果；官方范围来自十个 seed，训练 artifact 和导出口径也
不完全相同。可以说“单 seed 数值对齐实例”，不能说“完整复现整个 DMC Vision 结果”。

## AUC 有什么补充价值？

末段分数只看终点，AUC 同时反映学习速度。本地 AUC 虽进入官方逐 seed 范围，但前约 200K 学得更慢，
说明终点对齐不能抹掉早期差异。

## 视频能证明从像素学习吗？

视频证明固定终点策略能产生有效行为，配置检查证明 encoder 只消费图像。它不能单独证明跨 seed
稳定性，主结论仍需训练统计、配置和 checkpoint 完整性共同支持。

## GIF 是否被加速？

没有。源视频 20 FPS，GIF 每隔两帧取一帧并以 10 FPS 播放，总时长保持约 25 秒。
""",
        encoding="utf-8",
    )


def write_exp0011_docs(output: Path, report_date: str, summary: dict, evaluation: dict) -> None:
    early = summary["local"]["early_0_40k_frames"]["mean"]
    tail = summary["local"]["tail_360_400k_frames"]["mean"]
    output.joinpath("README.md").write_text(
        f"""# DreamerV3 Atari100K Breakout 日报材料包（{report_date}）

本目录由 `EXP-0011` 已登记证据派生，没有重新训练或挑选最好回合。固定 episode0 的 MP4 和 GIF
均保持原时间尺度；GIF 只降采样帧率，不改变总时长。

建议依次查看事实卡、`figures/result_overview.png`、`figures/learning_curve.png`、正常速度 GIF、
六时刻联系表和日报草稿。包内同时保留独立评测、协议、DQN 协议差异、冻结信标与哈希清单。

科学结论仍以 `EVT-0080`、`ART-0063`--`ART-0070` 为准，人工审查状态为 `pending`。
""",
        encoding="utf-8",
    )
    output.joinpath("01_日报事实卡.md").write_text(
        f"""# EXP-0011 日报事实卡

- 任务：Atari100K Breakout，现代 ALE，64x64 RGB。
- runtime：作者 2026 重实现 `5168475`；size50m、ratio256、repeat4、seed0。
- 预算：精确 100K agent decisions，即 400K nominal emulator frames。
- 完整性：258 个训练回合、精确终点 checkpoint、非空 replay、有限 losses 与资源账均通过。
- 训练早期均值：`{early:.3f}`；末 40K frames 均值：`{tail:.3f}`；提升
  `{summary['local']['tail_minus_early_mean']:.3f}`，预注册趋势门通过。
- 独立十局：`{[episode['return'] for episode in evaluation['episodes']]}`；均值
  `{evaluation['return_mean']:.1f}`，中位数 `{evaluation['return_median']:.1f}`，范围
  `{evaluation['return_min']:.0f}--{evaluation['return_max']:.0f}`。
- 正式训练：`{summary['resource']['wall_seconds'] / 60:.1f}min`；峰值显存
  `{summary['resource']['gpu_samples']['memory_used_mib_max']:.0f}MiB`；平均 GPU 利用率
  `{summary['resource']['gpu_samples']['gpu_util_percent_mean']:.2f}%`。

结论：在 size50m/ratio256 的单 seed 降规模协议下形成清晰学习趋势和可用终点策略；不是论文
200M/ratio128 严格复现、Atari100K 全域复现或与 DQN 的直接排行榜比较。
""",
        encoding="utf-8",
    )
    output.joinpath("02_实验过程与结果.md").write_text(
        """# 实验过程与结果

本轮先验证 Breakout ROM、minimal action set、repeat4、sticky false、随机 0--30 no-op、原始奖励、
64x64 图像与终止/reset 语义。两个早期 smoke 暴露了 CLI 代际变化和 report 时钟问题；失败信号完整
保留。第三个 smoke 在精确 4090 decisions 上通过 checkpoint、replay、有限 loss、ETA 和磁盘门。

正式训练自然完成精确 100K decisions。训练末段均值显著高于早期并通过冻结趋势门。独立终点
checkpoint 评测使用固定 agent/environment seed 跑十局，均值 9.1；固定第 0 局回报为 7，视频没有
按回报选择。两次评测前接口失败分别来自无换行 checkpoint 索引和新版 Agent 配置入口，均发生在
环境动作之前，替代流程只修复读取接口，不改变 checkpoint 或评测协议。

公开五 seed 曲线只作为描述性上下文。本地模型为 50M/ratio256，而论文上下文为 200M/ratio128，
训练与独立评测也不是同一采样口径，因此不能把曲线接近或独立均值直接解释成严格论文复现。
""",
        encoding="utf-8",
    )
    output.joinpath("03_日报草稿.md").write_text(
        f"""# DreamerV3 复现日报草稿（{report_date}）

今天整理了 DreamerV3 在 Atari100K Breakout 上的低数据预算实验。实验采用作者 2026 重实现，
配置为 size50m、replay ratio 256、action repeat 4 和单 agent seed，在精确 100K agent decisions
即 400K nominal emulator frames 内训练。

正式 run 自然完成并保存精确终点 checkpoint，共记录 258 个有限训练回合。首 0--40K emulator
frames 的平均回报为 {early:.3f}，末 360K--400K 为 {tail:.3f}，提升
{summary['local']['tail_minus_early_mean']:.3f}，通过预注册的学习趋势门。终点 checkpoint 独立评测
十局均值为 {evaluation['return_mean']:.1f}，中位数 {evaluation['return_median']:.1f}，范围
{evaluation['return_min']:.0f}--{evaluation['return_max']:.0f}。

训练耗时约 {summary['resource']['wall_seconds'] / 60:.1f} 分钟，峰值显存
{summary['resource']['gpu_samples']['memory_used_mib_max']:.0f} MiB，平均采样 GPU 利用率
{summary['resource']['gpu_samples']['gpu_util_percent_mean']:.1f}%。本次整理了训练曲线、结果总览、固定
episode0 的正常速度 GIF/MP4、六时刻截图和 DreamerV3/DQN 协议差异说明。

较准确的结论是：作者重实现在单 seed、降规模配置下形成了清晰的 Breakout 学习趋势和可用策略。
由于模型规模、train ratio 和代码代际与论文上下文不同，目前不能称为 200M/ratio128 严格复现，
也不能与此前 DQN 分数直接排名。
""",
        encoding="utf-8",
    )
    output.joinpath("04_口径与常见问答.md").write_text(
        """# 口径与常见问答

## 本地末段进入公开范围是否等于论文复现？

不等于。公开曲线使用 200M/ratio128 五 seed，本地使用 50M/ratio256 单 seed。准确说法是“降规模
作者重实现出现学习趋势且数值落入描述性范围”。

## 独立评测 9.1 能与训练末段 7.54 直接比较吗？

不能。独立评测来自固定终点 checkpoint 和新环境序列，训练末段来自训练过程完整回合；二者回答的
问题不同。前者检查终点策略可用性，后者检验训练趋势。

## 能否据此说 DreamerV3 优于 DQN？

不能。两条实验的模型、预算单位、输入、replay、奖励、ALE/no-op 和评测协议均不完全等价。

## GIF 是否加速？

没有。源视频 15 FPS，GIF 每三帧取一帧并以 5 FPS 播放，总时长保持约 30 秒。
""",
        encoding="utf-8",
    )


def validate_media(output: Path, expected_source_frames: int, expected_gif_frames: int) -> dict:
    gif_path = output / "demo/episode_000_full_realtime.gif"
    arrays = []
    durations = []
    with Image.open(gif_path) as image:
        gif_frames = int(getattr(image, "n_frames", 1))
        for index in range(gif_frames):
            image.seek(index)
            arrays.append(np.asarray(image.convert("RGB")))
            durations.append(int(image.info.get("duration", 0)))
    if gif_frames != expected_gif_frames:
        raise ValueError(f"GIF frame mismatch: {gif_frames}")
    dynamic_pairs = sum(bool(np.any(left != right)) for left, right in zip(arrays, arrays[1:]))
    if dynamic_pairs < gif_frames - 2:
        raise ValueError(f"GIF unexpectedly static: {dynamic_pairs}")
    figures = {}
    for relative in ("figures/result_overview.png", "figures/episode_000_contact_sheet.png"):
        with Image.open(output / relative) as image:
            array = np.asarray(image.convert("RGB"))
            figures[relative] = {"width": image.width, "height": image.height, "pixel_std": float(array.std())}
            if image.width < 1000 or image.height < 700 or array.std() < 5:
                raise ValueError(f"Invalid figure: {relative}")
    return {
        "schema_version": 1,
        "source_frames": expected_source_frames,
        "gif_frames": gif_frames,
        "gif_total_duration_seconds": sum(durations) / 1000,
        "dynamic_adjacent_pairs": dynamic_pairs,
        "figures": figures,
        "passed": True,
    }


def write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    output.joinpath("SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zip(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            archive.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8"
    )


def build_one(experiment_id: str, output_root: Path, report_date: str) -> dict:
    paths = experiment_paths(experiment_id)
    output = output_root / paths["slug"]
    zip_path = output_root / paths["zip"]
    if output.exists() or zip_path.exists() or zip_path.with_suffix(zip_path.suffix + ".sha256").exists():
        raise FileExistsError(f"Refusing to overwrite outputs for {experiment_id}")
    for subdir in ("data", "demo", "figures", "protocol", "provenance"):
        (output / subdir).mkdir(parents=True, exist_ok=False)
    copied = copy_sources(output, experiment_id, paths)
    summary = read_json(paths["summary"])
    evaluation = read_json(paths["evaluation"])
    if not summary["integrity"]["passed"] or evaluation["episode_count"] != 10:
        raise ValueError(f"{experiment_id}: integrity/evaluation gate mismatch")
    frames, fps = read_video(paths["video"])
    if evaluation.get("video_selection") != "episode index 0, preregistered; not best-of-N":
        raise ValueError(f"{experiment_id}: video was not preregistered episode0")
    gif_metadata = write_realtime_gif(
        output / "demo/episode_000_full_realtime.gif",
        frames,
        fps,
        paths["gif_stride"],
        "DMC Vision Walker" if experiment_id == "EXP-0010" else "Atari100K Breakout",
        "Fixed terminal-checkpoint evaluation | not best-of-N",
    )
    build_contact_sheet(
        output / "figures/episode_000_contact_sheet.png",
        frames,
        fps,
        ("DMC Vision Walker" if experiment_id == "EXP-0010" else "Atari100K Breakout") + " fixed episode0",
    )
    if experiment_id == "EXP-0010":
        extra = read_json(paths["extra"])
        build_exp0010_overview(summary, extra, evaluation, output / "figures/result_overview.png")
        write_exp0010_docs(output, report_date, summary, extra, evaluation)
    else:
        build_exp0011_overview(summary, evaluation, output / "figures/result_overview.png")
        write_exp0011_docs(output, report_date, summary, evaluation)
    validation = validate_media(output, len(frames), len(range(0, len(frames), paths["gif_stride"])))
    output.joinpath("provenance/media_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "purpose": "daily_report_presentation_bundle",
        "report_date": report_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance_quality": "derived_from_fingerprinted",
        "control_commit": git_commit(CONTROL),
        "runtime_commit": git_commit(paths["runtime"]),
        "registry_ids": {"closure": paths["closure"], "artifacts": paths["artifacts"]},
        "selection_policy": "Preregistered evaluation episode index0; never best-of-N.",
        "gif_timing": gif_metadata,
        "copied_sources": copied,
        "derived_files": [
            {"path": relative, "sha256": sha256(output / relative), "bytes": (output / relative).stat().st_size}
            for relative in (
                "demo/episode_000_full_realtime.gif",
                "figures/episode_000_contact_sheet.png",
                "figures/result_overview.png",
            )
        ],
    }
    output.joinpath("provenance/source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums(output)
    write_zip(output, zip_path)
    return {
        "experiment_id": experiment_id,
        "output": str(output),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "files": sum(1 for item in output.rglob("*") if item.is_file()),
        "bytes": sum(item.stat().st_size for item in output.rglob("*") if item.is_file()),
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        build_one("EXP-0010", args.output_root, args.report_date),
        build_one("EXP-0011", args.output_root, args.report_date),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
