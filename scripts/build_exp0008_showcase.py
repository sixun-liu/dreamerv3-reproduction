#!/usr/bin/env python3
"""Build a daily-report showcase from fingerprinted EXP-0008 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp")
CONTROL = ROOT / "dreamerv3-reproduction"
MATRIX = ROOT / "runs/EXP-0008__cheetah-run__five-seed__500k-env__20260805T171000Z"
SOURCE = ROOT / "artifacts/dreamerv3/EXP-0008"
SOURCE_REVIEW = ROOT / "artifacts/dreamerv3/review/EXP-0008-cheetah-five-seed"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

COLORS = {
    "ink": "#17212b",
    "muted": "#596673",
    "paper": "#f4f6f7",
    "dark": "#111820",
    "blue": "#2f6690",
    "teal": "#168a8a",
    "green": "#397a5b",
    "gold": "#d7a928",
    "red": "#b43b3b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_fonts():
    return {
        "title": ImageFont.truetype(str(FONT_BOLD), 38),
        "heading": ImageFont.truetype(str(FONT_BOLD), 25),
        "body": ImageFont.truetype(str(FONT), 19),
        "small": ImageFont.truetype(str(FONT), 15),
        "tiny": ImageFont.truetype(str(FONT), 12),
    }


def text_center(draw, box, value, font, fill):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=font,
        fill=fill,
    )


def read_video(path: Path):
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"Video has no frames: {path}")
    return frames, fps


def write_mp4(path: Path, frames, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(1000)
        )
        stream.width = first.width
        stream.height = first.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_gif(path: Path, frames, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted = [
        frame.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        for frame in frames
    ]
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )


def labeled_seed_frames(source, seed, train_mean, fps):
    fonts = get_fonts()
    output = []
    for index, frame in enumerate(source):
        canvas = Image.new("RGB", (720, 720), COLORS["dark"])
        draw = ImageDraw.Draw(canvas)
        text_center(
            draw,
            (0, 14, 720, 66),
            f"DreamerV3 | Cheetah Run | train seed {seed}",
            fonts["heading"],
            "white",
        )
        canvas.paste(frame.resize((600, 600), Image.Resampling.NEAREST), (60, 66))
        draw.rectangle((0, 666, 720, 720), fill=COLORS["dark"])
        text_center(
            draw,
            (0, 666, 720, 692),
            f"500K env steps | training final-window mean {train_mean:.1f}",
            fonts["small"],
            "#e8edf1",
        )
        text_center(
            draw,
            (0, 691, 720, 716),
            f"independent eval episode | t={index / fps:04.1f}s",
            fonts["tiny"],
            "#b9c5ce",
        )
        output.append(canvas)
    return output


def compose_comparison(sources, means, fps):
    fonts = get_fonts()
    length = min(len(frames) for frames in sources.values())
    positions = {
        0: (48, 126),
        1: (440, 126),
        2: (832, 126),
        3: (244, 492),
        4: (636, 492),
    }
    colors = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["red"],
        COLORS["gold"],
        COLORS["green"],
    ]
    output = []
    for index in range(length):
        canvas = Image.new("RGB", (1280, 900), COLORS["paper"])
        draw = ImageDraw.Draw(canvas)
        text_center(
            draw,
            (0, 12, 1280, 62),
            "DreamerV3 Cheetah Run: five terminal checkpoints",
            fonts["title"],
            COLORS["ink"],
        )
        text_center(
            draw,
            (0, 61, 1280, 96),
            "All checkpoints shown; panels are not selected by displayed behavior",
            fonts["body"],
            COLORS["muted"],
        )
        for seed in range(5):
            x, y = positions[seed]
            draw.rounded_rectangle((x, y, x + 352, y + 42), radius=5, fill=colors[seed])
            text_center(
                draw,
                (x, y, x + 352, y + 42),
                f"seed {seed} | train final-window {means[seed]:.1f}",
                fonts["small"],
                "white",
            )
            frame = sources[seed][index].resize((352, 320), Image.Resampling.NEAREST)
            canvas.paste(frame, (x, y + 42))
        draw.rectangle((0, 858, 1280, 900), fill=COLORS["dark"])
        text_center(
            draw,
            (0, 858, 1280, 898),
            "Independent eval episodes | proprioceptive policy | DMC initial states not strictly paired",
            fonts["small"],
            "#e8edf1",
        )
        output.append(canvas)
    return output


def build_contact_sheet(sources, means, output):
    fonts = get_fonts()
    length = min(len(frames) for frames in sources.values())
    picks = np.linspace(0, length - 1, 5, dtype=int)
    canvas = Image.new("RGB", (1280, 1190), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    text_center(
        draw,
        (0, 12, 1280, 62),
        "Cheetah Run behavior samples across all five checkpoints",
        fonts["title"],
        COLORS["ink"],
    )
    for col, pct in enumerate((0, 25, 50, 75, 100)):
        text_center(
            draw,
            (172 + col * 214, 70, 378 + col * 214, 102),
            f"episode {pct}%",
            fonts["small"],
            COLORS["muted"],
        )
    for seed in range(5):
        y = 110 + seed * 210
        draw.text((24, y + 70), f"seed {seed}", font=fonts["heading"], fill=COLORS["ink"])
        draw.text(
            (24, y + 106),
            f"train mean {means[seed]:.1f}",
            font=fonts["tiny"],
            fill=COLORS["muted"],
        )
        for col, index in enumerate(picks):
            frame = sources[seed][index].resize((206, 206), Image.Resampling.NEAREST)
            canvas.paste(frame, (172 + col * 214, y))
    draw.rectangle((0, 1160, 1280, 1190), fill=COLORS["dark"])
    text_center(
        draw,
        (0, 1160, 1280, 1190),
        "Frames visualize policy behavior only; the agent does not consume pixels.",
        fonts["tiny"],
        "#e8edf1",
    )
    canvas.save(output)


def build_summary_figure(summary, output):
    means = np.asarray(summary["local"]["seed_final_window_means"])
    official = summary["official"]
    curve = Image.open(SOURCE_REVIEW / "cheetah_replication.png").convert("RGB")
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=COLORS["paper"])
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.55, 1), height_ratios=(1.15, 0.85)
    )
    ax_curve = fig.add_subplot(grid[:, 0])
    ax_curve.imshow(curve)
    ax_curve.axis("off")
    ax_curve.set_title("Learning curves and paper reference", fontsize=17, weight="bold")

    ax_bar = fig.add_subplot(grid[0, 1])
    ax_bar.axvspan(
        official["range"][0],
        official["range"][1],
        color=COLORS["gold"],
        alpha=0.20,
        label="official five-seed final range",
    )
    ax_bar.axvline(
        official["mean"], color=COLORS["gold"], linewidth=2, label="official mean"
    )
    ax_bar.barh(
        np.arange(5),
        means,
        color=[
            COLORS["blue"],
            COLORS["teal"],
            COLORS["red"],
            COLORS["gold"],
            COLORS["green"],
        ],
    )
    ax_bar.set_yticks(np.arange(5), [f"seed {seed}" for seed in range(5)])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 720)
    ax_bar.set_xlabel("training final-window score")
    ax_bar.set_title("Terminal-window outcomes", fontsize=17, weight="bold")
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.legend(loc="lower right", fontsize=9)
    for seed, value in enumerate(means):
        ax_bar.text(value + 8, seed, f"{value:.1f}", va="center", fontsize=10)

    ax_text = fig.add_subplot(grid[1, 1])
    ax_text.axis("off")
    ax_text.text(0, 0.98, "Replication verdict", fontsize=17, weight="bold", va="top")
    lines = [
        ("Local five-seed mean", f"{summary['local']['mean']:.1f}"),
        ("Paper reference", f"{official['mean']:.1f} (rounded 614)"),
        ("Relative gap", f"{summary['local']['paper_score_relative_error'] * 100:.1f}%"),
        ("Learning direction", "5 / 5 seeds improved"),
        ("Frozen numerical gate", "FAILED"),
    ]
    for row, (label, value) in enumerate(lines):
        y = 0.78 - row * 0.15
        ax_text.text(0, y, label, fontsize=12, color=COLORS["muted"])
        color = COLORS["red"] if value == "FAILED" else COLORS["ink"]
        ax_text.text(0.56, y, value, fontsize=13, weight="bold", color=color)
    ax_text.text(
        0,
        -0.04,
        "Restricted conclusion: the 2024 author-reimplementation learned\n"
        "Cheetah Run consistently, but did not enter the frozen paper-scale envelope.",
        fontsize=11,
        color=COLORS["ink"],
        linespacing=1.5,
    )
    fig.suptitle(
        "EXP-0008 | DreamerV3 Cheetah Run | 500K environment steps",
        fontsize=23,
        weight="bold",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.05,
        wspace=0.08,
        hspace=0.22,
    )
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_readme(output: Path, eval_root: Path, media):
    readme = f"""# DreamerV3 Cheetah Run 日报展示包（2026-08-06）

本目录来自 `EXP-0008` 五个终点 checkpoint 的独立评测录制。它是展示派生产物，
没有重新训练，也不改变已经冻结的实验裁决。

## 推荐查看顺序

1. `figures/cheetah_showcase_summary.png`：先看论文参考、五 seed 终点统计和裁决。
2. `demo/cheetah_five_seed_comparison.mp4`：同时观察五个终点策略。
3. `demo/cheetah_five_seed_comparison.gif`：适合直接嵌入日报。
4. `figures/behavior_contact_sheet.png`：五个 episode 的分阶段行为截图。
5. `demo/seed_0.mp4` 至 `seed_4.mp4`：逐 seed 查看完整独立评测 episode。

## 可以怎样表述

- 五个训练 seed 的末 30K environment-step episode 均值分别为
  `595.24 / 512.70 / 464.80 / 573.96 / 605.99`。
- 五 seed 聚合为 `550.54`，低于论文官方聚合 `613.61`，也低于官方逐 seed
  终点范围下界 `584.00`；因此预注册数值门失败。
- 五个 seed 的后半程 episode 均值都高于前半程，说明学习方向一致。
- 视频表明终点 checkpoint 可以产生可辨认的奔跑行为，但视频中的一局不能代替
  多 episode 训练统计，也不能用肉眼给五个策略排数值名次。

## 展示边界

- 策略输入是 proprioception，不是画面；画面只供人观察，不能称为像素世界模型输入。
- 五栏使用相同 agent 评测 seed `10000`，但该 2024 runtime 没有把 seed 传给 DMC
  环境构造，所以五个环境初态未严格配对。
- 栏标题中的分数来自训练末窗口均值，不是所展示单局的得分。
- runtime 是 2024 作者重实现谱系，并非 2023 论文的 exact artifact。

## 来源与复核

- 录制 run：`{eval_root}`
- 源训练矩阵：`{MATRIX}`
- 统计摘要：`data/summary.json`
- 文件来源和哈希：`provenance/source_manifest.json`
- 全目录内容校验：`SHA256SUMS`
- 同屏视频：{media['comparison_frames']} 帧，{media['fps']:.1f} FPS，
  约 {media['comparison_frames'] / media['fps']:.1f} 秒。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    freeze = args.eval_root.with_suffix(".freeze")
    completed = args.eval_root.with_suffix(".completed")
    if not freeze.is_file() or not completed.is_file():
        raise FileNotFoundError("Evaluation freeze/completion signal missing")
    frozen = read_json(freeze)
    if frozen.get("train_seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("Showcase builder requires all five seeds")

    for subdir in ("demo", "figures", "data", "provenance"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)

    summary = read_json(SOURCE / "summary.json")
    means = summary["local"]["seed_final_window_means"]
    raw_paths = {}
    sources = {}
    fps_values = []
    for seed in range(5):
        pointer = args.eval_root / f"s{seed:03d}/video_path.txt"
        raw = Path(pointer.read_text(encoding="utf-8").strip())
        if not raw.is_file():
            raise FileNotFoundError(raw)
        raw_paths[seed] = raw
        frames, fps = read_video(raw)
        sources[seed] = frames
        fps_values.append(fps)
    if len({round(value, 6) for value in fps_values}) != 1:
        raise ValueError(f"Input video FPS mismatch: {fps_values}")
    fps = fps_values[0]

    for seed in range(5):
        frames = labeled_seed_frames(sources[seed], seed, means[seed], fps)
        write_mp4(args.output / f"demo/seed_{seed}.mp4", frames, fps)

    comparison = compose_comparison(sources, means, fps)
    write_mp4(args.output / "demo/cheetah_five_seed_comparison.mp4", comparison, fps)
    gif_indices = np.linspace(
        0, len(comparison) - 1, min(80, len(comparison)), dtype=int
    )
    gif_frames = [
        comparison[index].resize((800, 562), Image.Resampling.LANCZOS)
        for index in gif_indices
    ]
    write_gif(args.output / "demo/cheetah_five_seed_comparison.gif", gif_frames, 8.0)
    build_contact_sheet(sources, means, args.output / "figures/behavior_contact_sheet.png")
    build_summary_figure(summary, args.output / "figures/cheetah_showcase_summary.png")

    copies = {
        "data/summary.json": SOURCE / "summary.json",
        "data/per_seed.csv": SOURCE / "per_seed.csv",
        "data/curve.csv": SOURCE / "curve.csv",
        "data/independent_verification.json": SOURCE / "independent_verification.json",
        "figures/cheetah_replication.png": SOURCE_REVIEW / "cheetah_replication.png",
    }
    for destination, source in copies.items():
        shutil.copy2(source, args.output / destination)

    metadata = {
        "schema_version": 1,
        "fps": fps,
        "source_frames": {f"seed_{seed}": len(sources[seed]) for seed in range(5)},
        "comparison_frames": len(comparison),
        "comparison_duration_seconds": len(comparison) / fps,
        "gif_frames": len(gif_frames),
        "gif_duration_seconds": len(gif_frames) / 8.0,
        "source_pixel_std": {
            f"seed_{seed}": float(np.asarray(sources[seed][len(sources[seed]) // 2]).std())
            for seed in range(5)
        },
    }
    (args.output / "provenance/media_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    commit = subprocess.check_output(
        ["git", "-C", str(CONTROL), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0008",
        "provenance_quality": "derived_from_fingerprinted",
        "analysis_commit": commit,
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256(Path(__file__).resolve()),
        "eval_root": str(args.eval_root),
        "eval_freeze": str(freeze),
        "eval_freeze_sha256": sha256(freeze),
        "environment_seed_controlled": False,
        "raw_videos": [
            {
                "train_seed": seed,
                "source_path": str(raw_paths[seed]),
                "sha256": sha256(raw_paths[seed]),
            }
            for seed in range(5)
        ],
        "checkpoints": [
            {
                "train_seed": seed,
                "source_path": str(MATRIX / f"s{seed:03d}/train/checkpoint.ckpt"),
                "sha256": sha256(MATRIX / f"s{seed:03d}/train/checkpoint.ckpt"),
            }
            for seed in range(5)
        ],
        "copied_sources": [
            {
                "destination": destination,
                "source_path": str(source),
                "sha256": sha256(source),
            }
            for destination, source in copies.items()
        ],
    }
    (args.output / "provenance/source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(args.output, args.eval_root, metadata)

    checksum_lines = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
