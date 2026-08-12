#!/usr/bin/env python3
"""Build a provenance-linked daily-report bundle for EXP-0012."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/root/autodl-tmp")
CONTROL = ROOT / "Code/DreamerV3/dreamerv3-reproduction"
RUNTIME = ROOT / "Code/DreamerV3/dreamerv3-runtime-2026-crossdomain"
TRAIN_RUN = (
    ROOT
    / "Runs/EXP-0012__minecraft-diamond__s000__100k-env__20260812T080000Z"
)
EVAL_RUN = (
    ROOT
    / "Runs/EXP-0012__minecraft-diamond__eval-s10000-3eps__20260812T015527Z"
)
TRAIN_ARTIFACT = ROOT / "Artifacts/dreamerv3/EXP-0012/training"
REVIEW = ROOT / "Artifacts/dreamerv3/review/EXP-0012-minecraft-diamond-reduced"
SOURCE_VIDEO = EVAL_RUN / "evaluation/episode_000_preregistered_stride4.mp4"
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
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
        "title": ImageFont.truetype(str(FONT_BOLD), 30),
        "label": ImageFont.truetype(str(FONT_BOLD), 21),
        "body": ImageFont.truetype(str(FONT), 17),
        "small": ImageFont.truetype(str(FONT), 14),
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
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def decode_selected_video(
    path: Path, selected_indices: set[int]
) -> tuple[dict[int, Image.Image], float, int]:
    selected: dict[int, Image.Image] = {}
    decoded_count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            decoded_count += 1
            if index in selected_indices:
                selected[index] = frame.to_image().convert("RGB")
    missing = sorted(selected_indices - set(selected))
    if missing:
        raise ValueError(f"Missing selected frames: {missing[:10]}")
    return selected, fps, decoded_count


def presentation_frame(
    source: Image.Image, source_index: int, source_count: int, source_fps: float
) -> Image.Image:
    fs = fonts()
    canvas = Image.new("RGB", (768, 840), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    centered_text(
        draw,
        (0, 14, 768, 62),
        "DreamerV3 Minecraft Reduced",
        fs["title"],
        COLORS["ink"],
    )
    progress = 100 * source_index / max(1, source_count - 1)
    timestamp = source_index / source_fps
    centered_text(
        draw,
        (0, 61, 768, 96),
        f"Fixed evaluation episode 0 | timeline {progress:05.1f}% | source t={timestamp:05.1f}s",
        fs["small"],
        COLORS["muted"],
    )
    scaled = source.resize((704, 704), Image.Resampling.NEAREST)
    canvas.paste(scaled, (32, 104))
    centered_text(
        draw,
        (0, 810, 768, 838),
        "Uniform timeline sample; presentation only, not best-of-N selection",
        fs["small"],
        COLORS["muted"],
    )
    return canvas


def write_mp4(path: Path, frames: list[Image.Image], fps: float) -> None:
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(1000)
        )
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_gif(path: Path, frames: list[Image.Image], fps: float) -> None:
    resized = [
        frame.resize((384, 420), Image.Resampling.LANCZOS) for frame in frames
    ]
    resized[0].save(
        path,
        save_all=True,
        append_images=resized[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )


def build_contact_sheet(
    output: Path,
    frames: dict[int, Image.Image],
    indices: list[int],
    source_count: int,
    source_fps: float,
) -> None:
    fs = fonts()
    canvas = Image.new("RGB", (1180, 870), COLORS["paper"])
    draw = ImageDraw.Draw(canvas)
    centered_text(
        draw,
        (0, 12, 1180, 62),
        "Minecraft fixed episode 0: six uniformly sampled moments",
        fs["title"],
        COLORS["ink"],
    )
    for position, index in enumerate(indices):
        row, column = divmod(position, 3)
        x = 28 + column * 384
        y = 82 + row * 370
        image = frames[index].resize((352, 352), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        pct = 100 * index / max(1, source_count - 1)
        centered_text(
            draw,
            (x, y + 326, x + 352, y + 352),
            f"{pct:3.0f}% | source {index / source_fps:05.1f}s",
            fs["small"],
            "white",
        )
    centered_text(
        draw,
        (0, 824, 1180, 862),
        "Uniform positions across preregistered episode 0; world seed was not exposed by the runtime.",
        fs["body"],
        COLORS["muted"],
    )
    canvas.save(output)


def build_summary_figure(training: dict, evaluation: dict, output: Path) -> None:
    milestones = training["replay"]["milestones"]
    resource = training["resource"]
    eval_milestones = evaluation["milestones"]
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.0, 0.82), width_ratios=(1.2, 1.0, 1.0),
        hspace=0.38, wspace=0.32,
    )
    fig.suptitle(
        "DreamerV3 Minecraft Reduced | EXP-0012",
        fontsize=25,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        "Exact 100K environment steps | early L1 milestone evidence | not paper-scale Diamond replication",
        ha="center",
        fontsize=13,
        color=COLORS["muted"],
    )

    ax = fig.add_subplot(grid[0, 0])
    items = ["log", "planks", "crafting_table", "wooden_pickaxe", "cobblestone"]
    observed = [milestones[item]["first_global_step"] for item in items]
    plot_values = [value if value is not None else 100000 for value in observed]
    colors = [
        COLORS["green"] if value is not None else COLORS["gray"] for value in observed
    ]
    bars = ax.barh(items, plot_values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlim(0, 105000)
    ax.set_xlabel("First observed environment step")
    ax.set_title("Training inventory milestones", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, observed):
        label = f"step {value:,}" if value is not None else "not observed"
        x = min(bar.get_width() + 1800, 82500) if value is not None else 69000
        ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    eval_items = ["log", "planks", "crafting_table", "wooden_pickaxe"]
    values = [eval_milestones[item]["successful_episodes"] for item in eval_items]
    colors = [COLORS["teal"]] * 3 + [COLORS["gray"]]
    bars = ax.bar(eval_items, values, color=colors, width=0.62)
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Successful episodes out of 3")
    ax.set_title("Independent checkpoint evaluation", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            f"{value}/3",
            ha="center",
            fontweight="bold",
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    returns = [episode["return"] for episode in evaluation["episodes"]]
    resource_lines = [
        ("Training wall time", f"{training['completion']['wall_seconds'] / 60:.1f} min"),
        ("Peak VRAM", f"{resource['gpu']['memory_used_mib_max'] / 1024:.1f} GiB"),
        ("Mean sampled GPU util.", f"{resource['gpu']['gpu_util_percent_mean']:.1f}%"),
        ("Run output", f"{training['completion']['output_bytes'] / 1e9:.2f} GB"),
        ("Eval returns", " / ".join(f"{value:.2f}" for value in returns)),
    ]
    ax.set_title("Run and resource facts", loc="left", fontweight="bold", pad=12)
    for row, (label, value) in enumerate(resource_lines):
        y = 0.88 - row * 0.17
        ax.text(0.02, y, label, color=COLORS["muted"], fontsize=11, transform=ax.transAxes)
        ax.text(
            0.98,
            y,
            value,
            ha="right",
            color=COLORS["ink"],
            fontsize=13,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.plot([0.02, 0.98], [y - 0.055, y - 0.055], color="#D7DCE0", transform=ax.transAxes)

    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    ax.set_title("Bounded verdict", loc="left", fontsize=15, fontweight="bold", pad=8)
    columns = [
        (
            "ENGINEERING",
            "COMPLETE",
            "MineRL/Malmo L0, size50m smoke, exact checkpoint, replay and finite-loss gates passed.",
            COLORS["teal"],
        ),
        (
            "EARLY MILESTONE",
            "L1 OBSERVED",
            "log, planks and crafting table appeared in training and terminal-checkpoint evaluation.",
            COLORS["green"],
        ),
        (
            "BOUNDARY",
            "L2 INCOMPLETE",
            "One cobblestone observation without wooden pickaxe does not establish the L2 chain.",
            COLORS["gold"],
        ),
        (
            "PAPER CLAIM",
            "NOT REPRODUCED",
            "The 100K scaled run is far below the roughly 100M paper-scale Diamond experiment.",
            COLORS["red"],
        ),
    ]
    for index, (header, verdict, detail, color) in enumerate(columns):
        x = 0.01 + index * 0.247
        ax.add_patch(
            plt.Rectangle(
                (x, 0.12), 0.225, 0.70, transform=ax.transAxes,
                facecolor="white", edgecolor="#D7DCE0", linewidth=1.0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x, 0.73), 0.225, 0.09, transform=ax.transAxes,
                facecolor=color, edgecolor=color,
            )
        )
        ax.text(x + 0.012, 0.755, header, color="white", fontsize=9.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, 0.60, verdict, color=color, fontsize=15, fontweight="bold", transform=ax.transAxes)
        words = detail.split()
        lines, line = [], []
        for word in words:
            if len(" ".join(line + [word])) > 31:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        ax.text(x + 0.012, 0.49, "\n".join(lines), color=COLORS["muted"], fontsize=10, va="top", linespacing=1.45, transform=ax.transAxes)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def source_copies() -> dict[str, Path]:
    return {
        "data/training_analysis.json": TRAIN_ARTIFACT / "training_analysis.json",
        "data/evaluation.json": EVAL_RUN / "evaluation/evaluation.json",
        "data/review_summary.json": REVIEW / "review_summary.json",
        "figures/training_curve_and_milestones.png": TRAIN_ARTIFACT / "training_curve_and_milestones.png",
        "demo/episode_000_full_stride4.mp4": SOURCE_VIDEO,
        "demo/episode_000_first_frame.png": EVAL_RUN / "evaluation/episode_000_first_frame.png",
        "protocol/RESULT.md": REVIEW / "RESULT.md",
        "protocol/EXP0012_MINECRAFT_PROTOCOL.md": CONTROL / "docs/reproduction/EXP0012_MINECRAFT_PROTOCOL.md",
        "protocol/exp0012_minecraft_s000_100k_env.yaml": CONTROL / "docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml",
        "provenance/formal.freeze.json": TRAIN_RUN / ".freeze",
        "provenance/formal.completed.json": TRAIN_RUN / ".completed",
        "provenance/formal.integrity.json": TRAIN_RUN / "integrity_formal.json",
        "provenance/evaluation.freeze.json": EVAL_RUN / ".freeze",
        "provenance/evaluation.completed.json": EVAL_RUN / ".completed",
        "provenance/evaluation.SHA256SUMS": EVAL_RUN / "SHA256SUMS",
        "provenance/scripts/analyze_exp0012_minecraft.py": CONTROL / "scripts/analyze_exp0012_minecraft.py",
        "provenance/scripts/evaluate_exp0012_minecraft.py": CONTROL / "scripts/evaluate_exp0012_minecraft.py",
        "provenance/scripts/build_exp0012_daily_bundle.py": Path(__file__).resolve(),
    }


def copy_sources(output: Path) -> list[dict]:
    records = []
    for destination, source in source_copies().items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "destination": destination,
                "source_path": str(source),
                "source_sha256": sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    return records


def write_human_docs(
    output: Path, report_date: str, training: dict, evaluation: dict
) -> None:
    milestones = training["replay"]["milestones"]
    eval_milestones = evaluation["milestones"]
    returns = [episode["return"] for episode in evaluation["episodes"]]
    output.joinpath("README.md").write_text(
        f"""# DreamerV3 Minecraft Reduced 日报材料包（{report_date}）

本目录由 `EXP-0012` 已登记证据派生，没有重新训练或选择最好回合。原始固定评测 episode0 完整
保留；短视频、GIF 和联系表均按全时间轴等距采样，不能替代原视频或主统计。

## 推荐查看顺序

1. `01_日报事实卡.md`：可直接引用的协议、结果与边界。
2. `figures/result_overview.png`：工程、里程碑、评测和资源总览。
3. `figures/training_curve_and_milestones.png`：训练回报与首次里程碑。
4. `demo/episode_000_timeline.gif`：适合日报嵌入的全时段等距预览。
5. `demo/episode_000_timeline.mp4`：约 15 秒的全时段加速展示。
6. `figures/episode_000_contact_sheet.png`：六个固定时间位置的静态截图。
7. `03_日报草稿.md`：可按个人表达修改后提交导师。

## 证据入口

- 原固定评测视频：`demo/episode_000_full_stride4.mp4`
- 训练分析：`data/training_analysis.json`
- 三回合独立评测：`data/evaluation.json`
- 正式受限结论：`protocol/RESULT.md`
- 来源与选择规则：`provenance/source_manifest.json`
- 媒体验证：`provenance/media_validation.json`
- 全目录校验：`SHA256SUMS`

人工视觉确认仍为 `pending`。GIF/MP4 是 presentation-only 派生材料，科学结论仍以
`EVT-0086`、`ART-0071`--`ART-0081` 和原始 JSON 为准。
""",
        encoding="utf-8",
    )
    output.joinpath("01_日报事实卡.md").write_text(
        f"""# EXP-0012 日报事实卡

## 目标与协议

- 任务：Minecraft Diamond，检验作者重实现的真实环境链和 100K 早期物品里程碑。
- runtime：DreamerV3 作者 2026 重实现 `5168475`。
- 配置：size50m、ratio32、单环境、64x64 RGB、agent seed 0。
- 预算：精确 `100,000 environment steps`，未经授权不追加。
- 独立评测：终点 checkpoint、agent seed 10000、固定三回合；runtime 未暴露 world seed。

## 工程完整性

- MineRL/Malmo L0、动作、图像、reward/inventory、reset 与视频检查通过。
- 正式训练精确停在 step 100000；checkpoint SHA256 `b24527f...880d`。
- replay 含 100 个 chunk、100,000 个唯一 transition；8 个完整回合和 1 条预算截断轨迹。
- 40 行 metrics、32 行 warm-up 后 loss；未发现意外 NaN/Inf、OOM 或环境错误。

## 里程碑与评测

| 层级 | 物品 | 训练首次步 | 训练成功轨迹 | 独立评测成功回合 |
|---|---|---:|---:|---:|
| L1 | log | {milestones['log']['first_global_step']} | {milestones['log']['successful_segments']} | {eval_milestones['log']['successful_episodes']}/3 |
| L1 | planks | {milestones['planks']['first_global_step']} | {milestones['planks']['successful_segments']} | {eval_milestones['planks']['successful_episodes']}/3 |
| L1 | crafting_table | {milestones['crafting_table']['first_global_step']} | {milestones['crafting_table']['successful_segments']} | {eval_milestones['crafting_table']['successful_episodes']}/3 |
| L2 | wooden_pickaxe | - | 0 | 0/3 |
| L2 | cobblestone | {milestones['cobblestone']['first_global_step']} | {milestones['cobblestone']['successful_segments']} | 0/3 |
| L3+ | iron/diamond | - | 0 | 0/3 |

三回合回报为 `{returns[0]:.4f} / {returns[1]:.4f} / {returns[2]:.4f}`，均值
`{evaluation['return_mean']:.4f}`。训练和评测均观察到完整 L1；由于 wooden_pickaxe 始终缺失，
一次 cobblestone 观察不能被写成 L2 链完成。

## 资源与结论权限

- 正式训练墙钟：`{training['completion']['wall_seconds'] / 60:.1f} min`。
- 峰值显存：`{training['resource']['gpu']['memory_used_mib_max']:.0f} MiB`。
- 平均采样 GPU 利用率：`{training['resource']['gpu']['gpu_util_percent_mean']:.2f}%`。
- 正式输出：`{training['completion']['output_bytes'] / 1e9:.3f} GB`。
- 裁决：`promising_unresolved / scaled_100k_early_l1_milestone_observed`。

本轮成功完成工程闭环并观察到 100K 早期 L1；L2 未闭合，也没有复现论文约 100M 预算的
Diamond 主结果。
""",
        encoding="utf-8",
    )
    output.joinpath("02_实验过程与结果.md").write_text(
        """# 实验过程与结果

## 1. 先解决真实环境链

为避免把依赖问题误判成算法问题，本轮先建立独立 Python 3.11 环境并固定 MineRL wheel、Java 8、
Xvfb、JAX、NumPy 与 OpenCV 版本。真实 32-step L0 验证了 25 个离散动作、64x64 RGB、有限 reward、
inventory 字段、time-limit 终止、终止后 reset 和动态视频。

## 2. 通过模型与资源 gate

size50m smoke 自然训练到 driver 的 4100-step 边界。原 verifier 将 4096 请求与 4100 自然终点的
差异判为失败，该原始信号被保留；带哈希的离线 reconciliation 证明 checkpoint、replay、有限 loss
和资源门实际通过。这个修复只改变 instrumentation，不改变训练协议或结果。

## 3. 正式 100K 与独立评测

正式 run 使用冻结的 agent seed 0，自然完成精确 100K environment steps。训练后再从固定终点
checkpoint 启动三回合独立评测，agent seed 固定为 10000；Minecraft wrapper 没有 world-seed 接口，
因此世界随机性明确保留为限制。episode0 在看结果前已固定为视频对象，不进行 best-of-N 选择。

## 4. 主要现象

训练很早取得 log、planks 与 crafting table，完整 L1 链在 step 2147 前出现。终点评测同样稳定取得
log/planks，并在 2/3 回合取得 crafting table，说明 L1 不只是训练 replay 中一次偶然记录。

训练中 step 8098 曾记录一个 cobblestone，但 wooden_pickaxe 始终为零，且评测未再观察到
cobblestone。这一顺序可能来自环境初始条件、动作交互或 inventory 语义，当前只能记录为反常观察，
不能用常识补全缺失的中间步骤，也不能声称 L2 完成。

## 5. 展示材料如何生成

原 episode0 视频有 9001 个 64x64 帧，stride 4、15 FPS，约 10 分钟。日报短视频从 0% 到 100%
等距选取 120 帧，以 8 FPS 播放约 15 秒；GIF 等距取 80 帧，联系表固定取六个位置。这些派生材料
覆盖整条时间轴，不挑选回报高光，但时间压缩后不能用于逐动作机制判断。
""",
        encoding="utf-8",
    )
    output.joinpath("03_日报草稿.md").write_text(
        f"""# DreamerV3 复现日报草稿（{report_date}）

今天完成了 DreamerV3 在 Minecraft Diamond 任务上的缩减预算复现闭环。实验采用作者 2026
重实现，配置为 size50m、replay ratio 32、单环境和 agent seed 0；目标是在 100K environment
steps 的硬上限内验证真实 MineRL/Malmo 环境、模型训练和早期物品里程碑，而不是复现论文约
100M 预算的 Diamond 最终成绩。

工程上先完成了 MineRL/Malmo、Java/Xvfb、动作、图像、reward/inventory、reset 和视频的真实 L0，
再通过 size50m 模型 smoke 与资源 gate。正式训练自然结束在精确 step 100000，checkpoint、replay、
配置、有限 loss 和资源记录均通过完整性检查。训练耗时约
{training['completion']['wall_seconds'] / 60:.1f} 分钟，峰值显存
{training['resource']['gpu']['memory_used_mib_max']:.0f} MiB，输出约
{training['completion']['output_bytes'] / 1e9:.2f} GB。

训练中 log、planks 和 crafting table 首次出现在 step 2057、2103 和 2147，说明完整 L1 链在
100K 预算内出现。终点 checkpoint 的三回合独立评测中，log 和 planks 均为 3/3，crafting table
为 2/3，三局回报为 {returns[0]:.4f}、{returns[1]:.4f} 和 {returns[2]:.4f}。训练中还在一个轨迹
观察到 cobblestone，但 wooden_pickaxe 从未出现，评测也未复现 cobblestone，因此不能称为 L2 链
完成，更不能外推为 Diamond 能力。

为了方便展示，我基于预先固定的评测第 0 回合制作了全时段等距采样短视频、GIF、六时刻联系表和
结果总览图，同时保留原始约 10 分钟视频。所有派生材料都附有来源路径、SHA256、构建脚本和媒体
解码验证；没有重新训练，也没有按回报挑选最好回合。

本轮较准确的结论是：DreamerV3 作者重实现在本机完成了 Minecraft 的工程闭环，并在单 seed、
100K 缩减预算下取得可复核的早期 L1 里程碑；L2 链和论文尺度 Diamond 结果仍未解决。下一步先
人工审查曲线和视频，再优先考虑用已有 replay 分析 wooden-pickaxe 瓶颈，而不是立即扩大训练预算。

agent 辅助完成环境核验、实验执行、统计与材料生成；用户主导实验目标、预算授权和结论解释。
""",
        encoding="utf-8",
    )
    output.joinpath("04_口径与常见问答.md").write_text(
        """# 口径与常见问答

## 这次复现了论文的 Minecraft Diamond 结果吗？

没有。论文结果约为 100M environment-step 量级，本轮只有 100K，并且模型规模与训练比率也做了
兼容性缩减。准确说法是“真实环境和 100K 早期 L1 里程碑复现成功”。

## 为什么看到 cobblestone 仍不能说 L2 成功？

预定义的 L2 链包含 wooden_pickaxe 和 cobblestone。训练只观察到一次 cobblestone，wooden_pickaxe
始终为零，独立评测两者均为零。因此只能报告单个 inventory 观察，不能补写不存在的中间链条。

## 三回合评测为什么和训练回报不同？

它们使用相同终点 checkpoint，但评测是固定 agent seed 的独立新环境回合；Minecraft world seed
不可控。评测用于检查终点策略是否能重复产生早期物品，不与训练曲线逐样本等价。

## GIF 和短视频是最好的一局吗？

不是。实验前就固定了 episode0，短媒体又从该完整视频的全时间轴等距采样。它们适合展示，不用于
替代三回合统计，也不能在时间压缩后判断连续动作因果。

## 平均 GPU 利用率为什么只有约 9%？

Minecraft 单环境运行包含 Java/Malmo 仿真、图像传输和环境交互，CPU/环境等待占比较高；JAX 更新
会形成短时 GPU 峰值。显存接近 24 GiB 表示模型已驻留，不代表计算单元持续满载。

## 下一步最值得做什么？

优先利用已保存的 100K replay 统计动作、inventory 转移和失败轨迹，定位 wooden-pickaxe 前的瓶颈。
只有离线证据表明主要限制是探索时长而不是环境语义或动作交互后，扩大预算才更有信息价值。
""",
        encoding="utf-8",
    )


def validate_media(output: Path, expected_mp4_frames: int, expected_gif_frames: int) -> dict:
    mp4_path = output / "demo/episode_000_timeline.mp4"
    decoded = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            decoded.append(np.asarray(frame.to_image().convert("RGB")))
    if len(decoded) != expected_mp4_frames:
        raise ValueError(f"Derived MP4 frame mismatch: {len(decoded)}")
    dynamic_pairs = sum(
        bool(np.any(left != right)) for left, right in zip(decoded, decoded[1:])
    )
    if dynamic_pairs < len(decoded) - 2:
        raise ValueError(f"Derived MP4 is unexpectedly static: {dynamic_pairs}")
    gif_path = output / "demo/episode_000_timeline.gif"
    with Image.open(gif_path) as image:
        gif_frames = int(getattr(image, "n_frames", 1))
    if gif_frames != expected_gif_frames:
        raise ValueError(f"Derived GIF frame mismatch: {gif_frames}")
    checks = {}
    for relative in (
        "figures/result_overview.png",
        "figures/episode_000_contact_sheet.png",
    ):
        with Image.open(output / relative) as image:
            array = np.asarray(image.convert("RGB"))
            checks[relative] = {
                "width": image.width,
                "height": image.height,
                "pixel_std": float(array.std()),
            }
            if image.width < 1000 or image.height < 700 or array.std() < 5:
                raise ValueError(f"Invalid review image: {relative}")
    return {
        "schema_version": 1,
        "derived_mp4": {
            "decoded_frames": len(decoded),
            "fps": fps,
            "duration_seconds": len(decoded) / fps,
            "dynamic_adjacent_pairs": dynamic_pairs,
        },
        "derived_gif": {"frames": gif_frames},
        "figures": checks,
        "passed": True,
    }


def write_manifest(
    output: Path,
    report_date: str,
    copied_sources: list[dict],
    source_frames: int,
    source_fps: float,
    mp4_indices: list[int],
    gif_indices: list[int],
    contact_indices: list[int],
) -> None:
    derived = []
    for relative in (
        "figures/result_overview.png",
        "figures/episode_000_contact_sheet.png",
        "demo/episode_000_timeline.mp4",
        "demo/episode_000_timeline.gif",
    ):
        path = output / relative
        derived.append(
            {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "purpose": "daily_report_presentation_bundle",
        "report_date": report_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance_quality": "derived_from_fingerprinted",
        "control_commit": git_commit(CONTROL),
        "runtime_commit": git_commit(RUNTIME),
        "registry_ids": {
            "closure": "EVT-0086",
            "artifacts": [f"ART-{value:04d}" for value in range(71, 82)],
        },
        "scientific_scope": (
            "Minecraft Diamond size50m, one agent seed, exact100K environment steps, "
            "three terminal-checkpoint episodes; early L1 evidence only"
        ),
        "selection_policy": {
            "source": "preregistered evaluation episode index 0; never best-of-N",
            "source_frames": source_frames,
            "source_fps": source_fps,
            "mp4": "120 uniformly spaced source frames played at 8 FPS",
            "mp4_source_indices": mp4_indices,
            "gif": "80 uniformly spaced source frames played at 8 FPS",
            "gif_source_indices": gif_indices,
            "contact_sheet_source_indices": contact_indices,
        },
        "copied_sources": copied_sources,
        "derived_files": derived,
    }
    path = output / "provenance/source_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zip(output: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            archive.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.zip_path = args.zip_path.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.zip_path.exists() or args.zip_path.with_suffix(
        args.zip_path.suffix + ".sha256"
    ).exists():
        raise FileExistsError(f"Refusing to overwrite ZIP: {args.zip_path}")

    training = read_json(TRAIN_ARTIFACT / "training_analysis.json")
    evaluation = read_json(EVAL_RUN / "evaluation/evaluation.json")
    if not training["integrity"]["passed"]:
        raise ValueError("EXP-0012 formal integrity gate did not pass")
    if training["integrity"]["checkpoint_step"] != 100000:
        raise ValueError("EXP-0012 terminal checkpoint is not exact100K")
    if evaluation["episode_count"] != 3 or evaluation["video_selection"] != (
        "episode index 0 preregistered; not best-of-N"
    ):
        raise ValueError("EXP-0012 fixed evaluation protocol mismatch")

    for subdir in ("data", "demo", "figures", "protocol", "provenance"):
        (args.output / subdir).mkdir(parents=True, exist_ok=False)
    copied_sources = copy_sources(args.output)

    source_count = int(evaluation["video_frame_count"])
    mp4_indices = np.linspace(0, source_count - 1, 120, dtype=int).tolist()
    gif_indices = np.linspace(0, source_count - 1, 80, dtype=int).tolist()
    contact_indices = np.linspace(0, source_count - 1, 6, dtype=int).tolist()
    selected, source_fps, decoded_count = decode_selected_video(
        SOURCE_VIDEO, set(mp4_indices + gif_indices + contact_indices)
    )
    if decoded_count != source_count:
        raise ValueError(f"Source video frame mismatch: {decoded_count} != {source_count}")

    mp4_frames = [
        presentation_frame(selected[index], index, source_count, source_fps)
        for index in mp4_indices
    ]
    gif_frames = [
        presentation_frame(selected[index], index, source_count, source_fps)
        for index in gif_indices
    ]
    write_mp4(args.output / "demo/episode_000_timeline.mp4", mp4_frames, 8.0)
    write_gif(args.output / "demo/episode_000_timeline.gif", gif_frames, 8.0)
    build_contact_sheet(
        args.output / "figures/episode_000_contact_sheet.png",
        selected,
        contact_indices,
        source_count,
        source_fps,
    )
    build_summary_figure(training, evaluation, args.output / "figures/result_overview.png")
    write_human_docs(args.output, args.report_date, training, evaluation)

    validation = validate_media(args.output, len(mp4_indices), len(gif_indices))
    (args.output / "provenance/media_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        args.output,
        args.report_date,
        copied_sources,
        source_count,
        source_fps,
        mp4_indices,
        gif_indices,
        contact_indices,
    )
    write_checksums(args.output)
    write_zip(args.output, args.zip_path)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "zip": str(args.zip_path),
                "zip_sha256": sha256(args.zip_path),
                "files": sum(1 for item in args.output.rglob("*") if item.is_file()),
                "bytes": sum(
                    item.stat().st_size for item in args.output.rglob("*") if item.is_file()
                ),
                "media_validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
