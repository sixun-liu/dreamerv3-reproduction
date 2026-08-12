#!/usr/bin/env python3
"""Build a normal-speed continuous GIF supplement for EXP-0012."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp")
CONTROL = ROOT / "Code/DreamerV3/dreamerv3-reproduction"
SOURCE = (
    ROOT
    / "Runs/EXP-0012__minecraft-diamond__eval-s10000-3eps__20260812T015527Z"
    / "evaluation/episode_000_preregistered_stride4.mp4"
)
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centered(draw, box, text, font, fill):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text, font=font, fill=fill,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    source_frames = []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate)
        limit = round(args.seconds * source_fps)
        for index, frame in enumerate(container.decode(stream)):
            if index >= limit:
                break
            source_frames.append(frame.to_image().convert("RGB"))
    if len(source_frames) != limit:
        raise ValueError(f"Expected {limit} continuous frames, got {len(source_frames)}")

    stride = 3
    gif_fps = source_fps / stride
    title_font = ImageFont.truetype(str(FONT_BOLD), 25)
    small_font = ImageFont.truetype(str(FONT), 13)
    gif_frames = []
    indices = list(range(0, len(source_frames), stride))
    for index in indices:
        canvas = Image.new("RGB", (448, 510), "#F5F7F8")
        draw = ImageDraw.Draw(canvas)
        centered(draw, (0, 8, 448, 50), "Minecraft episode0", title_font, "#18212B")
        centered(draw, (0, 48, 448, 78), "Continuous opening clip | normal time scale", small_font, "#5D6873")
        canvas.paste(source_frames[index].resize((416, 416), Image.Resampling.NEAREST), (16, 82))
        centered(draw, (0, 481, 448, 508), f"source t={index / source_fps:04.1f}s / {args.seconds:.1f}s | realtime", small_font, "#5D6873")
        gif_frames.append(canvas)
    gif_path = output / "episode_000_opening_20s_realtime.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / gif_fps),
        loop=0,
        optimize=True,
        disposal=2,
    )

    decoded = []
    durations = []
    with Image.open(gif_path) as image:
        for index in range(image.n_frames):
            image.seek(index)
            decoded.append(np.asarray(image.convert("RGB")))
            durations.append(int(image.info.get("duration", 0)))
    dynamic_pairs = sum(bool(np.any(left != right)) for left, right in zip(decoded, decoded[1:]))
    if len(decoded) != len(gif_frames) or dynamic_pairs < len(decoded) - 2:
        raise ValueError("Realtime GIF validation failed")

    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "purpose": "normal_speed_continuous_gif_supplement",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "control_commit": subprocess.check_output(
            ["git", "-C", str(CONTROL), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "selection": "preregistered episode0, continuous first20.0 seconds; not best-of-N",
        "source_fps": source_fps,
        "source_clip_frames": len(source_frames),
        "sample_stride": stride,
        "gif_fps": gif_fps,
        "gif_frames": len(decoded),
        "gif_duration_seconds": sum(durations) / 1000,
        "dynamic_adjacent_pairs": dynamic_pairs,
        "speed_policy": "sample stride and GIF frame rate reduced by the same factor",
        "gif_sha256": sha256(gif_path),
        "gif_bytes": gif_path.stat().st_size,
        "passed": True,
    }
    output.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output.joinpath("README.md").write_text(
        """# EXP-0012 Minecraft 正常速度 GIF 补充

`episode_000_opening_20s_realtime.gif` 来自预先固定的独立评测 episode0，连续截取开头 20 秒。
源视频为 15 FPS；GIF 每三帧取一帧并以 5 FPS 播放，因此总时长与原视频一致，不是加速预览。

这份补充与此前覆盖全时间轴的 `episode_000_timeline.gif` 用途不同：前者适合观察连续行为，
后者适合快速浏览约 10 分钟回合的整体场景变化。两者都不是 best-of-N 选择。
""",
        encoding="utf-8",
    )
    checksums = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(item for item in output.iterdir() if item.is_file() and item.name != "SHA256SUMS")
    ]
    output.joinpath("SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
