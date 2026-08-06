#!/usr/bin/env python3
"""Build paired-evaluation media for EXP-0009 terminal checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARMS = ("baseline", "no_reward_value", "no_reconstruction")
LABELS = {
    "baseline": "Dreamer baseline",
    "no_reward_value": "No reward & value gradients",
    "no_reconstruction": "No reconstruction gradients",
}
CONTACT_LABELS = {
    "baseline": ("Dreamer baseline",),
    "no_reward_value": ("No reward/value", "gradients"),
    "no_reconstruction": ("No reconstruction", "gradients"),
}
COLORS = {
    "baseline": "#167D4A",
    "no_reward_value": "#D18B16",
    "no_reconstruction": "#B23A48",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def read_video(path: Path) -> tuple[list[Image.Image], float]:
  frames = []
  with av.open(str(path)) as container:
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames.extend(frame.to_image().convert("RGB") for frame in container.decode(stream))
  if not frames:
    raise ValueError(f"No frames: {path}")
  return frames, fps


def write_mp4(path: Path, frames: list[Image.Image], fps: float) -> None:
  with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    stream.width, stream.height = frames[0].size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}
    for image in frames:
      for packet in stream.encode(av.VideoFrame.from_image(image)):
        container.mux(packet)
    for packet in stream.encode():
      container.mux(packet)


def text_center(draw, box, text, font, fill):
  left, top, right, bottom = box
  bounds = draw.textbbox((0, 0), text, font=font)
  width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
  draw.text((left + (right - left - width) / 2, top + (bottom - top - height) / 2),
            text, font=font, fill=fill)


def compose(sources, episodes, fps):
  title_font = ImageFont.truetype(FONT_BOLD, 24)
  label_font = ImageFont.truetype(FONT_BOLD, 15)
  small_font = ImageFont.truetype(FONT, 12)
  length = min(len(sources[arm]) for arm in ARMS)
  output = []
  for index in range(length):
    canvas = Image.new("RGB", (1080, 440), "#F4F6F7")
    draw = ImageDraw.Draw(canvas)
    text_center(draw, (0, 8, 1080, 48), "DreamerV3 Reacher Hard: paired checkpoint evaluation",
                title_font, "#17212B")
    for column, arm in enumerate(ARMS):
      x = 24 + column * 352
      draw.rounded_rectangle((x, 58, x + 328, 94), radius=4, fill=COLORS[arm])
      text_center(draw, (x, 58, x + 328, 94), LABELS[arm], label_font, "white")
      frame = sources[arm][index].resize((328, 280), Image.Resampling.NEAREST)
      canvas.paste(frame, (x, 94))
      text_center(draw, (x, 378, x + 328, 410),
                  f"paired eval return {episodes[arm]['episode_return']:.1f}", small_font, "#36424D")
    text_center(draw, (0, 411, 1080, 438),
                f"same eval seed 10000 | t={index / fps:04.1f}s | pixels are visualization only",
                small_font, "#596673")
    output.append(canvas)
  return output


def contact_sheet(sources, episodes, output):
  title_font = ImageFont.truetype(FONT_BOLD, 24)
  label_font = ImageFont.truetype(FONT_BOLD, 14)
  small_font = ImageFont.truetype(FONT, 11)
  length = min(len(sources[arm]) for arm in ARMS)
  picks = np.linspace(0, length - 1, 5, dtype=int)
  canvas = Image.new("RGB", (1080, 690), "#F4F6F7")
  draw = ImageDraw.Draw(canvas)
  text_center(draw, (0, 8, 1080, 48), "Reacher Hard behavior at five episode times",
              title_font, "#17212B")
  for column, pct in enumerate((0, 25, 50, 75, 100)):
    text_center(draw, (190 + column * 174, 50, 350 + column * 174, 78),
                f"{pct}%", small_font, "#596673")
  for row, arm in enumerate(ARMS):
    y = 82 + row * 194
    draw.rounded_rectangle((16, y + 48, 178, y + 112), radius=4, fill=COLORS[arm])
    lines = CONTACT_LABELS[arm]
    if len(lines) == 1:
      text_center(draw, (16, y + 50, 178, y + 86), lines[0], label_font, "white")
    else:
      text_center(draw, (16, y + 45, 178, y + 70), lines[0], label_font, "white")
      text_center(draw, (16, y + 65, 178, y + 90), lines[1], label_font, "white")
    text_center(draw, (16, y + 88, 178, y + 112),
                f"return {episodes[arm]['episode_return']:.1f}", small_font, "white")
    for column, index in enumerate(picks):
      frame = sources[arm][index].resize((160, 160), Image.Resampling.NEAREST)
      canvas.paste(frame, (190 + column * 174, y))
  text_center(draw, (0, 660, 1080, 688),
              "Same initial environment seed; stochastic policy trajectories may diverge.",
              small_font, "#596673")
  canvas.save(output)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  args.output.mkdir(parents=True)
  sources, episodes, fps_values = {}, {}, []
  for arm in ARMS:
    capture = args.eval_root / arm / "capture"
    sources[arm], fps = read_video(capture / "policy.mp4")
    fps_values.append(fps)
    episodes[arm] = json.loads((capture / "episode.json").read_text(encoding="utf-8"))
    if not episodes[arm]["environment_seed_controlled"]:
      raise ValueError(f"{arm}: environment seed was not controlled")
  if len(set(fps_values)) != 1:
    raise ValueError(f"FPS mismatch: {fps_values}")
  fps = fps_values[0]
  frames = compose(sources, episodes, fps)
  write_mp4(args.output / "three_arm_comparison.mp4", frames, fps)
  gif_frames = frames[::4]
  gif_frames[0].save(args.output / "three_arm_comparison.gif", save_all=True,
                     append_images=gif_frames[1:], duration=round(4000 / fps), loop=0,
                     optimize=True, disposal=2)
  contact_sheet(sources, episodes, args.output / "contact_sheet.png")
  manifest = {
      "experiment_id": "EXP-0009",
      "purpose": "presentation_only_paired_initial_state_checkpoint_evaluation",
      "eval_root": str(args.eval_root),
      "frames": len(frames),
      "fps": fps,
      "episodes": episodes,
  }
  for name in ("three_arm_comparison.mp4", "three_arm_comparison.gif", "contact_sheet.png"):
    manifest.setdefault("files", {})[name] = {
        "bytes": (args.output / name).stat().st_size,
        "sha256": sha256(args.output / name),
    }
  (args.output / "manifest.json").write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  (args.output / "README.md").write_text(
      "# EXP-0009 paired behavior showcase\n\n"
      "三臂从各自 terminal checkpoint 在同一 DMC eval seed 10000 初态下录制。"
      "策略有随机性，因此只把视频作为行为展示，不把单局回报作为科学主指标。\n\n"
      "- `three_arm_comparison.mp4`：同步三栏视频\n"
      "- `three_arm_comparison.gif`：日报预览\n"
      "- `contact_sheet.png`：五时刻静态对照\n"
      "- `manifest.json`：来源与 hash\n",
      encoding="utf-8")
  print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
