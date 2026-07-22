#!/usr/bin/env python3
"""Build a presentation package from fingerprinted EXP-0006 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/autodl-tmp")
MATRIX = ROOT / "runs/EXP-0006__walker-kl__matrix-2seed__20260721T142500Z"
SOURCE_ARTIFACTS = ROOT / "artifacts/dreamerv3/EXP-0006"
SOURCE_REVIEW = ROOT / "artifacts/dreamerv3/review/EXP-0006-walker-kl-three-arm"
EXP5_ARTIFACTS = ROOT / "artifacts/dreamerv3/EXP-0005"
EXP5_REVIEW = ROOT / "artifacts/dreamerv3/review/EXP-0005-runtime-2411-walker"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

COLORS = {
    "ink": "#16212b",
    "muted": "#5c6773",
    "paper": "#f5f7f8",
    "baseline": "#2f6690",
    "e1": "#bf5b3f",
    "p4": "#397a5b",
    "gold": "#d7a928",
    "teal": "#168a8a",
    "red": "#b43b3b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fonts():
    return {
        "title": ImageFont.truetype(str(FONT_BOLD), 42),
        "heading": ImageFont.truetype(str(FONT_BOLD), 27),
        "body": ImageFont.truetype(str(FONT), 20),
        "small": ImageFont.truetype(str(FONT), 15),
        "tiny": ImageFont.truetype(str(FONT), 12),
    }


def video_frames(path: Path, limit: int | None = None, start: int = 0):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = []
        for index, frame in enumerate(container.decode(stream)):
            if index < start:
                continue
            frames.append(frame.to_image().convert("RGB"))
            if limit is not None and len(frames) >= limit:
                break
    return frames, fps


def write_mp4(path: Path, frames, fps: float = 10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
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


def save_gif(path: Path, frames, fps: float = 5.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = round(1000 / fps)
    paletted = [frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT) for frame in frames]
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def fit_center(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#111820")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def text_center(draw, box, text, font, fill):
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


def draw_panel_header(draw, x, width, title, subtitle, color, f):
    draw.rounded_rectangle((x, 76, x + width, 142), radius=6, fill=color)
    text_center(draw, (x, 76, x + width, 110), title, f["heading"], "white")
    text_center(draw, (x, 108, x + width, 139), subtitle, f["small"], "white")


def render_three_arm_video(video_paths, summary, output: Path):
    f = fonts()
    arms = ("baseline", "e1", "p4")
    labels = {
        "baseline": ("Baseline", "free nats=1, rep scale=0.1"),
        "e1": ("E1", "free nats=0, rep scale=0.1"),
        "p4": ("P4 reconstructed", "free nats=0, rep scale=1.0"),
    }
    colors = {arm: COLORS[arm] for arm in arms}
    eval_means = {
        run["arm"]: run["eval"]["mean"]
        for run in summary["runs"]
        if run["seed"] == 1
    }
    sources = {}
    for arm in arms:
        sources[arm], fps = video_frames(video_paths[arm], limit=251)
    length = min(map(len, sources.values()))
    composed = []
    for index in range(length):
        canvas = Image.new("RGB", (1280, 600), COLORS["paper"])
        draw = ImageDraw.Draw(canvas)
        text_center(
            draw,
            (0, 12, 1280, 65),
            "DreamerV3 Walker: controlled policy comparison",
            f["title"],
            COLORS["ink"],
        )
        for col, arm in enumerate(arms):
            x = 32 + col * 416
            draw_panel_header(draw, x, 384, labels[arm][0], labels[arm][1], colors[arm], f)
            frame = sources[arm][index].resize((384, 384), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x, 152))
            draw.rectangle((x, 502, x + 384, 536), fill="#101820")
            score = f"64-episode eval mean: {eval_means[arm]:.1f}"
            text_center(draw, (x, 502, x + 384, 536), score, f["small"], "white")
        draw.rectangle((0, 552, 1280, 600), fill="#101820")
        footer = "Train seed s001 | Eval seed 10000 | same initial condition | one episode"
        text_center(draw, (0, 552, 1280, 600), footer, f["small"], "#e8edf1")
        composed.append(canvas)
    write_mp4(output / "demo/three_arm_comparison.mp4", composed, fps=fps)
    gif_frames = composed[::2][:50]
    save_gif(output / "demo/three_arm_comparison.gif", gif_frames, fps=fps / 2)
    contact = Image.new("RGB", (1280, 850), COLORS["paper"])
    d = ImageDraw.Draw(contact)
    text_center(d, (0, 10, 1280, 65), "Controlled behavior samples", f["title"], COLORS["ink"])
    picks = [0, length // 4, length // 2, 3 * length // 4, length - 1]
    for row, arm in enumerate(arms):
        d.text((24, 88 + row * 244), labels[arm][0], font=f["heading"], fill=colors[arm])
        for col, idx in enumerate(picks):
            frame = sources[arm][idx].resize((220, 220), Image.Resampling.LANCZOS)
            contact.paste(frame, (160 + col * 224, 78 + row * 244))
    contact.save(output / "figures/behavior_contact_sheet.png")
    return {"fps": fps, "frames": length, "duration_seconds": length / fps}


def render_baseline_demo(video_path: Path, output: Path):
    f = fonts()
    source, fps = video_frames(video_path, limit=301)
    composed = []
    for index, frame in enumerate(source):
        canvas = Image.new("RGB", (720, 720), "#101820")
        draw = ImageDraw.Draw(canvas)
        text_center(draw, (0, 18, 720, 70), "DreamerV3 | DMC Walker Walk", f["heading"], "white")
        scaled = frame.resize((640, 640), Image.Resampling.LANCZOS)
        canvas.paste(scaled, (40, 70))
        draw.rectangle((0, 672, 720, 720), fill="#101820")
        text_center(
            draw,
            (0, 672, 720, 716),
            f"Baseline checkpoint | 500K environment steps | t={index / fps:04.1f}s",
            f["small"],
            "#e7edf1",
        )
        composed.append(canvas)
    write_mp4(output / "demo/baseline_s001.mp4", composed, fps=fps)
    save_gif(output / "demo/baseline_opening.gif", composed[::2][:40], fps=fps / 2)


def latest_training_video(arm: str, seed: int, index: int) -> Path:
    video_dir = MATRIX / arm / f"s{seed:03d}/train/scope/epstats-policy_log-image.mp4"
    files = sorted(video_dir.glob("*.mp4"))
    return files[index]


def render_learning_progression(output: Path):
    f = fonts()
    paths = [
        latest_training_video("baseline", 0, 0),
        latest_training_video("baseline", 0, len(list((MATRIX / "baseline/s000/train/scope/epstats-policy_log-image.mp4").glob("*.mp4"))) // 2),
        latest_training_video("baseline", 0, -1),
    ]
    titles = ("Early | 18K steps", "Middle | ~250K steps", "Late | 489K steps")
    sources = []
    for path in paths:
        frames, fps = video_frames(path, limit=121)
        sources.append(frames)
    length = min(map(len, sources))
    composed = []
    for index in range(length):
        canvas = Image.new("RGB", (1050, 430), COLORS["paper"])
        draw = ImageDraw.Draw(canvas)
        text_center(draw, (0, 8, 1050, 58), "One seed learning to walk", f["title"], COLORS["ink"])
        for col, frames in enumerate(sources):
            x = 24 + col * 342
            draw.rounded_rectangle((x, 66, x + 318, 110), radius=5, fill=COLORS["baseline"])
            text_center(draw, (x, 66, x + 318, 110), titles[col], f["small"], "white")
            frame = frames[index].resize((318, 318), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x, 110))
        composed.append(canvas)
    save_gif(output / "demo/baseline_learning_progression.gif", composed[::2], fps=fps / 2)
    still = composed[min(length - 1, int(5 * fps))]
    still.save(output / "figures/baseline_learning_progression.png")
    return [str(path) for path in paths]


def setup_slide_ax(title: str):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.65, 8.35, title, fontsize=28, weight="bold", color=COLORS["ink"])
    return fig, ax


def box(ax, xy, width, height, text, color, fontsize=16, text_color="white"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color,
        edgecolor="none",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        weight="bold",
    )


def arrow(ax, start, end, color=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color or COLORS["muted"],
        )
    )


def render_world_model_pipeline(output: Path):
    fig, ax = setup_slide_ax("DreamerV3: learn a world model, then improve behavior in imagination")
    box(ax, (0.7, 5.7), 2.1, 1.05, "Observation\nx_t", COLORS["teal"], fontsize=14)
    box(ax, (0.7, 3.7), 2.1, 1.05, "Previous action\na_{t-1}", COLORS["gold"], fontsize=14, text_color=COLORS["ink"])
    box(ax, (3.6, 5.7), 2.2, 1.05, "Encoder\ne_t", COLORS["baseline"])
    box(ax, (6.6, 4.6), 2.7, 2.3, "RSSM latent state\nh_t deterministic\nz_t stochastic", "#334b5e")
    box(ax, (10.2, 6.0), 2.2, 0.9, "Decoder", COLORS["teal"])
    box(ax, (10.2, 4.85), 2.2, 0.9, "Reward head", COLORS["gold"], text_color=COLORS["ink"])
    box(ax, (10.2, 3.7), 2.2, 0.9, "Continue head", COLORS["p4"], fontsize=14)
    box(ax, (13.2, 4.6), 2.1, 2.3, "Learned\nworld model", "#27313b")
    arrow(ax, (2.8, 6.22), (3.6, 6.22))
    arrow(ax, (5.8, 6.22), (6.6, 6.1))
    arrow(ax, (2.8, 4.22), (6.6, 5.05))
    arrow(ax, (9.3, 6.15), (10.2, 6.45))
    arrow(ax, (9.3, 5.75), (10.2, 5.30))
    arrow(ax, (9.3, 5.30), (10.2, 4.15))
    arrow(ax, (12.4, 6.45), (13.2, 6.25))
    arrow(ax, (12.4, 5.30), (13.2, 5.75))
    arrow(ax, (12.4, 4.15), (13.2, 5.25))

    ax.plot([0.65, 15.35], [3.05, 3.05], color="#cbd2d8", linewidth=1.5)
    ax.text(0.7, 2.65, "IMAGINATION", fontsize=13, weight="bold", color=COLORS["muted"])
    box(ax, (2.2, 1.0), 2.7, 1.25, "Imagine latent\ntrajectories", "#334b5e")
    box(ax, (6.4, 1.0), 2.2, 1.25, "Actor\nchoose actions", COLORS["e1"], fontsize=14)
    box(ax, (10.1, 1.0), 2.2, 1.25, "Critic\nestimate returns", COLORS["baseline"], fontsize=12)
    box(ax, (13.2, 1.0), 2.1, 1.25, "Act in the\nreal world", COLORS["p4"], fontsize=14)
    arrow(ax, (4.9, 1.62), (6.4, 1.62))
    arrow(ax, (8.6, 1.62), (10.1, 1.62))
    arrow(ax, (12.3, 1.62), (13.2, 1.62))
    arrow(ax, (14.2, 2.25), (14.2, 3.65), COLORS["p4"])
    ax.text(
        0.7,
        0.35,
        "This experiment uses proprioceptive observations; the camera video is a human-readable policy trace, not the model input.",
        fontsize=13,
        color=COLORS["muted"],
    )
    fig.savefig(output / "figures/world_model_pipeline.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_mechanism_chain(summary, output: Path):
    runs = {(item["arm"], item["seed"]): item for item in summary["runs"]}
    p4_kl = [runs[("p4", s)]["mechanism"]["late"]["kl_raw"] / runs[("e1", s)]["mechanism"]["late"]["kl_raw"] for s in (0, 1)]
    p4_rec = [runs[("p4", s)]["mechanism"]["late"]["reconstruction"] / runs[("e1", s)]["mechanism"]["late"]["reconstruction"] for s in (0, 1)]
    p4_score = [(runs[("p4", s)]["train"]["final_30k"]["mean"] / runs[("e1", s)]["train"]["final_30k"]["mean"] - 1) * 100 for s in (0, 1)]
    fig, ax = setup_slide_ax("What EXP-0006 changes, and what the two seeds actually show")
    box(ax, (0.7, 5.3), 3.0, 1.5, "Baseline\nfree nats=1\nrep scale=0.1", COLORS["baseline"])
    box(ax, (5.0, 5.3), 3.0, 1.5, "E1\nfree nats=0\nrep scale=0.1", COLORS["e1"])
    box(ax, (9.3, 5.3), 3.0, 1.5, "P4 reconstructed\nfree nats=0\nrep scale=1.0", COLORS["p4"])
    arrow(ax, (3.7, 6.05), (5.0, 6.05), COLORS["e1"])
    arrow(ax, (8.0, 6.05), (9.3, 6.05), COLORS["p4"])
    ax.text(4.35, 7.05, "remove free bits", ha="center", fontsize=12, color=COLORS["e1"], weight="bold")
    ax.text(8.65, 7.05, "10x representation KL pressure", ha="center", fontsize=12, color=COLORS["p4"], weight="bold")

    ax.text(0.8, 4.55, "Observed", fontsize=15, color=COLORS["muted"], weight="bold")
    box(ax, (0.8, 2.85), 3.8, 1.35, "E1 raw-KL direction\nnot consistent\nacross two seeds\nscore lower in both", "#734b40", fontsize=12)
    box(
        ax,
        (5.0, 2.85),
        3.8,
        1.35,
        f"P4 KL: {p4_kl[0]*100:.1f}% / {p4_kl[1]*100:.1f}% of E1\nReconstruction:\n{p4_rec[0]:.2f}x / {p4_rec[1]:.2f}x",
        "#2e6650",
        fontsize=12,
    )
    box(
        ax,
        (9.2, 2.85),
        3.8,
        1.35,
        f"P4 final score vs E1\n{p4_score[0]:.1f}% / {p4_score[1]:.1f}%\nNo strict collapse trigger",
        "#344b5e",
        fontsize=12,
    )
    arrow(ax, (4.6, 3.52), (5.0, 3.52))
    arrow(ax, (8.8, 3.52), (9.2, 3.52))

    ax.text(0.8, 2.22, "Local mechanism supported by this task", fontsize=15, color=COLORS["muted"], weight="bold")
    box(ax, (0.8, 1.05), 2.7, 1.0, "stronger KL\npressure", COLORS["p4"], fontsize=13)
    box(ax, (4.1, 1.05), 2.7, 1.0, "posterior closer\nto prior", "#344b5e", fontsize=13)
    box(ax, (7.4, 1.05), 2.7, 1.0, "less usable\ndetail", "#5b6670", fontsize=13)
    box(ax, (10.7, 1.05), 3.2, 1.0, "worse modeling\nand control", COLORS["red"], fontsize=13)
    arrow(ax, (3.5, 1.55), (4.1, 1.55))
    arrow(ax, (6.8, 1.55), (7.4, 1.55))
    arrow(ax, (10.1, 1.55), (10.7, 1.55))
    ax.text(
        0.8,
        0.45,
        "Scope: two seeds, one DMC task, reconstructed P4 semantics. This is not a numerical replication of paper Figure 6/17.",
        fontsize=13,
        color=COLORS["muted"],
    )
    fig.savefig(output / "figures/mechanism_causal_chain.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_headline(summary, output: Path):
    runs = {(item["arm"], item["seed"]): item for item in summary["runs"]}
    fig, ax = setup_slide_ax("DreamerV3 reproduction: from running the code to testing a mechanism")
    ax.text(0.7, 7.55, "DMC Walker Walk", fontsize=16, color=COLORS["muted"], weight="bold")
    box(ax, (0.7, 5.0), 4.4, 2.1, "6 complete runs\n3 arms x 2 seeds\n500K env steps each", COLORS["baseline"], fontsize=17)
    box(ax, (5.8, 5.0), 4.4, 2.1, "Baseline final-30K\n903.4 / 833.5\nusable walking policies", COLORS["teal"], fontsize=17)
    box(ax, (10.9, 5.0), 4.4, 2.1, "P4 raw KL\n0.093 / 0.048\nlarge representation shift", COLORS["p4"], fontsize=17)
    ax.text(0.7, 4.15, "MAIN FINDING", fontsize=15, color=COLORS["muted"], weight="bold")
    ax.text(
        0.7,
        3.35,
        "Forcing the posterior too close to the prior reduced KL,",
        fontsize=25,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.7,
        2.75,
        "but reconstruction and control performance became worse.",
        fontsize=25,
        weight="bold",
        color=COLORS["red"],
    )
    baseline_mean = sum(runs[("baseline", s)]["train"]["final_30k"]["mean"] for s in (0, 1)) / 2
    p4_mean = sum(runs[("p4", s)]["train"]["final_30k"]["mean"] for s in (0, 1)) / 2
    ax.text(
        0.7,
        1.62,
        f"Mean final-30K score: {baseline_mean:.1f} baseline  ->  {p4_mean:.1f} P4 reconstructed",
        fontsize=18,
        color=COLORS["ink"],
    )
    ax.text(
        0.7,
        0.65,
        "Evidence level: controlled local mechanism study; cross-task generalization remains open.",
        fontsize=14,
        color=COLORS["muted"],
    )
    fig.savefig(output / "figures/headline_summary.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_reproduction_status(output: Path):
    summary = load_json(EXP5_ARTIFACTS / "summary.json")
    official = summary["official_final"]["mean"]
    local = summary["final_30k"]["mean"]
    delta = (local / official - 1) * 100
    checkpoint = summary["checkpoints"]["250000"]
    fig, ax = setup_slide_ax("Paper-result check: terminal value aligns, early learning speed does not")
    ax.text(0.7, 7.55, "EXP-0005 | DMC Walker Walk | author lineage 2411f7d", fontsize=16, color=COLORS["muted"], weight="bold")
    box(ax, (0.7, 5.25), 4.3, 1.7, f"Official final\n{official:.2f}\n5-seed aggregate", "#27313b", fontsize=17)
    box(ax, (5.75, 5.25), 4.3, 1.7, f"Local final-30K\n{local:.2f}\none complete run", COLORS["baseline"], fontsize=17)
    box(ax, (10.8, 5.25), 4.3, 1.7, f"Terminal delta\n{delta:+.2f}%\nfinal gate passed", COLORS["p4"], fontsize=17)
    ax.text(0.7, 4.35, "EARLY-CURVE CHECK AT 250K ENVIRONMENT STEPS", fontsize=15, color=COLORS["muted"], weight="bold")
    box(ax, (0.7, 2.45), 5.2, 1.45, f"Local median\n{checkpoint['median']:.2f}", COLORS["e1"], fontsize=18)
    box(
        ax,
        (6.65, 2.45),
        5.2,
        1.45,
        f"Official seed range\n{checkpoint['official_min']:.2f} - {checkpoint['official_max']:.2f}",
        "#5b6670",
        fontsize=18,
    )
    box(ax, (12.6, 2.45), 2.5, 1.45, "Outside range\nearly gate failed", COLORS["red"], fontsize=14)
    ax.text(0.7, 1.55, "Verdict: partial numerical reproduction, not stable replication", fontsize=23, weight="bold", color=COLORS["ink"])
    ax.text(
        0.7,
        0.55,
        "One local run and uncontrolled DMC environment randomness support lineage plausibility, not a multi-seed replication claim.",
        fontsize=13,
        color=COLORS["muted"],
    )
    fig.savefig(output / "figures/reproduction_status.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def copy_sources(output: Path):
    shutil.copy2(SOURCE_REVIEW / "mechanism_timeseries.png", output / "figures/mechanism_timeseries.png")
    shutil.copy2(SOURCE_REVIEW / "representation_and_performance.png", output / "figures/representation_and_performance.png")
    shutil.copy2(SOURCE_ARTIFACTS / "summary.json", output / "data/summary.json")
    shutil.copy2(SOURCE_ARTIFACTS / "paired_comparisons.csv", output / "data/paired_comparisons.csv")
    shutil.copy2(SOURCE_ARTIFACTS / "per_run_summary.csv", output / "data/per_run_summary.csv")
    shutil.copy2(EXP5_REVIEW / "curve_comparison.png", output / "figures/paper_curve_comparison.png")
    shutil.copy2(EXP5_ARTIFACTS / "summary.json", output / "data/exp0005_summary.json")
    shutil.copy2(EXP5_ARTIFACTS / "curve.csv", output / "data/exp0005_curve.csv")


def main():
    args = parse_args()
    output = args.output.resolve()
    for directory in ("figures", "demo", "data", "provenance"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if not (args.eval_root / ".completed").is_file():
        raise SystemExit(f"Missing completed eval signal: {args.eval_root / '.completed'}")

    summary = load_json(SOURCE_ARTIFACTS / "summary.json")
    copy_sources(output)
    video_paths = {
        arm: Path((args.eval_root / arm / "video_path.txt").read_text().strip())
        for arm in ("baseline", "e1", "p4")
    }
    for arm, path in video_paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {arm} video: {path}")
        shutil.copy2(path, output / f"demo/{arm}_s001_raw.mp4")

    comparison_meta = render_three_arm_video(video_paths, summary, output)
    render_baseline_demo(video_paths["baseline"], output)
    progression_sources = render_learning_progression(output)
    render_world_model_pipeline(output)
    render_mechanism_chain(summary, output)
    render_headline(summary, output)
    render_reproduction_status(output)

    freeze = load_json(args.eval_root.with_suffix(".freeze"))
    checkpoint_provenance = {
        "schema_version": 1,
        "experiment_id": "EXP-0006",
        "control_commit_at_recording": freeze["control_commit"],
        "runtime_commit": freeze["runtime_commit"],
        "workflow_commit": freeze["workflow_commit"],
        "train_seed": freeze["train_seed"],
        "eval_seed": freeze["eval_seed"],
        "checkpoint_paths": freeze["checkpoint_paths"],
        "checkpoint_agent_sha256": freeze["checkpoint_agent_sha256"],
    }
    (output / "provenance/checkpoints.json").write_text(
        json.dumps(checkpoint_provenance, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    demo_metadata = {
        "schema_version": 1,
        "experiment_id": "EXP-0006",
        "purpose": "presentation artifacts derived from fixed-condition checkpoint evaluation",
        "builder_script_sha256": file_sha256(Path(__file__).resolve()),
        "source_video_paths": {arm: str(path) for arm, path in video_paths.items()},
        "source_video_sha256": {arm: file_sha256(path) for arm, path in video_paths.items()},
        "comparison": comparison_meta,
        "learning_progression_source_paths": progression_sources,
        "limitations": [
            "Policy traces are 64x64 DMC logger renders upscaled for presentation.",
            "Behavior video shows one controlled episode per arm; score labels use the prior 64-episode evaluation means.",
            "This dmc_proprio experiment does not support pixel open-loop world-model prediction videos.",
        ],
    }
    (output / "provenance/demo_metadata.json").write_text(
        json.dumps(demo_metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    copied_sources = {
        "figures/paper_curve_comparison.png": EXP5_REVIEW / "curve_comparison.png",
        "data/exp0005_summary.json": EXP5_ARTIFACTS / "summary.json",
        "data/exp0005_curve.csv": EXP5_ARTIFACTS / "curve.csv",
        "figures/mechanism_timeseries.png": SOURCE_REVIEW / "mechanism_timeseries.png",
        "figures/representation_and_performance.png": SOURCE_REVIEW / "representation_and_performance.png",
        "data/summary.json": SOURCE_ARTIFACTS / "summary.json",
        "data/paired_comparisons.csv": SOURCE_ARTIFACTS / "paired_comparisons.csv",
        "data/per_run_summary.csv": SOURCE_ARTIFACTS / "per_run_summary.csv",
    }
    source_manifest = {
        "schema_version": 1,
        "builder_script_sha256": file_sha256(Path(__file__).resolve()),
        "fixed_eval_freeze_path": str(args.eval_root.with_suffix(".freeze")),
        "fixed_eval_freeze_sha256": file_sha256(args.eval_root.with_suffix(".freeze")),
        "copied_sources": [
            {
                "destination": destination,
                "source_path": str(source),
                "source_sha256": file_sha256(source),
            }
            for destination, source in copied_sources.items()
        ],
    }
    (output / "provenance/source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    showcase_summary = {
        "schema_version": 1,
        "experiment_ids": ["EXP-0005", "EXP-0006"],
        "scope": summary["scope"],
        "integrity_passed": summary["integrity_passed"],
        "run_count": len(summary["runs"]),
        "arms": ["baseline", "e1", "p4"],
        "seeds": [0, 1],
        "headline": "Stronger representation KL pressure reduced KL but worsened reconstruction and control in this local study.",
        "verdict": "promising_unresolved",
        "exp0005_reproduction_status": "terminal gate passed; early-curve gate failed; single-run evidence",
    }
    (output / "data/showcase_summary.json").write_text(
        json.dumps(showcase_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
