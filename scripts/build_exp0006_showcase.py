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
    box(ax, (0.7, 5.7), 2.1, 1.05, "Observation\nx_t", COLORS["teal"])
    box(ax, (0.7, 3.7), 2.1, 1.05, "Previous action\na_{t-1}", COLORS["gold"], text_color=COLORS["ink"])
    box(ax, (3.6, 5.7), 2.2, 1.05, "Encoder\ne_t", COLORS["baseline"])
    box(ax, (6.6, 4.6), 2.7, 2.3, "RSSM latent state\nh_t deterministic\nz_t stochastic", "#334b5e")
    box(ax, (10.2, 6.0), 2.2, 0.9, "Decoder", COLORS["teal"])
    box(ax, (10.2, 4.85), 2.2, 0.9, "Reward head", COLORS["gold"], text_color=COLORS["ink"])
    box(ax, (10.2, 3.7), 2.2, 0.9, "Continue head", COLORS["p4"])
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
    box(ax, (6.4, 1.0), 2.2, 1.25, "Actor\nchoose actions", COLORS["e1"])
    box(ax, (10.1, 1.0), 2.2, 1.25, "Critic\nestimate returns", COLORS["baseline"])
    box(ax, (13.2, 1.0), 2.1, 1.25, "Act in the\nreal environment", COLORS["p4"])
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
    ax.text(4.35, 6.55, "remove free bits", ha="center", fontsize=13, color=COLORS["e1"], weight="bold")
    ax.text(8.65, 6.55, "10x representation KL pressure", ha="center", fontsize=13, color=COLORS["p4"], weight="bold")

    ax.text(0.8, 4.55, "Observed", fontsize=15, color=COLORS["muted"], weight="bold")
    box(ax, (0.8, 2.85), 3.8, 1.35, "E1 raw-KL direction\nnot consistent across seeds\nperformance lower in both", "#734b40")
    box(
        ax,
        (5.0, 2.85),
        3.8,
        1.35,
        f"P4 KL = {p4_kl[0]*100:.1f}% / {p4_kl[1]*100:.1f}% of E1\nreconstruction = {p4_rec[0]:.2f}x / {p4_rec[1]:.2f}x",
        "#2e6650",
    )
    box(
        ax,
        (9.2, 2.85),
        3.8,
        1.35,
        f"P4 final score\n{p4_score[0]:.1f}% / {p4_score[1]:.1f}% vs E1\nno strict collapse trigger",
        "#344b5e",
    )
    arrow(ax, (4.6, 3.52), (5.0, 3.52))
    arrow(ax, (8.8, 3.52), (9.2, 3.52))

    ax.text(0.8, 2.22, "Local mechanism supported by this task", fontsize=15, color=COLORS["muted"], weight="bold")
    ax.text(
        0.8,
        1.45,
        "stronger KL pressure",
        fontsize=18,
        color=COLORS["p4"],
        weight="bold",
    )
    arrow(ax, (3.25, 1.60), (4.05, 1.60))
    ax.text(4.2, 1.45, "posterior closer to prior", fontsize=18, color=COLORS["ink"], weight="bold")
    arrow(ax, (7.2, 1.60), (8.0, 1.60))
    ax.text(8.15, 1.45, "less usable detail", fontsize=18, color=COLORS["ink"], weight="bold")
    arrow(ax, (10.75, 1.60), (11.55, 1.60))
    ax.text(11.7, 1.45, "worse modeling + control", fontsize=18, color=COLORS["red"], weight="bold")
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
    box(ax, (0.7, 5.0), 4.4, 2.1, "6 complete runs\n3 arms x 2 seeds\n500K env steps each", COLORS["baseline"], fontsize=21)
    box(ax, (5.8, 5.0), 4.4, 2.1, "Baseline final-30K\n903.4 / 833.5\nusable walking policies", COLORS["teal"], fontsize=21)
    box(ax, (10.9, 5.0), 4.4, 2.1, "P4 raw KL\n0.093 / 0.048\nlarge representation shift", COLORS["p4"], fontsize=21)
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


def copy_sources(output: Path):
    shutil.copy2(SOURCE_REVIEW / "mechanism_timeseries.png", output / "figures/mechanism_timeseries.png")
    shutil.copy2(SOURCE_REVIEW / "representation_and_performance.png", output / "figures/representation_and_performance.png")
    shutil.copy2(SOURCE_ARTIFACTS / "summary.json", output / "data/summary.json")
    shutil.copy2(SOURCE_ARTIFACTS / "paired_comparisons.csv", output / "data/paired_comparisons.csv")
    shutil.copy2(SOURCE_ARTIFACTS / "per_run_summary.csv", output / "data/per_run_summary.csv")


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
    showcase_summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0006",
        "scope": summary["scope"],
        "integrity_passed": summary["integrity_passed"],
        "run_count": len(summary["runs"]),
        "arms": ["baseline", "e1", "p4"],
        "seeds": [0, 1],
        "headline": "Stronger representation KL pressure reduced KL but worsened reconstruction and control in this local study.",
        "verdict": "promising_unresolved",
    }
    (output / "data/showcase_summary.json").write_text(
        json.dumps(showcase_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
