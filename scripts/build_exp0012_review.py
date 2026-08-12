#!/usr/bin/env python3
"""Build the compact EXP-0012 human-review summary from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


L1_ITEMS = ("log", "planks", "crafting_table")
PLACEHOLDER_README_SHA256 = (
    "420afee68abd5061ff52add09d5d01fcdece7605d09ef20a27299f4aa8a1c138"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-analysis", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    placeholder = args.output_dir / "README.md"
    if args.output_dir.exists():
        existing = sorted(args.output_dir.iterdir())
        if existing != [placeholder] or sha256(placeholder) != PLACEHOLDER_README_SHA256:
            raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    training = json.loads(args.training_analysis.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if training["experiment_id"] != "EXP-0012" or evaluation["experiment_id"] != "EXP-0012":
        raise ValueError("Unexpected experiment ID")
    training_l1 = any(
        training["replay"]["milestones"][item]["first_global_step"] is not None
        for item in L1_ITEMS
    )
    evaluation_l1 = any(
        evaluation["milestones"][item]["successful_episodes"] > 0
        for item in L1_ITEMS
    )
    l1_observed = training_l1 or evaluation_l1
    decision = "promising_unresolved" if l1_observed else "negative_within_budget"
    verdict = (
        "scaled_100k_early_l1_milestone_observed"
        if l1_observed
        else "scaled_100k_no_l1_milestone_observed"
    )
    milestone_rows = []
    for tier, items in training["replay"]["tiers"].items():
        for item in items["items"]:
            train_item = training["replay"]["milestones"][item]
            eval_item = evaluation["milestones"].get(item, {})
            milestone_rows.append(
                {
                    "tier": tier,
                    "item": item,
                    "training_first_step": train_item["first_global_step"],
                    "training_successful_segments": train_item["successful_segments"],
                    "evaluation_successful_episodes": eval_item.get(
                        "successful_episodes", 0
                    ),
                }
            )
    summary = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "decision": decision,
        "verdict": verdict,
        "l1_observed": l1_observed,
        "training_l1_observed": training_l1,
        "evaluation_l1_observed": evaluation_l1,
        "milestones": milestone_rows,
        "training_episode_returns": training["training_scores"],
        "evaluation_returns": {
            key: evaluation[key]
            for key in (
                "episode_count",
                "return_mean",
                "return_median",
                "return_min",
                "return_max",
                "return_std_population",
            )
        },
        "resource": training["resource"],
        "evidence": {
            "training_analysis": str(args.training_analysis.resolve()),
            "training_analysis_sha256": sha256(args.training_analysis),
            "evaluation": str(args.evaluation.resolve()),
            "evaluation_sha256": sha256(args.evaluation),
            "training_figure": training["figure"],
            "episode0_video": evaluation["video"],
            "episode0_first_frame": evaluation["first_frame"],
        },
        "limitations": [
            "Single agent seed and only three terminal-checkpoint evaluation episodes.",
            "Minecraft world seeds are uncontrolled because the runtime wrapper exposes none.",
            "The local size50m/ratio32/100K protocol is far below the paper-scale Diamond run.",
            "A positive L1 signal would establish only an early milestone, not Diamond competence.",
        ],
        "human_visual_confirmation": "pending",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "review_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        "| 层级 | 物品 | 训练首次步 | 训练成功轨迹 | 评测成功回合 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in milestone_rows:
        first = row["training_first_step"]
        rows.append(
            f"| {row['tier']} | `{row['item']}` | "
            f"{first if first is not None else '-'} | "
            f"{row['training_successful_segments']} | "
            f"{row['evaluation_successful_episodes']} |"
        )
    result = f"""# EXP-0012 Minecraft Reduced 复现结果

## 受限结论

- Decision：`{decision}`
- Verdict：`{verdict}`
- 100K 训练或三回合终点评测中是否观察到 L1：`{str(l1_observed).lower()}`
- 该结果只裁决作者 2026 重实现的环境链、size50m 训练完整性和 100K 早期里程碑，**不构成**
  论文约 100M 预算 Diamond 数值复现。

## 里程碑

{chr(10).join(rows)}

## 数值与资源

- 训练完整回合数：`{training['training_scores']['count']}`；训练回报均值：
  `{training['training_scores'].get('mean', 'n/a')}`。
- 独立评测：`{evaluation['episode_count']}` 回合；均值 `{evaluation['return_mean']:.4f}`，
  中位数 `{evaluation['return_median']:.4f}`，范围
  `[{evaluation['return_min']:.4f}, {evaluation['return_max']:.4f}]`。
- 训练墙钟：`{training['resource']['wall_seconds'] / 3600:.3f} h`；峰值显存：
  `{training['resource']['gpu']['memory_used_mib_max']:.0f} MiB`；平均 GPU 利用率：
  `{training['resource']['gpu']['gpu_util_percent_mean']:.2f}%`。

## 审查入口

- 训练曲线与里程碑：`{training['figure']}`
- 固定 episode-0 视频（stride=4）：`{evaluation['video']}`
- 固定 episode-0 首帧：`{evaluation['first_frame']}`
- 训练分析 JSON：`{args.training_analysis.resolve()}`
- 三回合评测 JSON：`{args.evaluation.resolve()}`

## 限制

- 单 agent seed，终点评测仅三回合。
- runtime 未暴露 Minecraft world seed，因此世界随机性不可控。
- 本地 size50m/ratio32/100K 远低于论文 Diamond 预算；即使出现 L1，也不能外推为钻石能力。
- 人工视觉审查：`pending`。
"""
    (args.output_dir / "RESULT.md").write_text(result, encoding="utf-8")
    placeholder.write_text(
        "# EXP-0012 Review\n\n"
        "- 受限结论：`RESULT.md`\n"
        "- 机器可读摘要：`review_summary.json`\n"
        "- 人工视觉确认：`pending`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
