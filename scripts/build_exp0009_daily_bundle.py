#!/usr/bin/env python3
"""Build a compact, provenance-linked daily-report bundle for EXP-0009."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CONTROL = Path("/root/autodl-tmp/dreamerv3-reproduction")
RUNTIME = Path("/root/autodl-tmp/dreamerv3-2411f7d")
ARTIFACT = Path("/root/autodl-tmp/artifacts/dreamerv3/EXP-0009")
REVIEW = Path(
    "/root/autodl-tmp/artifacts/dreamerv3/review/"
    "EXP-0009-reacher-signal-ablation/README.md"
)
GRADIENT = Path(
    "/root/autodl-tmp/runs/"
    "EXP-0009__reacher-hard__gradient-gate__s000__debug-120-dec__20260806T032742Z"
)
MATRIX = Path(
    "/root/autodl-tmp/runs/"
    "EXP-0009__reacher-hard__three-arm__s000__1m-env__20260806T034000Z"
)
SHOWCASE = Path(
    "/root/autodl-tmp/runs/"
    "EXP-0009__reacher-hard__showcase-eval-three-arm__20260806T093903Z"
)
ARMS = ("baseline", "no_reward_value", "no_reconstruction")


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


def copy_sources(output: Path) -> list[dict]:
    copies = {
        "data/summary.json": ARTIFACT / "summary.json",
        "data/resource_summary.csv": ARTIFACT / "resource_summary.csv",
        "data/gradient_gate.json": GRADIENT / "gradient_gate.json",
        "figures/learning_curves.png": ARTIFACT / "learning_curves.png",
        "figures/behavior_contact_sheet.png": ARTIFACT / "showcase/contact_sheet.png",
        "demo/three_arm_comparison.mp4": ARTIFACT
        / "showcase/three_arm_comparison.mp4",
        "demo/three_arm_comparison.gif": ARTIFACT
        / "showcase/three_arm_comparison.gif",
        "protocol/EXP0009_REACHER_SIGNAL_PROTOCOL.md": CONTROL
        / "docs/reproduction/EXP0009_REACHER_SIGNAL_PROTOCOL.md",
        "protocol/exp0009_reacher_matrix.yaml": CONTROL
        / "docs/reproduction/configs/exp0009_reacher_matrix.yaml",
        "protocol/REVIEW.md": REVIEW,
        "provenance/gradient_gate.freeze.json": GRADIENT.with_suffix(".freeze"),
        "provenance/gradient_gate.completed.json": GRADIENT.with_suffix(".completed"),
        "provenance/training_matrix.freeze.json": MATRIX.with_suffix(".freeze"),
        "provenance/training_matrix.completed.json": MATRIX.with_suffix(".completed"),
        "provenance/showcase_eval.freeze.json": SHOWCASE.with_suffix(".freeze"),
        "provenance/showcase_eval.completed.json": SHOWCASE.with_suffix(".completed"),
        "provenance/showcase_manifest.json": ARTIFACT / "showcase/manifest.json",
        "provenance/scripts/analyze_exp0009.py": CONTROL
        / "scripts/analyze_exp0009.py",
        "provenance/scripts/build_exp0009_showcase.py": CONTROL
        / "scripts/build_exp0009_showcase.py",
        "provenance/scripts/build_exp0009_daily_bundle.py": Path(__file__).resolve(),
    }
    for arm in ARMS:
        copies[f"provenance/{arm}.integrity.json"] = MATRIX / f"{arm}.integrity.json"

    records = []
    for destination, source in copies.items():
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


def result_table(summary: dict) -> str:
    labels = {
        "baseline": "Dreamer baseline",
        "no_reward_value": "No reward/value representation gradients",
        "no_reconstruction": "No reconstruction representation gradients",
    }
    rows = []
    for arm in ARMS:
        item = summary["arms"][arm]
        rows.append(
            f"| {labels[arm]} | {item['normalized_auc']:.2f} | "
            f"{item['final_200k_mean']:.2f} | {item['final_200k_std']:.2f} | "
            f"{item['episodes']} | {item['wall_seconds'] / 3600:.2f} |"
        )
    return "\n".join(rows)


def write_human_docs(output: Path, report_date: str, summary: dict, showcase: dict) -> None:
    table = result_table(summary)
    returns = "/".join(
        str(int(showcase["episodes"][arm]["episode_return"])) for arm in ARMS
    )
    (output / "README.md").write_text(
        f"""# DreamerV3 Reacher Hard 学习信号消融材料包（{report_date}）

本目录整理自 `EXP-0009` 的已登记 artifact、冻结协议和完整性记录。它没有重新训练，
也不改变实验裁决；ZIP 是 presentation-only 派生包，canonical evidence 仍保留在原路径。

## 推荐查看顺序

1. `01_日报事实卡.md`：实验协议、三臂结果和结论权限。
2. `figures/learning_curves.png`：固定 100K environment-step 分箱学习曲线。
3. `figures/behavior_contact_sheet.png`：同环境 seed 的五时刻行为对照。
4. `demo/three_arm_comparison.mp4`：同步三栏完整 episode。
5. `03_日报草稿.md`：可直接改写为给导师的日报。
6. `04_口径与常见问答.md`：避免把 pilot 写成完整 Figure 18 复现。

## 证据入口

- 主统计：`data/summary.json`
- 梯度验证：`data/gradient_gate.json`
- 资源账：`data/resource_summary.csv`
- 协议：`protocol/EXP0009_REACHER_SIGNAL_PROTOCOL.md`
- 来源清单：`provenance/source_manifest.json`
- 全目录校验：`SHA256SUMS`

人工审查状态仍为 `pending`。showcase 单局回报 `{returns}` 只帮助观察行为，
不替代训练期 AUC 和末窗口统计。
""",
        encoding="utf-8",
    )
    (output / "01_日报事实卡.md").write_text(
        f"""# EXP-0009 日报事实卡

## 任务

- 目标：验证 DreamerV3 Figure 18 学习信号消融在 DMC Reacher Hard 上的定性排序。
- runtime：作者 2024 重实现 `2411f7d` 谱系，加冻结的 representation-gradient routing。
- 配置：proprio、size12m、16 envs、replay ratio 512、action repeat 2。
- 预算：每臂 500K agent decisions = 1M environment steps。
- 重复：paired agent/environment seed 0，仅一个 seed。

## 三臂语义

| Arm | reconstruction -> representation | reward -> representation | replay value -> representation |
|---|---:|---:|---:|
| baseline | on | on | on |
| no_reward_value | on | stopped | stopped |
| no_reconstruction | stopped | on | on |

`stopped` 只阻断对应 loss 对 encoder/RSSM 的梯度；decoder、reward head 和 critic 自身仍学习。

## 结果

| Arm | fixed-bin AUC | final 200K mean | final 200K std | episodes | wall h |
|---|---:|---:|---:|---:|---:|
{table}

- 预注册方向门通过：`baseline > no_reward_value > no_reconstruction`。
- 三臂均精确完成到 checkpoint step 500000，各有 992 个有限 episode。
- 无 traceback、OOM、意外 nonfinite metric 或配置漂移。
- 正式三臂矩阵 wall time 约 5.75 小时。

## 结论权限

支持 Reacher Hard、单 paired seed、1M environment steps 下与 Figure 18 一致的定性机制顺序。
不等于完整复现 Figure 18：只有 1/14 任务、1 个 seed，预算仅为论文 Reacher Hard 的十分之一，
而且 runtime 不是 exact 2023 training artifact。registry 裁决为 `promising_unresolved`。
""",
        encoding="utf-8",
    )
    (output / "02_实验过程与结果.md").write_text(
        """# 实验过程与结果

## 1. 先恢复消融语义

本轮没有把某个 loss scale 直接设为零，因为那会让对应 head 也停止学习。实现采用
`stop_gradient` 作用于 head 输入 latent：只阻断该信号塑形 encoder/RSSM，同时保留 head 更新。

## 2. 梯度 gate

先运行代数单元测试，再在真实 Dreamer 小模型 Reacher batch 上检查参数组 grad norm。

- baseline：reconstruction、reward、replay critic 对 encoder/RSSM 均为非零；
- no_reward_value：reward 和 replay critic 对 encoder/RSSM 精确为 0，reward head/critic 非零；
- no_reconstruction：reconstruction 对 encoder/RSSM 精确为 0，decoder grad norm 为 1.5009；
- gate 总结果：passed。

最初的 1024-decision probe 因 report 时钟早于 replay 可采样且 terminal step 不整除而被判为
instrumentation failure；没有形成梯度裁决。修正 runner 后用独立 tag 重跑 120-decision gate，通过后
才启动正式矩阵。

## 3. 三臂正式运行

顺序在结果前冻结为 baseline、no_reward_value、no_reconstruction。三臂均使用相同 agent/env seed
派生规则、预算和超参数；不按中间分数停车或调参。全部自然完成并通过终点 checkpoint、配置、有限性、
日志和资源完整性检查。

## 4. 主要现象

- baseline 第一箱约 285.5，第二箱升到 820.6，随后稳定在约 925--956；
- no_reward_value 学得更慢且波动明显，后期约 500--565；
- no_reconstruction 十个箱始终约 4--11，基本没有形成有效控制策略。

这支持“重建梯度提供主要的任务无关表征学习信号；奖励/价值梯度进一步把表征朝任务方向塑形”的
解释。单任务单 seed 不能证明跨任务普遍因果，下一判别问题是相同 1M 协议下 seeds 1、2 是否保持排序。

## 5. 行为展示

三份终点 checkpoint 在同一个 DMC eval seed 10000 初态下各录制 501 帧。策略轨迹可能因随机性分叉。
单局回报 baseline/no_reward_value/no_reconstruction 为 985/107/3，只作为展示 metadata；科学裁决使用
训练期 992 episodes/arm 的固定分箱统计。
""",
        encoding="utf-8",
    )
    (output / "03_日报草稿.md").write_text(
        f"""# DreamerV3 复现日报草稿（{report_date}）

## 今日目标

验证 DreamerV3 Figure 18 中“去奖励/价值表征梯度”和“去重建表征梯度”的精确实现，并在
DMC Reacher Hard 上完成一个低成本三臂 pilot。

## 完成工作

1. 用参数组梯度测试确认被阻断信号不再更新 encoder/RSSM，但对应 decoder/reward head/critic 仍学习。
2. 完成 baseline、no reward/value representation gradients、no reconstruction representation gradients
   三臂训练；每臂 1M environment steps、paired seed 0。
3. 生成固定分箱学习曲线、资源统计、同初态行为 contact sheet、GIF 和视频。

## 关键结果

三臂固定分箱 AUC 分别为 `867.42 / 391.61 / 7.84`，末 200K 平均回报分别为
`952.75 / 538.88 / 8.80`。排序为 `baseline > no_reward_value > no_reconstruction`，与论文 Figure 18
的定性机制方向一致。去掉重建梯度后几乎无法学习，说明重建是这一任务中表征形成的主要信号；
奖励/价值梯度则进一步提升了任务相关性。

## 口径与限制

这是单任务、单 paired seed、1M environment-step pilot，只占论文 Reacher Hard 预算的十分之一；
runtime 是作者 2024 重实现，不是 exact 2023 artifact。因此目前只能称为“Reacher Hard 上的机制方向
获得支持”，不能称为完整复现 Figure 18。

## 算力账

三臂正式矩阵 wall time 约 5.75 小时；包含梯度验证和展示评估的结案 compute 账约 6.18 GPU 小时。
三臂均自然完成，GPU 已释放。

## 下一步

先人工审查学习曲线和行为材料；若继续实验，在相同 1M 协议下补 paired seeds 1、2，优先验证
排序的跨 seed 稳定性，不直接启动 10M 或 14-task 全矩阵。

## 分工说明

agent 辅助完成代码核验、实验执行、统计和材料生成；用户主导实验目标、成本授权和结果解释。
""",
        encoding="utf-8",
    )
    (output / "04_口径与常见问答.md").write_text(
        """# 口径与常见问答

## 这次复现了 Figure 18 吗？

没有完整复现。Figure 18 是 14 个任务的聚合消融；本轮只做 Reacher Hard、单 seed、论文十分之一
预算。准确说法是：在这个受限 pilot 上复现了论文报告的定性排序。

## no_reward_value 为什么仍能学习？

它仍有 reconstruction、dynamics/KL 和 continuation 等信号。世界模型可以从这些任务无关预测目标中
学习状态表征，只是缺少 reward/replay-value 对表征的任务导向塑形，因此更慢且更不稳定。

## no_reconstruction 是否等于关闭 decoder？

不是。decoder 自身继续学习，只是 decoder reconstruction loss 不再反向更新 encoder/RSSM。
真实模型梯度 gate 验证 decoder grad norm 非零，而 reconstruction -> encoder/RSSM 精确为零。

## 视频中的 985/107/3 能当最终分数吗？

不能。它们是同一 eval seed 下各一局 stochastic policy 的展示 metadata。主指标来自训练期每臂
992 个 episode 的固定 100K environment-step 分箱 AUC，末窗口均值也来自 208 个 episode。

## 为什么先补 seed，而不是直接跑 10M？

当前最大不确定性是 seed 0 是否偶然。用同预算增加两个 paired seed 比直接把每臂延长到 10M 更便宜，
也能更直接区分“机制排序稳定”与“单 seed 偶然”。

## 结果能否说明重建在所有 DreamerV3 任务中都最重要？

不能。Reacher Hard 的短预算结果只支持本任务。本轮提供的是受控干预证据，不是跨任务普遍定律。
""",
        encoding="utf-8",
    )
    (output / "05_论文定位.md").write_text(
        """# 论文定位

- 论文：DreamerV3, arXiv:2301.04104v2。
- 对应主张：主文 Figure 6b 与 supplementary Figure 18 的 learning-signal ablation。
- 论文对比：Dreamer、No Reward & Value Gradients、No Reconstruction Gradients。
- Figure 18 范围：14 个任务；Reacher Hard 曲线使用约 10M environment steps。
- 本轮范围：Reacher Hard 单任务、paired seed 0、每臂 1M environment steps。

本轮最重要的研读价值不是追求相同曲线外形，而是通过精确梯度干预回答：世界模型的 latent
representation 主要由哪些预测目标塑形，以及任务特定信号在任务无关重建表征之上提供了什么增益。
""",
        encoding="utf-8",
    )


def write_manifest(output: Path, report_date: str, copied_sources: list[dict]) -> None:
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0009",
        "purpose": "daily_report_presentation_bundle",
        "report_date": report_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance_quality": "derived_from_fingerprinted",
        "canonical_artifact_root": str(ARTIFACT),
        "control_commit": git_commit(CONTROL),
        "runtime_commit": git_commit(RUNTIME),
        "registry_ids": {
            "closure": "EVT-0057",
            "claim": "C-0002",
            "sources": ["ART-0049", "ART-0050", "ART-0051", "ART-0052"],
        },
        "scientific_scope": (
            "Reacher Hard, one paired seed, 1M environment steps per arm; "
            "not a complete Figure 18 replication"
        ),
        "copied_sources": copied_sources,
    }
    target = output / "provenance/source_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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

    args.output.mkdir(parents=True)
    summary = read_json(ARTIFACT / "summary.json")
    showcase = read_json(ARTIFACT / "showcase/manifest.json")
    if not summary.get("directional_gate_passed"):
        raise ValueError("Frozen EXP-0009 directional gate did not pass")
    if showcase.get("frames") != 501:
        raise ValueError("Unexpected showcase frame count")

    copied_sources = copy_sources(args.output)
    write_human_docs(args.output, args.report_date, summary, showcase)
    write_manifest(args.output, args.report_date, copied_sources)
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
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
