#!/usr/bin/env python3
"""Build the bounded EXP-0013 review sheet from frozen run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLACEHOLDER_README_SHA256 = (
    "64afe94b687810c00de23cea2735c0beea6f4f20df1878a41da18bd11c9d7d9b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(analysis_path: Path, integrity_path: Path, completed_path: Path) -> dict:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if any(
        item.get("experiment_id") != "EXP-0013"
        for item in (analysis, completed)
    ):
        raise ValueError("Unexpected experiment ID")
    if "passed" not in integrity or "config_semantically_equal" not in integrity:
        raise ValueError("Unexpected integrity report schema")
    if analysis["arm"] != "envs2" or completed["arm"] != "envs2":
        raise ValueError("EXP-0013 review expects the envs2 stopping arm")
    expansion_stopped = not analysis["expand_recommended"]
    return {
        "schema_version": 1,
        "experiment_id": "EXP-0013",
        "arm": "envs2",
        "decision": "negative" if expansion_stopped else "promote",
        "verdict": (
            "envs2_safe_but_expansion_gate_failed"
            if expansion_stopped
            else "envs2_expansion_gate_passed"
        ),
        "integrity_passed": integrity["passed"],
        "resource_gate_passed": analysis["resource_gate_passed"],
        "replay_ratio_ok": analysis["replay_ratio_ok"],
        "expand_recommended": analysis["expand_recommended"],
        "throughput": {
            "requested_environment_steps": analysis["requested_environment_steps"],
            "training_wall_seconds": completed["training_wall_seconds"],
            "end_to_end_steps_per_second": analysis["end_to_end_steps_per_second"],
            "historical_long_run_steps_per_second": analysis[
                "historical_wall_steps_per_second"
            ],
            "end_to_end_speedup_vs_historical": analysis[
                "end_to_end_speedup_vs_historical_long_run"
            ],
            "policy_fps_tail_mean": analysis["policy_fps_tail_mean"],
            "historical_policy_fps_mean": analysis["historical_policy_fps_mean"],
            "policy_speedup_vs_historical": analysis[
                "policy_speedup_vs_historical"
            ],
            "target_policy_speedup_met": analysis["target_policy_speedup_met"],
        },
        "resource": analysis["resource"],
        "evidence": {
            "analysis": str(analysis_path.resolve()),
            "analysis_sha256": sha256(analysis_path),
            "integrity": str(integrity_path.resolve()),
            "integrity_sha256": sha256(integrity_path),
            "completed": str(completed_path.resolve()),
            "completed_sha256": sha256(completed_path),
        },
        "limitations": [
            "The 5040-step arm has a much larger startup/JIT fraction than the historical 100K baseline.",
            "The historical baseline used envs=1/debug=true, so it is not a same-budget causal control.",
            "The final policy-FPS sample includes natural-end checkpoint overhead and depresses the three-sample tail mean.",
            "This diagnostic does not measure policy quality or reproduce the paper's Minecraft score.",
        ],
        "next_discriminating_question": (
            "At the same 5040-step budget and debug=false setting, does envs=2 materially "
            "outperform envs=1 after preserving the frozen integrity and resource gates?"
        ),
        "human_review": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    placeholder = args.output_dir / "README.md"
    if args.output_dir.exists():
        existing = sorted(args.output_dir.iterdir())
        if existing != [placeholder] or sha256(placeholder) != PLACEHOLDER_README_SHA256:
            raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    summary = build(args.analysis, args.integrity, args.completed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "review_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    throughput = summary["throughput"]
    resource = summary["resource"]
    placeholder.write_text(
        f"""# EXP-0013 Minecraft 多环境吞吐探针

## 受限结论

- `envs=2` 在 5040 environment steps 后自然结束，完整性、replay ratio、内存、磁盘和 OOM 门均通过。
- 端到端吞吐为 `{throughput['end_to_end_steps_per_second']:.2f} steps/s`，是历史 100K 单环境基线的 `{throughput['end_to_end_speedup_vs_historical']:.3f}x`。
- 三点尾窗 policy FPS 均值为 `{throughput['policy_fps_tail_mean']:.2f}`，是历史均值的 `{throughput['policy_speedup_vs_historical']:.3f}x`，低于预注册的 `1.10x` 扩档门，更未达到 `1.50x` 目标。
- 因此 `envs=4/8` 不放行。这个裁决只表示**当前冻结门槛下停止扩档**，不证明双环境在同预算下必然比单环境慢。

## 资源账

| 指标 | 结果 |
|---|---:|
| 训练墙钟 | {throughput['training_wall_seconds']} s |
| 平均 CPU 使用 | {resource['cpu_usage_cores_average']:.2f} cores |
| CPU throttled wall fraction | {resource['cpu_throttled_wall_fraction']:.3f} |
| cgroup 峰值内存 | {resource['max_cgroup_memory_bytes'] / 2**30:.2f} GiB |
| Java 峰值实例数 | {resource['max_java_count']} |
| GPU 平均/峰值利用率 | {resource['gpu_util_mean_percent']:.2f}% / {resource['gpu_util_peak_percent']:.0f}% |
| 峰值显存 | {resource['gpu_memory_peak_mib']:.0f} MiB |
| 系统盘净增长 | {resource['system_disk_loss_bytes'] / 2**20:.2f} MiB |
| 数据盘净增长 | {resource['data_disk_growth_bytes'] / 2**30:.2f} GiB |

## 下一判别问题

同为 5040 步、`debug=false` 时，`envs=2` 是否显著优于 `envs=1`。该配对对照负责拆分启动/JIT、debug 与环境并发的影响；EXP-0013 本身不事后修改门槛。

## 证据与限制

- 机器摘要：`review_summary.json`
- 原始分析：`{summary['evidence']['analysis']}`
- 完整性报告：`{summary['evidence']['integrity']}`
- 5040 步比历史 100K 基线有更高的启动/JIT 占比。
- policy FPS 最后一个采样点含自然结束存档开销，三点尾窗偏保守。
- 本实验是基础设施诊断，不评判 Minecraft 策略质量或论文分数。
- 人工审查：`pending`。
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
