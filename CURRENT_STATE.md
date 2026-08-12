# CURRENT_STATE

> Updated: 2026-08-12T00:22:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` Minecraft Reduced 的真实 L0 与 size50m smoke 均通过；smoke 的 4096 请求因冻结
runtime 的 10-step driver 量子自然停在 4100，原 instrumentation failure 已保留并完成带哈希的
离线 reconciliation。GPU 空闲，formal 100K 资源门全绿。

## 当前主要矛盾

DMC Vision 与 Atari 低数据预算实例均已闭环。Minecraft 的 Malmo/headless 环境链、动作、图像、
reward/inventory/reset、checkpoint/replay/有限损失及 size50m 资源已取得真实运行证据；剩余问题是
100K 内是否出现可复核 L1 早期里程碑。论文钻石任务是 100M 量级，本轮不裁决论文数值。

## 下一项决策

只启动冻结的 seed0 formal 100K；自然结束后按预注册口径完成里程碑分析、固定终点 checkpoint
三回合独立评测、episode0 视频、曲线、资源材料和受限裁决，未经新授权不追加预算。
