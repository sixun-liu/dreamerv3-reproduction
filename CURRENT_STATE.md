# CURRENT_STATE

> Updated: 2026-08-11T20:02:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0010` DMC Vision Walker 已完成 100K gate 与 1M 全程，单 seed 末段数值、AUC、独立
10 局评测和视频均形成闭环；GPU 已释放，下一项是 Atari100K Breakout。

## 当前主要矛盾

DMC Vision 的首个像素控制实例已经闭环，但 Atari 低数据预算与 Minecraft 环境链仍无运行证据。
Atari 使用不同的 2026 runtime、50M 模型和 ALE 计步语义，必须独立建立环境、smoke 和评测协议。

## 下一项决策

创建并冻结 `EXP-0011` Atari100K Breakout；先做 ALE L0 与越过 1024-decision replay warmup 的
2048-decision 模型 smoke，通过完整性和资源门后运行单 seed 100K decisions。Minecraft 继续 parked。
