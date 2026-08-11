# CURRENT_STATE

> Updated: 2026-08-11T23:58:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` Minecraft Reduced 已预注册；独立 Python 3.11/MineRL 环境、Java 8、Xvfb 和 JAX GPU
兼容门通过，GPU 空闲，下一步是真实 Minecraft L0。

## 当前主要矛盾

DMC Vision 与 Atari 低数据预算实例均已闭环。Minecraft 静态依赖和 GPU 兼容性已有本机证据，
但 Malmo 启动、headless 图像、动作/reward/inventory/reset 及 size50m 模型资源仍未经过真实运行。
论文钻石任务是 100M 量级，本轮只能裁决环境链和 100K 早期里程碑。

## 下一项决策

冻结 `EXP-0012` 后运行 32-step 测试专用 L0；只有 reset/step、图像、动作、reward、inventory、
terminal/reset 和视频通过，才进入 4096-step size50m smoke，资源门通过后才运行最多 100K。
