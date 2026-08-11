# CURRENT_STATE

> Updated: 2026-08-11T23:24:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0011` Atari100K Breakout 已完成 100K decisions、独立 10 局评测和视频：末段训练均值
`7.54`、独立评测均值 `9.10`，冻结学习趋势门通过；GPU 已释放，下一项是 Minecraft Reduced。

## 当前主要矛盾

DMC Vision 与 Atari 低数据预算实例均已闭环，但 Minecraft 的 Java、MineRL/Malmo、headless 渲染、
wrapper 和依赖版本尚无本机证据。论文钻石任务是 100M 量级，本轮只能裁决环境链和 100K 早期里程碑。

## 下一项决策

在新的 `exp/EXP-0012-minecraft-reduced` 分支创建并冻结 `EXP-0012`。先做依赖审计和 L0；只有
reset/step、图像、动作、reward、inventory、terminal、视频及模型资源门全部通过，才运行最多 100K。
