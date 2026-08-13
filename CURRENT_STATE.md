# CURRENT_STATE

> Updated: 2026-08-13T06:42:41Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

跨域实验已在 DMC Vision、Atari100K Breakout 和 Minecraft 三条路径形成有界结果。近期暂停新增计算，
当前重点转为规范发布仓库、串联 DreamerV3 理解并准备后续讲解材料。

## 已冻结的实验边界

- DMC Vision Walker 完成单 seed、1M environment steps 的像素控制数值对齐实例。
- Breakout 完成 100K decisions 的降规模作者重实现，观察到学习趋势，但不是论文 200M 严格复现。
- Minecraft 推进至精确 500K；固定三局中 `2/3` 局达到木镐和圆石。训练 Replay 的铁矿和铁锭信号仅为
  `5/304`，固定评测为 `0/3`，因此不能表述为稳定炼铁或 Diamond 复现。

## 当前工作

发布分支统一三条代表路径的 `check`、`smoke` 和 `formal` 入口。启动器在正式运行前检查 runtime commit、
依赖、GPU 和磁盘，运行后保存命令、配置、日志、checkpoint、Replay 与结构完整性报告。默认 smoke 不承诺
策略质量，正式长预算必须显式选择。

## 暂停项

`size50m/size100m` 对照、Minecraft 1M 以上续训、多 seed 和 full-suite 扩展全部保留为候选，不在当前
发布阶段自动启动。只有出现新的明确研究问题、预算和验收标准时才重新预注册。
