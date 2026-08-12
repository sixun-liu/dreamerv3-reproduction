# CURRENT_STATE

> Updated: 2026-08-12T11:07:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

EXP-0020 已以 `envs=4` 从只读 100K 源精确训练到 200K：工程门全绿，固定三回合评测
维持 crafting table `2/3`，并首次在新增 replay 与评测中观察到 wooden pickaxe；尚无 cobblestone。

## 当前主要矛盾

新增 100K 训练墙钟 `32.18` 分钟，平均 CPU `10.73` 核、峰值内存 `36.15 GiB`、GPU 平均
利用率 `12.35%`、峰值 `94%`；精确终点、200K unique stepid、ratio32、源隔离、OOM、磁盘和
清场全部通过。单环境三回合评测却耗时 `32.87` 分钟，主要只使用约 3 核 Java、不到 1 核 Python
和低 GPU 利用率，当前效率矛盾已从训练转移到终点评测。

## 下一项决策

先做三环境并行终点评测的短 smoke 与同 checkpoint 配对 probe；只有端到端至少加速 `1.5x`，且
逐局终止、回报/里程碑、固定 episode-0 视频、checkpoint 绑定和清场语义不漂移，才替换后续默认
评测。随后再决定是否从当前 200K 模型有界推进到 500K。
