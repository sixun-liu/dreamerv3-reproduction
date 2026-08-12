# CURRENT_STATE

> Updated: 2026-08-12T12:16:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

训练侧 EXP-0020 已以 `envs=4` 精确到 200K 并观察到木镐推进；评测侧 EXP-0022 已修复 inventory
映射并选择三环境并行 evaluator，将固定三回合墙钟由 `1972s` 降至 `796s`。

## 当前主要矛盾

EXP-0022 完整三局的运行时索引、reset 全零、逐局边界、固定视频、资源和清场门全部通过；端到端
`2.48x` 加速包含本轮随机世界总动作更少的影响，动作归一吞吐提升为更保守的 `1.42x`。并行评测
平均使用 `8.06` 核 CPU、峰值 cgroup 内存 `26.94 GB`，无 OOM；评测瓶颈已显著降低。

当前 `size50m` 实际 optimizer 参数为 `46,812,213`。约 `24.6 GiB` GPU 分配主要受 JAX 预分配
影响，既不能证明 100M 可行，也不能说明 100M 必然 OOM；模型规模需用独立初始化和短吞吐 gate。

## 下一项决策

以 EXP-0020 精确 200K 为只读源，固定 patched runtime、`envs=4`、`size50m` 与 ratio32，有界新增
300K 到精确 500K，终点使用三环境并行 evaluator。之后再做 50M/100M 资源与约 5K 同预算吞吐
gate；更大模型是否以更少交互达到里程碑，要通过从零、等交互预算学习对照回答，而不是由规模推断。
