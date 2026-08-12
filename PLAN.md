# PLAN

> Updated: 2026-08-12T02:57:14Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在可审计的作者重实现上建立 DreamerV3 跨域最小证据闭环，区分像素控制、Atari
  低数据预算与 Minecraft 工程可行性。
- 当前问题：三个跨域最小闭环已完成；需要人工审查 Minecraft Reduced 的 L1 证据与 L2 边界，
  再决定是否先做已有 replay 的机制分析，或另行授权更大预算。

## 阶段退出门

- [x] EXP-0010 DMC Vision 完成 100K gate、1M 续训、独立评测和图像材料。
- [x] EXP-0011 Atari100K Breakout 完成 100K decisions、独立评测和 DQN 协议差异表。
- [x] EXP-0012 Minecraft 完成依赖/L0、size50m smoke、100K 训练、独立评测与审查材料。

## 活动路线

1. `EXP-0010`、`EXP-0011`、`EXP-0012` 均已完成受限裁决，三个域不合并成单一论文复现主张。
2. `EXP-0012` 的工程链和 L1 早期里程碑成立；L2 链与论文 Diamond 主结果均不成立。
3. 当前只进行人工审查与知识综合；任何更大 Minecraft 预算必须重新预注册并取得用户授权。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 预算；当前授权只到 100K。
