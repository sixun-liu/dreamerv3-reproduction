# PLAN

> Updated: 2026-08-11T23:24:00Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `exploration`
- 北极星：在可审计的作者重实现上建立 DreamerV3 跨域最小证据闭环，区分像素控制、Atari
  低数据预算与 Minecraft 工程可行性。
- 当前问题：DMC Vision 与 Atari100K 已形成受限闭环；剩余问题是 Minecraft Reduced 的真实环境链
  与 100K 早期里程碑是否具备工程和计算可行性。

## 阶段退出门

- [x] EXP-0010 DMC Vision 完成 100K gate、1M 续训、独立评测和图像材料。
- [x] EXP-0011 Atari100K Breakout 完成 100K decisions、独立评测和 DQN 协议差异表。
- [ ] EXP-0012 Minecraft 完成依赖/L0；通过时训练至 100K，否则以可复现工程阻塞结案。

## 活动路线

1. `EXP-0010`、`EXP-0011` 已关闭；当前只建立 `EXP-0012` Minecraft 环境、协议与证据阶梯。
2. 先依赖/L0，再做最小模型 smoke；通过后才允许同一冻结协议训练至 100K 硬上限。
3. 每项保留展开配置、资源账、checkpoint、独立评测、曲线/视频和受限裁决。
4. 阴性结果或可复现工程阻塞均正常结案，不为追求正结果改协议。

## Parked Lanes

- EXP-0009 Figure 18 paired seeds 1、2，以及 10M/14-task 扩展。
- 三个新域的多 seed 与 full-suite 扩展。
- Minecraft 1M、5M 与 100M 预算；当前授权只到 100K。
