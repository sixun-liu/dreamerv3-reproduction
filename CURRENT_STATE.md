# CURRENT_STATE

> Updated: 2026-08-11T15:07:45Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

用户已批准串行建立 DMC Vision、Atari100K 与 Minecraft Reduced 的跨域最小证据闭环；
数据盘扩至 150 GB，两个目标 runtime 已固定为 clean clone，当前无 active experiment。

## 当前主要矛盾

既有正式证据只覆盖 DMC Proprio，尚不能回答作者重实现能否在像素控制、Atari 低数据预算和复杂
Minecraft 环境上形成可复核行为。三个域的 runtime 与协议不同，必须逐项冻结和裁决，不能合并成
“DreamerV3 多域复现成功”。

## 下一项决策

完成 P0 版本/环境/资源审计后，只创建 `EXP-0010` DMC Vision Walker。先执行 100K environment-step
gate；完整性、趋势、ETA 和磁盘门通过时，按预注册第二阶段续至 1M。Figure 18 paired seeds 已 parked。
