# CURRENT_STATE

> Updated: 2026-08-12T06:00:16Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` 已完成 100K 受限闭环；Minecraft 临时实例已安全迁移到数据盘。`EXP-0013` 的
`envs=2` 虽完整运行，但未通过预注册扩档门，当前只需用同预算单环境对照完成并发收益归因。

## 当前主要矛盾

EXP-0013 已证明双 MineRL/Malmo 实例可以只写数据盘并安全清理，但 5040 步双环境的端到端吞吐
仅为 `17.03 steps/s`，尾窗 policy FPS 相对历史长程基线仅 `1.092x`，低于 `1.10x` 扩档门。
历史基线同时混有 100K 长程和 `debug=true` 差异，因此还不能把结果因果归因于环境数量。

## 下一项决策

只补一个 5040 步 `envs=1/debug=false` 配对诊断，并用同口径端到端与固定稳态窗口比较 `envs=2`。
若没有足以改变成本判断的收益，就固定单环境资源安全方案并转入 EXP-0012 等价恢复验证。
