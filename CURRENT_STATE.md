# CURRENT_STATE

> Updated: 2026-08-12T05:08:54Z
> Maintainer: codex
> Source of truth: research/project_state.yaml and research/experiments.jsonl

机器状态请运行 `researchctl status`；本文只保存人工综合。

## 一句话判断

`EXP-0012` 已完成 100K 受限闭环并建立单环境吞吐基线；当前新目标是在不改变算法语义的前提下，
先解决系统盘风险并寻找本机多环境吞吐甜点，再依据实测性价比决定是否进行有界增量训练。

## 当前主要矛盾

EXP-0012 使用 `envs=1/debug=true`，100K 墙钟 4053 秒、平均 GPU 利用率 8.88%，说明训练主要受
Minecraft/CPU 等待限制；同时系统盘只剩约 3.8 GiB，而 MineRL 默认把实例临时目录写入 `/tmp`。
当前主要矛盾是先建立资源安全、可审计且稳定增益的运行方案，而不是直接扩大训练预算。

## 下一项决策

完成数据盘临时目录重定向和自适应短吞吐探针；若存在稳定且足以改变成本判断的增益，再另行验证
EXP-0012 的 checkpoint/replay/step 恢复并冻结合适的增量预算。吞吐探针不计入论文结果。
