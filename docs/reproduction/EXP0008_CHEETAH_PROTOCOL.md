# EXP-0008 Cheetah Run 五 seed 复现协议

> Updated: 2026-08-05T16:06:22Z
> Maintainer: codex
> Source of truth: research/experiments.jsonl

## 目标

复现 arXiv:2301.04104v2 的 DMC proprio Cheetah Run 结果：Figure 14 学习曲线和
Table 11 的 DreamerV3 分数 `614`。本轮使用作者 2024 年 `2411f7d` 重实现及仅用于
JAX 0.6/RTX 5090 兼容的 commit `6642b94`，证据等级为 `author_reimplementation`。

## 冻结协议

- `dmc_cheetah_run`、proprio、12M、action repeat 2、16 envs、replay ratio 512。
- 每 seed 250K agent decisions，即 500K environment steps；agent seeds `0..4` 顺序运行。
- 官方参考为 `dmc_proprio-dreamerv3.json.gz`，SHA256 `8182860a...cc7f4`。
- 五个 expanded config、hash、顺序、成本和门槛见 `configs/exp0008_cheetah_matrix.yaml`。

## 验收与限制

完整性门要求五个 run 自然结束、日志有限、配置一致、checkpoint 精确为 step 250000。
主数值门使用每 seed 在 `(470K, 500K]` 的完整 training episodes 均值，再跨五 seed 求均值；
该值需进入官方五个 seed 的末三曲线点均值范围 `[583.995404, 651.095806]`。

官方 `614` 是每个 seed 的 470K/480K/490K 三个导出曲线点先求均值，再跨 seed 聚合。
官方未公开 raw-log 到 JSON 的导出脚本，而本地 16 个同步环境使部分 10K bin 为空，因此本地
末 30K raw episode 统计与官方曲线统计只做语义接近的数值比较，不声称逐样本同构。学习曲线
只作为预注册的时序证据，不以形状肉眼相似作为单独成败门槛。

2411 runtime 没有把 agent seed 传入 DMC 环境构造；五次运行提供独立重复和 agent 初始化变化，
但不能解释为完全可重放、agent/env 配对的 seed 实验。

## 无人值守规则

16,384-decision smoke 先验证环境、训练、完整 episode、checkpoint、配置比对、ETA 和磁盘增长。
正式矩阵健康后不因中途分数、
曲线形状、断线或正常耗时终止；只在冻结的 OOM、NaN/Inf、traceback、GPU 冲突、完整性漂移或
数据盘低于 8 GiB 时安全停止。正式结案前不追加别的实验。
