# EXP-0010 DMC Vision Walker 协议

> Updated: 2026-08-11T16:00:00Z
> Maintainer: codex
> Source of truth: `research/experiments.jsonl` 与本协议引用的展开配置

## 问题与假说

- 问题：在 DMC `walker_walk` 只编码 `64x64` RGB 像素时，2024 作者重实现能否在 100K environment steps 内出现明确学习，并在 gate 通过后于 1M 达到公开曲线的单 seed 数值包络？
- 假说：100K 末 20K 平均分至少 200，且相对首 20K 提升至少 100；通过后续训至 1M，末 100K 平均分进入官方 10-seed 末段范围。
- 替代解释：短程改善可能来自单次 environment RNG、训练 episode 采样差异或公开曲线平滑口径，而非稳定数值复现。
- 唯一实验变化：相对既有 DMC Proprio Walker，encoder/decoder 的消费空间切换为 `image`；本实验内部不改算法、模型规模、train ratio、repeat 或 seed。

## 冻结协议

| 字段 | 值 |
|---|---|
| Reproduction kind | `author_reimplementation` |
| Runtime | `dreamerv3-runtime-2411-crossdomain@6642b941...` |
| Task | `dmc_walker_walk` |
| Config | `dmc_vision + size12m` |
| Observation | `64x64x3` RGB；`enc.spaces=image`、`dec.spaces=image` |
| Agent seed | smoke 31415；formal 0 |
| Environment seed | 不受 agent seed 控制，作为限制报告 |
| Action repeat | 2 |
| Environments | 16 |
| Train ratio | 512 |
| Pilot | 50K decisions = 100K environment steps |
| Full | 同一 logdir/replay/checkpoint 续至 500K decisions = 1M environment steps |
| Reference | `scores/dmc_vision-dreamerv3.json.gz`, SHA256 `68dd7e...b08` |

完整配置分别为：

- `configs/exp0010_dmcvision_smoke_s31415_8192_dec.yaml`
- `configs/exp0010_dmcvision_s000_100k_env.yaml`
- `configs/exp0010_dmcvision_s000_1m_env.yaml`
- `configs/exp0010_dmcvision_staged_manifest.yaml`

## 证据与完成信号

1. Smoke：checkpoint、replay、metrics、scores、像素 checkpoint 评测视频和完整性检查均存在；无 NaN/OOM/Traceback。
2. Pilot：checkpoint step 为 50K decisions；保留独立 `checkpoint_100k.ckpt`；原始 scores/metrics 与资源采样完整。
3. 100K continuation gate：
   - 完整性通过；
   - `80K < environment step <= 100K` episode mean `>= 200`；
   - 上述末窗口比 `0 < step <= 20K` episode mean 至少高 100；
   - 按 pilot 实测估算，额外墙钟不超过 12 小时；
   - 线性外推新增数据后仍保留至少 5 GiB。
4. Full：checkpoint step 为 500K decisions；`900K < step <= 1M` 窗口、0--1M 曲线/AUC、官方 10-seed 包络和独立 10-episode checkpoint eval 均输出。
5. 视觉证据：固定选择评测序列中的第 1 局生成 MP4 和首帧 PNG，不用最佳局替代统计。

## 停止条件

- 任意非预期 NaN/Inf、OOM、CUDA/XLA 致命错误、checkpoint/replay 不可恢复或计步语义不符时停止当前层级。
- smoke 未通过不启动 pilot；pilot gate 未通过不续至 1M。
- GPU 同时出现其他 compute process、预计磁盘跌破 5 GiB 或续训 ETA 超过 12 小时时停止放大并结案/诊断。
- 不为追求正结果更换 seed、降低模型、修改奖励、action space 或超参数。

## 裁决边界

- 100K gate 只决定是否放大，不等于论文数值复现。
- 单 seed 进入官方 seed 包络最多支持“该作者重实现谱系在本任务形成数值对齐实例”；不支持跨 seed 稳定性或 DMC Vision 域整体复现。
- 若完整协议下落在包络外，结论是本地单 seed 数值门未通过；不得据此否定 DreamerV3 或像素世界模型。
