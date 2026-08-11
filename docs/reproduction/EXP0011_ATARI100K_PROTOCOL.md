# EXP-0011 Atari100K Breakout 协议

> Updated: 2026-08-11T20:10:00Z
> Maintainer: codex
> Source of truth: `research/experiments.jsonl` 与本协议引用的展开配置

## 问题与边界

- 问题：2026 作者重实现的 `size50m / train_ratio=256` 降规模路线，能否在 Breakout 的
  100K agent decisions 内形成可复核学习趋势和终点行为？
- 假说：完整性通过；末 40K emulator frames 训练回报均值至少 3.0，且比首 40K 至少高 1.0；
  固定终点 checkpoint 的独立 10 局评测与视频完整。
- 这不是论文 `size200m / ratio=128 / 5 seeds` 的严格数值复现。官方曲线和表格只提供上下文，
  本地结果最高只能支持降规模作者重实现的单 seed 可行性。

## 冻结协议

| 字段 | 值 |
|---|---|
| Runtime | `dreamerv3-runtime-2026-crossdomain@5168475b...` |
| Task | `atari100k_breakout` |
| Model | `size50m` |
| Observation | `64x64x3` RGB |
| Action | ALE minimal action set；categorical policy |
| Reward | raw game score；不 clip |
| Environment | repeat 4；sticky false；每局随机 0--30 no-op；训练 RNG 不显式固定 |
| Formal seed | agent seed 0；单环境 |
| Train ratio | 256 |
| Smoke | r2 为 4,090 decisions；必须越过 1,024-decision replay warmup 并留下训练 loss |
| Formal | 100,000 decisions = 400,000 nominal emulator frames |
| Eval | agent seed 10000；environment/no-op seed 20260812；10 局；第 0 局固定视频 |

展开配置：

- `configs/exp0011_atari100k_smoke_r2_s31415_4090_dec.yaml`，SHA256
  `1a69657051612140d21479788854dfb9215849643546f5971dff2d14376aa6c1`
- `configs/exp0011_atari100k_s000_100k_dec.yaml`，SHA256
  `014da22ccad6427836fa6b1b7af40d0e49758cd81d1f1b25fc41aef5024ed7f4`

参考与 ROM：

- 官方曲线：`scores/atari100k-dreamerv3.json.gz`，SHA256
  `c839ef2fd28a5c4b7c6f36151a8d16cb41d4d2fb72338c265850b2b4dd5b5e15`
- Breakout ROM：MD5 `f34f08e5eb96e500e851a80be3277a56`，SHA256
  `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`。ROM 只存 Git 外，
  用户已批准其许可证，不进入 artifact bundle 或 GitHub。

## 证据阶梯

1. ALE L0：使用相同图像、动作、repeat、sticky、no-op 与 raw reward 配置；以 L0 专用短 time
   limit 强制检查 first/last/terminal/reset，保存非空视频。
2. 模型 smoke r2：精确运行 4,090 decisions；checkpoint、replay、post-warmup loss、资源采样和完整性
   全部通过；按 steady policy FPS 估算正式 ETA 不超过 12 小时，线性磁盘外推后至少保留 5 GiB。
3. 正式训练：精确 100K decisions；不根据中间分数换 seed、模型、ratio 或奖励。
4. 独立评测：固定 agent/environment seed 完成 10 局，报告完整分布和固定第 0 局视频。

## 指标与裁决

- 训练曲线横轴使用 logger 已乘 repeat 4 的 emulator frames。
- early：`0 < frame <= 40K`；tail：`360K < frame <= 400K`。
- 趋势门：tail mean `>= 3.0` 且 `tail mean - early mean >= 1.0`。阈值在看正式结果前冻结，
  依据论文随机分约 1.7、PPO 约 3 与 Dreamer 表格约 10 的量级，只判断可辨认学习趋势。
- 官方五 seed tail 均值范围约 `6.20--11.00`，仅作描述性背景，不加入本地趋势门。
- 终点独立评测与论文训练曲线协议不同，必须单独列出，不直接替代论文数字。

## 停止条件

- ROM/hash、图像、动作、reward 或 terminal 语义不符时停在 L0。
- smoke OOM、异常 NaN/Inf、无 replay、无 post-warmup loss、checkpoint step 不精确、ETA 超过 12 小时
  或预计磁盘低于 5 GiB 时，不启动 formal。
- formal 非零退出、重复 GPU 进程或完整性失败时保留原始证据并结案/诊断，不自动重启。
- 阴性趋势结果正常结案；不为追求正结果追加 seed 或改变协议。

## 启动层恢复记录

- 首个 smoke tag `...smoke-2048-dec__20260811T201000Z` 在 0 秒、构造环境/模型之前失败：
  2026 runtime 使用顶层 `--script train`，而 runner 误用了 2024 接口 `--run.script train`。
- 失败目录和信标保留。replacement 只修正版本化 CLI 字段并更换 tag；seed、配置、预算和算法不变，
  且在任何训练 outcome 出现前重新提交与 freeze。
- replacement r1 自然训练结束，但揭示 2026 runtime 的 checkpoint 是 `train/ckpt/<timestamp>/`
  目录，driver 以 10 decisions 为粒度把请求的 2,048 推进到 2,050；默认 120 秒日志周期也没有留下
  post-warmup loss。该运行只支持工程诊断，不用于科学裁决。
- r2 将 smoke 调整到可精确到达的 4,090 decisions，并把 `log_every` 改为 15 秒。正式 100K 协议、
  seed、算法和趋势门不变；目录 checkpoint 校验同时要求 `agent.pkl`、`step.pkl` 与 `done`。

## 允许结论

- 允许：当前 2026 作者重实现、降规模配置和现代 ALE 下的单 seed 工程可行性与学习趋势。
- 不允许：论文 200M 配置严格复现、跨 seed 稳定性、DreamerV3 优于 DQN 或 Atari100K 全域复现。
