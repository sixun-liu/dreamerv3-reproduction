# DreamerV3 跨域最小实验计划

> 日期：2026-08-11
> 用途：传给 AutoDL 服务器 Codex，作为新实验的预注册执行说明。
> 当前设备：RTX 5090 32 GB；已验证 JAX 可在 Blackwell 上运行。
> 当前证据：正式实验只覆盖 DMC Proprio，包括 Walker、Cheetah 与 Reacher。下一步希望用最小成本检查像素控制、Atari 和 Minecraft 三类不同环境，不追求会前完成论文级多域复现。
> 服务器裁决（2026-08-11）：用户批准按证据阶梯串行执行；DMC Vision 使用 2024 runtime 的兼容提交，Atari/Minecraft 使用 2026 runtime；Minecraft 未经新授权不越过 1M。数据盘已扩到 150 GB。

## 1. 总体问题

DreamerV3 论文覆盖 DMC Proprio、DMC Vision、BSuite、Atari100K、Atari、ProcGen、DMLab 和 Minecraft 八个域。目前本地结果虽然包含跨任务、多 seed 和受控消融，但都位于 DMC Proprio。新一轮实验只回答三个更窄的问题：

1. **DMC Vision**：把已有 Walker 的本体状态输入换成像素输入后，当前代码与设备能否形成有效策略？
2. **Atari100K Breakout**：DreamerV3 能否在与此前 DQN 工作相同的游戏上跑通像素建模与低数据预算训练？
3. **Minecraft**：Linux 无桌面环境下，MineRL/Malmo 链路能否稳定运行；缩小模型和预算后能否达到木头或石头等低阶里程碑？

这三项分别是**受控跨模态比较**、**与 DQN 的跨算法连接**和**复杂环境可行性检查**，不能合并成一个“多域复现成功”的结论。

## 2. 执行顺序

只允许顺序运行 GPU 任务，不并发占用同一张卡。

| 优先级 | 候选实验 | 初始预算 | 放大上限 | 主要产物 |
|---|---|---:|---:|---|
| P0 | 环境、代码与资源审计 | 每域 200--1000 steps smoke | 无 | 版本清单、真实配置、显存与吞吐 |
| P1 | EXP-0010 DMC Vision Walker | 100K environment steps pilot | 1M environment steps，单 seed | 学习曲线、固定评测、GIF/MP4 |
| P2 | EXP-0011 Atari100K Breakout | 完整 100K interaction 预算，单 seed | 暂不追加多 seed | 训练曲线、独立评测、与 DQN 的协议差异表 |
| P3 | EXP-0012 Minecraft Reduced | L0 smoke，再到 100K steps | 当前授权上限 100K；1M 需新授权 | 环境审计、里程碑成功率、首次达成步数、视频 |
| 备用 | BSuite | 依环境定义的短预算 | 单任务单 seed | 当 Minecraft 依赖失败时提供低成本跨域证据 |

EXP 编号需先与服务器控制仓现状核对；如果服务器已有 `EXP-0010` 及后续编号，依次顺延，不覆盖任何现有记录。

## 3. P0：运行前审计

服务器 Codex 在启动正式任务前必须先完成以下检查，并把结果写入新的研究卡或冻结协议：

1. 记录 runtime 仓库路径、Git commit、dirty status、Python、JAX、CUDA、驱动和环境包版本。
2. 检查服务器实际使用的是哪一代 DreamerV3 代码；不得根据本地批注副本直接假设命令完全一致。
3. 从服务器当前 `configs.yaml` 展开并保存每个候选任务的最终配置，重点核对 task、model size、action repeat、环境数、train ratio、总步数和计数单位。
4. 核对已有 run、checkpoint 和实验编号，禁止复用旧 logdir，也不得为了启动成功修改 reward、action space 或算法语义。
5. 每个域先运行独立 smoke logdir，确认观测、动作、奖励、终止信号、checkpoint 和日志均有效。
6. smoke 期间采样整卡显存、进程显存、GPU 利用率、policy FPS 和训练 FPS。JAX 预分配显存不得直接解释为活跃峰值。
7. 在首个稳定吞吐窗口后估算墙钟时间；只有预算和停止条件仍合理时才放大。
8. 记录数据盘剩余量和 smoke 的每千步磁盘增长；放大前必须满足“预计新增量 + 5 GiB 保底 < 可用空间”。

审计不通过时停止在当前层级，保留错误日志和最小复现步骤，不自行改造环境语义来绕过问题。

## 4. EXP-0010：DMC Vision Walker

### 4.1 研究问题

在同一个 `walker_walk` 任务上，关闭 proprioception、仅使用像素观测后，当前 DreamerV3 链路能否在有限预算内出现明确学习趋势？这项实验的价值是与已有 DMC Proprio Walker 形成相对受控的输入模态比较。

### 4.2 协议冻结

- 候选配置入口：`--configs dmc_vision size12m --task dmc_walker_walk`。
- 冻结 runtime：作者 2024 `2411f7d` 加 JAX 0.6/Blackwell 兼容提交 `6642b941f578cd72147bc2be3c3343d5bc72931c`。该版本 `dmc_vision` 为 size12m、train ratio 512、repeat 2；100K environment steps 对应 50K agent decisions。当前 2026 runtime 的 1.1M steps、ratio 256、repeat 1 属于另一条谱系，不得静默混用。
- 主实验先固定单 seed，不以单 seed 外推稳定性。
- 先运行 100K environment steps pilot；若分数、模型损失和行为均无学习迹象，先诊断而不是直接续跑。
- 100K pilot 与 1M 上限作为同一预注册的两阶段预算冻结；若 pilot 有可辨认趋势且 ETA 可接受，第二阶段只允许改变预声明的总步数/恢复点，算法、超参、seed 和 logdir 保持不变。

### 4.3 指标与停止条件

- 主指标：训练 `episode/score` 的固定环境步分箱曲线和 AUC。
- 辅助指标：固定协议 checkpoint 评测、world model 相关损失、吞吐与资源曲线。
- 行为材料：固定评测条件的 GIF/MP4；视频只展示行为，不单独承担结论。
- 立即停止：NaN/Inf、持续 OOM、日志计数异常、观测并非纯像素、reward 或 action 语义异常。
- 100K 后裁决：若没有趋势且工程门通过，记录为短预算阴性结果；不得仅因想要正结果而事后改口径。

### 4.4 允许的结论

- 允许：当前代码和设备下的像素 Walker 可行性、单 seed 学习趋势，以及与已有 proprio 结果的描述性差异。
- 不允许：DMC Vision 域整体复现、像素输入优于或劣于状态输入、跨 seed 稳定性。

## 5. EXP-0011：Atari100K Breakout

### 5.1 研究问题

在此前 DQN 已完成 Breakout 复现的基础上，检查 DreamerV3 的世界模型路线能否在 Atari100K 的有限交互预算下学习同一游戏。这里关注的是跨算法连接，不做 DQN 与 DreamerV3 的直接排行榜比较。

### 5.2 协议冻结

- 候选配置入口：`--configs atari100k size50m --task atari100k_breakout`。
- 冻结候选 runtime：作者当前 2026 upstream `e3f0224` 加本地两项运行期补丁的 `5168475b7a4413f9575933b4580e7073caea2114`。该 upstream 修复了 Atari reset frame max-pooling；配置使用 `64x64` 彩色观测、action repeat 4、无 sticky action、随机 0--30 no-op 和 minimal action set。100K agent decisions 约对应 400K emulator frames。
- 先做 200--1000 steps smoke，再执行单 seed 完整 Atari100K 预算。
- 若 `size50m` 在真实活跃显存或吞吐上不可接受，先记录资源门失败；只有在协议中明确降档目的后才能改用 `size25m` 或 `size12m`，并把结果标为降规模可行性实验。

### 5.3 指标与比较边界

- 主指标：训练分数曲线、固定 checkpoint 独立评测的均值与分布。
- 必须记录 environment steps、agent decisions 与 emulator frames 的换算，避免复用 DQN 时的帧数歧义。
- 与 DQN 的比较只展示方法与协议差异：输入尺寸、frame stack/latent state、replay、模型规模、训练预算和评测方式。
- 不用一次单 seed 结果声称 DreamerV3 优于 DQN，也不把现代 ALE 下的两个运行当作论文间严格对照。

## 6. EXP-0012：Minecraft Reduced

### 6.1 定位

Minecraft 不是符号内核或桌面自动点击。当前 wrapper 依赖真实 Minecraft/MineRL 风格的 Java 游戏逻辑，在 Linux 中通过 headless 渲染产生 `64x64` 观测，并使用抽象动作与加速挖掘设置。论文挖钻石约需 `1 A100 x 9 days` 和 100M 量级环境步，本轮不承担该复现目标。

### 6.2 分层里程碑

| 层级 | 验收对象 | 最低证据 |
|---|---|---|
| L0 | 环境可用 | reset/step 正常；图像非空；动作、reward、inventory、done 更新；可保存视频 |
| L1 | 木头链路 | 固定世界集合上的 obtain wood/planks/crafting table 成功率和首次达成步数 |
| L2 | 石头链路 | wooden pickaxe 与 obtain stone 的成功率和首次达成步数 |
| L3 | 铁链路 | obtain iron 或 iron pickaxe |
| L4 | 钻石 | 只保留为远期论文目标 |

### 6.3 预算与放行

1. 使用 2026 runtime `5168475`，先审计 Java、MineRL/Malmo、Xvfb/OpenGL、`numpy` 约束和当前 wrapper 的真实安装方式。Minecraft 依赖必须放入独立环境，不得污染已验证的 `dv3` 环境。
2. 候选入口为 `--configs minecraft size50m --task minecraft_diamond`。本地 wrapper 显示原始 diamond 任务已经对 log、planks、crafting table、wooden pickaxe、cobblestone、iron 与 diamond 等中间产物提供一次性奖励，因此本轮按里程碑评测即可，不需要修改奖励函数。
3. 完成 L0 后，从 `size50m` 开始做显存与吞吐 smoke；无法容纳时按 `size25m -> size12m` 降档，并明确结果不对应论文 200M 模型。
4. 当前训练授权只到 `100K`。只有 L0、模型 smoke、资源和 ETA 门全部通过才启动；`1M` 与 `5M` 均保持 parked，必须分别获得新授权。
5. 100K 若未达到 L1/L2，仍按阴性早期结果结案，不因无人值守而自动追加预算。
6. 不得通过修改奖励、预置物品、缩短任务逻辑或加入人工演示来把“环境跑通”包装成 DreamerV3 学会任务。

### 6.4 评测

- 使用预先固定的世界或评测 seed 集合。
- 报告每个里程碑的成功率、首次达成 environment step、回报分布和失败类型。
- 视频必须与评测记录绑定，不挑选单个成功视频代替统计结果。
- 若只完成 L0，结论就是“Minecraft 环境链路可用”；若 100K 或 1M 未达到 L1/L2，也应如实保留阴性结果。

## 7. BSuite 备用路线

只有 Minecraft 环境依赖在合理时间内无法解决时，才启用 BSuite 作为低成本跨域备选。服务器先确认可用任务、episode 上限和官方评测方式，再选一个能与价值学习或记忆机制对应的任务。BSuite 的意义是补一个不同问题结构的廉价域，不是替代 Minecraft 的像素与复杂行动证据。

## 8. 统一归档要求

每个正式实验至少保留：

1. 研究卡、冻结协议、展开后的最终配置和源码 commit。
2. 原始 `metrics.jsonl`、`scores.jsonl`、终点 checkpoint、stdout/stderr 和资源采样。
3. 独立重算脚本及机器可读摘要；主数字不能只从展示图人工读取。
4. 固定评测协议、评测 seed、完整回合结果、GIF/MP4 及媒体 provenance。
5. 一页结果说明，严格分开观察、解释、边界和最终裁决。
6. 实验完成后再打包传回本地；不要只传曲线截图。

## 9. 给服务器 Codex 的首轮任务

首轮只执行以下内容，不直接启动三个正式长跑：

1. 读取本计划和服务器仓库的 `AGENTS.md`、实验控制文档与 scoreboard。
2. 完成 P0 审计和三个候选协议草案，但始终只允许一个 active experiment；不得同时在 registry 中冻结三张正式卡。
3. 按 DMC Vision → Atari100K Breakout → Minecraft 的顺序逐项建立、冻结和运行 smoke；每项关闭后才创建下一项，记录是否通过、资源占用、吞吐和阻塞原因。
4. 基于 smoke 结果优先启动 EXP-0010 的 100K DMC Vision pilot。
5. 把 P0 报告和 EXP-0010 启动信息落盘；用户已批准长期作业，可按冻结 gate 串行继续，但不得自行越过 Minecraft 100K 当前授权上限。

## 10. 会前最低可交付结果

最理想的会前结果是：

- 一条 DMC Vision Walker 学习曲线与固定评测视频；
- 一条 Atari100K Breakout 单 seed 可行性结果；
- 一份 Minecraft 环境与算力审计，最好完成 L0，若 100K 有早期信号再追加展示。

即使只完成其中一项正式训练，只要协议、指标和边界完整，也比同时启动多个未闭环长跑更有价值。现有 EXP-0001 至 EXP-0009 已足以支撑 DreamerV3 阶段汇报，新实验负责扩大证据类型，不负责用数量堆叠进展。
