# DEVLOG

> Updated: 2026-08-12T02:57:14Z
> Maintainer: codex
> Source of truth: decision synthesis linked to research IDs

只追加持久决策、协议变化、路线升降级、正式裁决和迁移；原始运行输出留在 registry/artifact。

## 2026-07-16

### 2026-07-16T15:50:00Z | decision | crafter-pilot

- Actor: codex
- Summary: 用户批准停止健康但约需 48 小时的 Crafter pilot；单 run 不能复现论文 scaling claim，首个目标改为有官方逐 seed 曲线的 DMC walker_walk。
- Evidence: `/root/autodl-tmp/runs/crafter_baseline_0716/`, checkpoint step 12330, log step 16200
- Next: 完成 walker_walk claim-protocol matrix 后再启动 replication。
- Approval: user

### 2026-07-16T17:14:57Z | protocol | EXP-0001

- Actor: codex
- Summary: 冻结 Nature-era DMC proprio walker_walk：12M、repeat 2、16 env、ratio 512、seed 0、500K environment steps。
- Evidence: EVT-0001; config SHA256 `94d3054f...b1b435`; reference SHA256 `8182860a...cc7f4`
- Next: 先过 10K pilot gate，再继续完整预算。
- Git: runtime `danijar/dreamerv3@e3f0224`

### 2026-07-16T18:26:44Z | result | EXP-0001

- Actor: codex
- Summary: 490K bin median 914.2 进入官方五 seed 包络，但前半程明显落后；裁决 `promising_unresolved`。
- Evidence: EVT-0006, ART-0001--ART-0005
- Next: 用户看图，并核对官方 JSON 的代码与评估生成协议。
- Approval: pending-user-review

## 2026-07-17

### 2026-07-17T05:02:24Z | migration | control-repo-bootstrap

- Actor: codex
- Summary: 建立独立 DreamerV3 control repo；历史 freeze 继续引用官方 runtime 与旧绝对路径。
- Evidence: `MIGRATION.md`, `research/repositories.yaml`
- Next: 后续实验从 clean control/runtime/workflow commit 分别冻结。
- Approval: user
- Git: control `6822079`; workflow `e656c16` (`v0.1.0`); runtime `e3f0224`

### 2026-07-17T05:31:57Z | workflow | control-doc-strategy

- Actor: codex
- Summary: 项目显式采用事件触发式控制文档规范；TODO 只留未完成动作，DEVLOG 改为带 actor/evidence/next 的持久事件。
- Evidence: `researchctl docs --strict` 0 error/0 warning
- Next: 只在方向变化、持久决策或正式结案时更新相应人读视图。
- Approval: user
- Git: workflow `3a5bd50` (`v0.1.1`)

### 2026-07-17T05:43:36Z | workflow | multi-repo-provenance

- Actor: codex
- Summary: 项目采用 schema 2 repository manifest；未来 new/freeze 自动快照 control、runtime 和 workflow，并拒绝 dirty/pinned commit drift。
- Evidence: `research/repositories.yaml`, workflow multi-repo regression tests
- Next: 下一 replication 从多仓 clean snapshot 预注册和冻结。
- Approval: user
- Git: workflow `f3a120e` (`v0.2.0`)

### 2026-07-17T05:47:34Z | workflow | human-review-status-fix

- Actor: codex
- Summary: pending human review 继续保留 strict warning，但不再被误报为需要修复的审计损坏。
- Evidence: workflow 15/15 tests; `researchctl status`
- Next: 下一控制动作直接指向 `human_review:EXP-0001`。
- Approval: user
- Git: workflow `ffc2d66` (`v0.2.1`)

## 2026-07-21

### 2026-07-21T02:40:00Z | result | two-paper-synthesis

- Actor: codex
- Summary: DreamerV3 walker 与 Nature DQN Breakout 均完成单任务单 seed 部分数值复现；新增计算停止，交付重心转为人工图审和论文理解。
- Evidence: DreamerV3 EXP-0001 / ART-0001--ART-0005；DQN EXP-0004 / ART-0019--ART-0027；`reports/TWO_PAPER_REPRODUCTION_SUMMARY.md`
- Next: 用户复核两张主图；Claude 交叉核验研读材料；DMC 官方参考曲线生成谱系继续离线取证。
- Approval: DQN autonomous option 2 approved；human visual review pending

### 2026-07-21T04:53:18Z | protocol | dmc-table4-aggregation

- Actor: codex
- Summary: 恢复 DMC proprio Table 4 聚合为官方五 seed 曲线最后 3 个 10K 点的 mean；18 个任务 RMSE 0.30，walker `935.752 -> 936`。用户批准先评估旧 checkpoint，再运行三个 clean seeds。
- Evidence: `references/DMC_SCORE_PROTOCOL_AUDIT.md`；score SHA256 `8182860a...cc7f4`；author commits `423291a`, `2411f7d`
- Next: 运行 462K checkpoint 的固定 `eval_only` 诊断，随后验证 final checkpoint 管道。
- Approval: user
- Git: runtime `e3f0224` (2026 post-Nature); reference score lineage `423291a` (2023)

### 2026-07-21T05:50:39Z | result | EXP-0002

- Actor: codex
- Summary: 462K checkpoint 的固定 seed stochastic `eval_only` 自然完成；64 个 episode mean 893.48、median 915.44，预注册可用性门通过。两次得分前工程失败分别定位为 EGL 和多环境 `log/image`，均保留且未用于选分。
- Evidence: EVT-0009--EVT-0011；ART-0006--ART-0008；runtime compatibility commit `b98e975`
- Next: 补自然结束 final checkpoint 保存与 smoke，再冻结三个训练 seed；独立 eval 与论文 training-return 表值继续分列。
- Approval: user；human visual review pending

### 2026-07-21T06:11:06Z | result | EXP-0003

- Actor: codex
- Summary: 默认关闭的 `run.save_at_end` 在2048-decision smoke显式开启后保存精确终点checkpoint；独立eval成功加载并完成16个有限episodes，仪器门通过。
- Evidence: EVT-0013--EVT-0014；ART-0009--ART-0010；runtime `5168475`
- Next: 冻结并顺序运行walker_walk seeds 0、1、2的500K environment-step replication。
- Approval: user；smoke score禁止用于性能结论

### 2026-07-21T10:19:02Z | result | EXP-0004

- Actor: codex
- Summary: 三个clean seeds全部自然完成且完整性门通过；final-30K aggregate 785.53±90.34，较官方935.75低16.05%，仅1/3 seed进入官方逐seed范围，预注册性能门失败，裁决`negative`。
- Evidence: EVT-0016--EVT-0019；ART-0011--ART-0015；runtime `5168475`
- Next: 离线对账2023参考曲线代码/配置与2026 runtime、dm-control/MuJoCo及seed语义；不直接进入消融或盲目加seed。
- Approval: user批准三seed自主运行；human visual review pending

### 2026-07-21T13:14:12Z | result | EXP-0005

- Actor: codex
- Summary: 作者旧runtime `2411f7d`单run自然完成；final-30K mean930.72对齐官方935.75，终值门通过；250K median658.26低于官方下限826.85，早期曲线诊断门失败。裁决`promising_unresolved`，曲线形状不作为论文复现总验收。
- Evidence: EVT-0024--EVT-0025；ART-0016--ART-0019；runtime `6642b94`
- Next: 人工图审和独立复算后，在“转入论文理解”和“补两个旧runtime重复验证500K终值稳定性”之间裁决；谱系曲线归因park。
- Approval: user批准本次单run；scientific human review pending
- Git: control freeze `b77b4a1`；runtime upstream `2411f7d` + compatibility `6642b94`

### 2026-07-21T14:23:46Z | protocol | EXP-0006-kl-mechanism

- Actor: codex
- Summary: 用户批准进入机制探索；冻结三臂设计为 seeded walker baseline、只关 free bits 的 E1、以及去 free bits 并把 representation KL 权重从 0.1 恢复到 1.0 的 P4-reconstructed。
- Evidence: runtime `ad49802`；patch SHA256 `db249897...78ca`；bundle SHA256 `dbf7cb61...394b`
- Next: baseline/E1/P4 各顺序运行 seeds 0、1 至 500K environment steps，比较 raw KL、表征指标与性能。
- Approval: user

### 2026-07-21T21:45:00Z | result | EXP-0006-kl-mechanism

- Actor: codex
- Summary: seeded walker三臂各两个配对seed全部自然完成且完整性门通过。E1相对baseline的late raw-KL delta为-0.803/+0.337，预注册双seed同向门失败；P4相对E1的KL ratio为0.0399/0.0161且posterior entropy双seed更低，P4机制门通过。P4同时使重构损失升至2.99/2.46倍、final-30K score再下降约27%/28%。
- Evidence: EVT-0035--EVT-0036；ART-0020--ART-0024；runtime `ad49802`；matrix `/root/autodl-tmp/runs/EXP-0006__walker-kl__matrix-2seed__20260721T142500Z`
- Next: 用户审查两张主图；随后在“恢复论文原P4配置”和“第二代表任务验证跨任务性”之间裁决，不自动补seed。
- Approval: user批准12小时自主矩阵；scientific human review pending
- Scope: 2026作者重实现、seeded DMC walker_walk与P4代码语义；不构成论文Figure6/17数值复现。

## 2026-07-27

### 2026-07-27T04:31:50Z | result | EXP-0007-openloop-prediction

- Actor: codex
- Summary: 六个 EXP-0006 checkpoint 在同一 576-window panel 上完成有限上下文预测。P4 相对 E1 在两个 seed 均表现为更低 H15 KL、更差 prior 和 teacher-forced NRMSE；独立 453 项复算零不一致。来源交叉矩阵同时显示三类模型均有 own-replay advantage，裁决 `promising_unresolved`。
- Evidence: EVT-0038--EVT-0042；ART-0026--ART-0037；panel SHA256 `9fb6a644...36990f`
- Next: 用户审查主图；若继续归因，只预注册能区分 representation specialization 与访问分布难度的匹配/交换分布实验，不自动追加计算。
- Approval: experiment user-approved；scientific human review pending
- Git: control audit `06115a0`；runtime `cdb3d00`

## 2026-08-05

### 2026-08-05T16:06:22Z | protocol | EXP-0008-cheetah-five-seed

- Actor: codex
- Summary: 用户批准夜间无人值守补强论文结果证据；预注册 Cheetah Run 12M、500K environment steps、五 seed 的 Figure 14 / Table 11 作者重实现复现。健康正式运行不按中途分数或曲线终止。
- Evidence: `research/cards/EXP-0008.md`；`docs/reproduction/EXP0008_CHEETAH_PROTOCOL.md`；score SHA256 `8182860a...cc7f4`；runtime `6642b94`；修复后 freeze `EVT-0044`
- Next: 提交推送后运行独立 smoke；通过完整性、ETA 和磁盘门再 detached 启动五 seed 顺序矩阵。
- Approval: user
- Git: protocol/runner commit `43b4b7a`；integrity-gate fix `e466398`；`EVT-0044` supersedes `EVT-0043`

### 2026-08-05T16:57:08Z | protocol | EXP-0008-smoke-gate

- Actor: codex
- Summary: 两个 smoke 均自然训练并写出精确终点 checkpoint，但原 runner 的完整性检查先后把“预算内无完整 episode”和空条件统计 NaN 误判为失败；失败产物完整保留。收紧检查语义后，16,384-decision smoke 离线复核通过，实际主损失 NaN 仍会失败。
- Evidence: `EVT-0045`--`EVT-0046`；step16384 checkpoint SHA256 `56bdc1e4...16aa0`；steady FPS `72.78`；replay ratio `520`；`runs/STATUS.md`
- Next: 按最终 freeze detached 启动五 seed matrix；健康运行不中途停止。
- Approval: within user-approved autonomous run
- Git: verifier fix `47d357b`；final freeze `EVT-0045`

### 2026-08-05T17:00:20Z | protocol | EXP-0008-detachment-recovery

- Actor: codex
- Summary: 首次正式 tag 仅创建 matrix `.started`；nohup child 随 exec session 清理，在 seed0 信标、GPU 进程和训练输出出现前退出。现场保留并标记 `detachment_before_seed0`，不构成 scientific run。
- Evidence: `/root/autodl-tmp/runs/EXP-0008__cheetah-run__five-seed__500k-env__20260805T160000Z.{started,failed}`；GPU query empty；seed0 signal absent
- Next: 只改变唯一 run tag 与 detached transport，重新提交/冻结；使用 named `screen` 启动相同五 seed 协议。
- Approval: normal in-scope recovery；no training outcome observed
- Git: replacement transport/tag commit `514b684`；replacement freeze `EVT-0047`

### 2026-08-05T22:06:08Z | result | EXP-0008-cheetah-five-seed

- Actor: codex
- Summary: 2411f7d 谱系 Cheetah Run 五 seed 全部自然完成并通过完整性门；5/5 后半程均值高于前半程，但 final-30K 聚合 `550.54` 低于官方范围下限 `584.00`，主数值门失败，裁决 `negative`。
- Evidence: `EVT-0049`--`EVT-0050`；`ART-0040`--`ART-0046`；checkpoint hashes 与独立 raw-file 复算全部一致
- Next: 先人工审图；新计算前恢复 Figure 18 两种梯度阻断的精确语义和 known-answer test。
- Approval: 用户批准无人值守五 seed 运行；scientific human review pending
- Git: control analysis `6fe34f3`；runtime upstream `2411f7d` + compatibility `6642b94`

## 2026-08-06

### 2026-08-06T09:58:02Z | result | EXP-0009-reacher-signal-ablation

- Actor: codex
- Summary: Reacher Hard 单配对 seed 的 1M environment-step 三臂 pilot 全部自然完成；真实模型梯度 gate 通过，fixed-bin AUC 为 baseline `867.42`、no-reward/value `391.61`、no-reconstruction `7.84`，定性方向门通过。裁决 `promising_unresolved`，待人工审查与跨 seed 验证。
- Evidence: `EVT-0056`--`EVT-0057`；`ART-0049`--`ART-0052`；working claim `C-0002`
- Next: 用户审查学习曲线和配对行为展示；若继续，只先冻结相同 1M 协议的 paired seeds 1、2，不自动进入 10M 或 14-task 矩阵。
- Approval: 用户批准梯度验证与 1M 三臂 pilot；scientific human review pending
- Git: control freeze `10f76c7`；runtime gradient routing `990123a`；showcase code `f950ce2`

## 2026-08-07

### 2026-08-07T14:13:57Z | migration | workspace-uppercase-layout

- Actor: codex
- Summary: 用户批准将稳定工作区入口迁移为大写目录；DreamerV3 control/runtime 仓进入
  `Code/DreamerV3/`，run、artifact、paper、env、workflow 等切换到大写 canonical 路径，旧根级
  路径作为 compatibility alias 保留。
- Evidence: `/root/autodl-tmp/Discussion/workspace/2026-08-07_uppercase-layout-migration.md`；
  `research/repositories.yaml`；Git worktree repair 与新旧 inode 对账。
- Next: 用新 canonical 路径完成 research audit、环境导入和一个后续实验周期；在此之前不移除 alias。
- Approval: user-approved
- Git: branch `infra/workspace-uppercase-layout`; historical JSONL/freeze unchanged

## 2026-08-11

### 2026-08-11T15:07:45Z | protocol | cross-domain-minimal-loop

- Actor: codex
- Summary: 用户批准将当前主问题切换为 DMC Vision、Atari100K 和 Minecraft Reduced 的串行跨域最小闭环；Figure 18 paired seeds 暂时 parked，三个域不得合并成单一复现主张。
- Evidence: `discussion/2026-08-11_DreamerV3_cross_domain_SERVER_CODEX_RUNBOOK.md`；`research/project_state.yaml`；数据盘 150 GB/可用 108 GB；runtime `6642b94` 与 `5168475` clean clone。
- Next: 完成 P0 审计后只创建并冻结 EXP-0010 DMC Vision 100K gate。
- Approval: user
- Git: control `main@077e8c9`; branch `exp/EXP-0010-dmc-vision-walker`; runtime_2411 `6642b94`; runtime_2026 `5168475`

### 2026-08-11T19:59:51Z | result | EXP-0010-dmc-vision-walker

- Actor: codex
- Summary: 2024 作者重实现谱系的 DMC Vision Walker 通过 100K gate 并续至 1M environment steps；终点 checkpoint 精确为 500K decisions。末 100K 训练均值 `959.43` 进入官方 10-seed 范围 `954.96--964.79`，独立 10 局均值 `956.43`，裁决 `promising_unresolved`。
- Evidence: `EVT-0064`--`EVT-0067`；`ART-0056`--`ART-0062`；完整性通过，fixed-bin AUC `849.43` 进入官方逐 seed AUC 范围，视频 501 帧且运动性检查通过。
- Next: 只创建并冻结 `EXP-0011` Atari100K Breakout；ALE L0 与 2048-decision smoke 通过后再运行 100K decisions。
- Approval: user-approved long-running cross-domain loop；scientific human review pending
- Git: control freeze `d6d15ba` / monitoring `464c116`；runtime `6642b94`

### 2026-08-11T23:23:07Z | result | EXP-0011-atari100k-breakout

- Actor: codex
- Summary: 2026 作者重实现的 50M/ratio256 Breakout 单 seed 完成精确 100K decisions；early/tail
  均值为 `1.286/7.542`，改善 `6.256`，预注册趋势门通过。独立固定 checkpoint 10 局均值
  `9.10`、范围 `5--16`，固定第 0 局视频完整，裁决 `promising_unresolved`。
- Evidence: `EVT-0077`--`EVT-0080`；`ART-0063`--`ART-0070`；正式训练 `54.1 min`，峰值显存
  `24645 MiB`。两次评测前 transport/config 接口失败均保留，不进入科学结果。
- Next: 创建 `EXP-0012`，只按依赖审计 → Minecraft L0 → 模型资源 gate → 最多 100K 串行推进。
- Approval: user-approved cross-domain long run；scientific human review pending
- Git: runtime `5168475`；formal control freeze `ea7c73a`；evaluation repair `2469eab`

### 2026-08-11T23:58:00Z | protocol | EXP-0012-minecraft-reduced

- Actor: codex
- Summary: Minecraft 独立 Python 3.11 环境已建立；作者补丁 MineRL 0.4.4 wheel 通过长度与 CRC
  校验，Java 8/Xvfb/GL、`pip check`、MineRL import 和 JAX 0.6.2 RTX 5090 matmul 均通过。
  作者旧 JAX/CUDA pin 因 Blackwell 不兼容被替换；NumPy/OpenCV 固定为 `1.26.4/4.11.0.86`。
- Evidence: `docs/reproduction/EXP0012_MINECRAFT_PROTOCOL.md`；wheel SHA256 `b04e2cd...c631`；
  environment `/root/autodl-tmp/Envs/dv3-minecraft-2026`；card `EXP-0012`。
- Next: 结果盲冻结后运行真实 32-step L0；通过时再进入 size50m 4096-step 模型资源 smoke。
- Approval: user-approved cross-domain long run；100K hard limit
- Git: runtime `5168475`；control preparation `fde4ce6`

## 2026-08-12

### 2026-08-12T02:57:14Z | result | EXP-0012-minecraft-reduced

- Actor: codex
- Summary: 2026 作者重实现的 Minecraft Diamond size50m 单 seed 精确完成 100K environment
  steps；训练与三回合终点评测均观察到完整 L1。训练出现一次 cobblestone，但 wooden_pickaxe
  从未出现，因此 L2 链未闭合；裁决 `promising_unresolved`，不构成论文约 100M Diamond 复现。
- Evidence: `EVT-0085`--`EVT-0086`；`ART-0071`--`ART-0081`；训练/评测 checkpoint、replay、
  metrics、资源、曲线与固定 episode0 视频均通过完整性检查。
- Next: 用户审查本轮主图与视频后，在已有 replay 的 wooden-pickaxe 瓶颈分析和重新授权的更大
  预算之间只选择一个有界问题；不自动追加训练。
- Approval: user-approved cross-domain long run；scientific human review pending
- Git: runtime `5168475`；formal/evaluation freeze `f76236e`；analysis `3842dda`

### 2026-08-12T05:59:31Z | result | EXP-0013-minecraft-throughput

- Actor: codex
- Summary: 双环境 5040 步完整性、数据盘定向与资源门全部通过，但端到端吞吐仅为历史长程基线
  `0.690x`，尾窗 policy FPS `1.092x` 未通过预注册 `1.10x` 扩档门；`envs=4/8` 因此停止。
- Evidence: `EVT-0090`--`EVT-0091`；`ART-0089`；无 OOM、系统盘净增仅 64 KiB、临时实例已安全清理。
- Next: 另开同为 5040 步、`debug=false` 的单环境配对诊断，避免把运行长度和 debug 差异误归因于并发数。
- Approval: 用户批准当前长期 goal；human review pending
- Git: control analysis `3e22730`；runtime `5168475`

### 2026-08-12T06:35:02Z | result | EXP-0014-minecraft-paired-throughput

- Actor: codex
- Summary: 同为 5040 步、`debug=false` 时，`envs=2` 相对 `envs=1` 的端到端吞吐提升
  `27.36%`，固定稳态 policy FPS 提升 `46.10%`，以明显余量通过预注册 `10%/5%` 双门。
- Evidence: `EVT-0093`--`EVT-0094`；`ART-0090`--`ART-0091`；两臂完整性通过，图像已由 codex 检查。
- Next: 只晋级 `envs=2` 到 EXP-0012 checkpoint/replay/step 等价恢复验证，不外推为策略质量结论。
- Approval: 用户批准当前长期 goal；scientific human review pending
- Git: control freeze `89a2ad9`；analysis `00ce496`；runtime `5168475`

### 2026-08-12T07:09:15Z | result | EXP-0015-minecraft-recovery

- Actor: codex
- Summary: EXP-0012 的 100K checkpoint 与 replay 经独立 inode 的 XFS CoW 克隆后，在
  `envs=2` 下精确恢复并自然结束于 105040；源树哈希不变，旧 100000 个 stepid 全保留，新增
  5040 个唯一 stepid 与两个新 worker 起点全部通过。
- Evidence: `EVT-0095`--`EVT-0097`；`ART-0092`；训练墙钟 `283 s`，稳态 policy FPS
  `40.51`，无 OOM、系统盘泄漏或 Minecraft/GPU 残留。
- Next: 从原始 EXP-0012 另做独立克隆，正式新增 100K 到绝对 200K，再沿用三回合独立评测；
  不续接或混入 105040 诊断输出。
- Approval: 用户批准当前长期 goal；本 cycle 为 infrastructure 诊断，无需科学看图裁决
- Git: control freeze `195d909`；runtime `5168475`
