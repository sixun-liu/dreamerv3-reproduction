# DEVLOG

> Updated: 2026-07-21T05:51:00Z
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
