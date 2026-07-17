# DEVLOG

> Updated: 2026-07-17T05:31:57Z
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
