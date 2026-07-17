# AGENTS.md · DreamerV3 复现项目契约（服务器侧）

> 2026-07-16 由 Claude（沙盒侧）交接。本文件 = 服务器侧 agent（codex）的项目上下文 + 纪律 + 分工契约。

## 零、仓库边界

- 本目录 `/root/autodl-tmp/dreamerv3-reproduction` 是 control repo，只保存研究状态、配置、分析和文档。
- 官方 runtime 位于 `/root/autodl-tmp/dreamerv3`，固定上游 commit；训练必须从 runtime 启动。
- runs/checkpoint/artifact 位于 Git 外的数据盘。仓库角色真源见 `research/repositories.yaml`。
- `EXP-0001` 早于本 control repo，历史 freeze 继续引用旧 runtime 路径；不得把首次导入提交伪装成 pre-run commit。

## 一、任务与验收（导师口径）

- 任务：**复现 DreamerV3（arXiv 2301.04104 = Nature 2025）paper 结果**，A 档：跑通官方码、曲线/分数对齐论文，开源码合规可用。
- 当前状态：**论文理解与协议恢复阶段**。`crafter_baseline_0716` 已于 2026-07-16 23:50 CST 经用户批准停止，日志、replay 和 step 12330 checkpoint 完整保留；GPU 当前空闲。详见 `/root/autodl-tmp/runs/STATUS.md`。
- **验收标准待冻结**：runtime 的 `/root/autodl-tmp/dreamerv3/scores/` 含 DMC、Atari、DMLab、Minecraft、ProcGen 官方参考 JSON，**不含 `crafter*.json.gz`**。Crafter 只出现在论文 scaling 分析中，单个 200M/ratio-512 run 不能独立复现该结论。下一次长跑前必须绑定论文版本、目标图表/主张、官方参考产物和评测协议。
- 后续顺序：完成 claim-protocol 对账 → 选择一个低成本论文结果 → smoke/pilot → 正式 replication → 基线确认后再进入消融矩阵（见 §四）。

## 二、服务器纪律（前人血泪，SSH_GUIDE 精华，违者必踩坑）

1. **启动实验 = detached + 文件信标判重**（`runs/<tag>.started`），**绝不因超时/断连重试启动**——重试 = 双进程抢显存。
2. **判忙/清场以 `nvidia-smi --query-compute-apps=pid` 为准**，pgrep 有自匹配三变体假阳性，只用于杀进程。
3. 启动前清 `/dev/shm`（`rm -f /dev/shm/torch_* /dev/shm/sem.loky-* /dev/shm/cuda.shm.*`）——SIGKILL 遗留会死锁下个 run。
4. 大产物（logdir/checkpoint/数据）一律放 `/root/autodl-tmp/`（50G 数据盘），系统盘仅剩 ~10G。
5. 每个 run 必须有：`runs/<tag>.started` 信标 + `runs/<tag>.freeze`（commit/config/seed/时间）+ 日志。

## 三、环境事实（已验证，勿动）

- conda env **dv3**（py3.12）：`jax[cuda12]==0.6.2` —— **勿升级 jax≥0.7（必挂）、勿按 requirements.txt 装 0.4.33（不支持本卡 sm_120）**。
- 镜像自带 CUDA 12.1 陈旧无碍（jax 轮子自带 runtime，驱动 580 已验）。
- 外网（GitHub 等）走本机代理：`export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890`（用完 unset，pip 走阿里云镜像不需要代理）。
- 冒烟验证已全绿（DEVICES/MATMUL/SMOKE-CRAFTER-OK，见 `/root/setup_dv3.log`）。

## 四、后期消融实验矩阵（复现闭环后再进入）

基线复现闭环后，按 `research-agent-kit` 的 new(预注册)→freeze→observe→close 循环逐个做，**每个实验先预注册假说再跑**。组合消融用于判断机制组是否重要；因果归因时再拆成单部件实验：

| 消融 | 预注册假说 |
|---|---|
| 关 symlog/twohot（换回归） | 奖励尺度大的域训练崩——尺度无关化的直接证据 |
| 关 free bits | 表征塌缩、世界模型失效 |
| 关 unimix（注意有三处：RSSM 先验/后验/离散 actor） | 数值稳定性劣化 |
| 关 replay critic 0.3（loss_scales.repval=0） | 复刻社区 torch 版缺陷，分数下降的实证归因 |
| 关百分位回报归一化 | 稀疏奖励域探索停滞 |

小配置（debug/size12m 短程）先扫趋势，有信号再放大。**实验选择与结果解读权在用户**——你产出数据与现象记录，裁决写进 close 时保持"观察与解释分离"。

## 五、分工契约（双 agent 不踩脚）

- **服务器执行权（启动/杀进程/实验循环/researchctl）：codex 独有**。Claude（沙盒侧）不再启动任何进程，只读查询（看日志/拉数据）。
- **研读/日报草稿/独立验收：Claude**。codex 的状态与产出请落两处：`/root/autodl-tmp/runs/STATUS.md`（当前所有 run 一览，勤更新）+ `/root/autodl-tmp/artifacts/`（图/表/结论）——Claude 定期只读拉取汇总进给导师的日报。
- codex 报"复现成功/实验结论"时，Claude 会做独立抽验（拉原始 metrics 复算）——两侧互为审计，不是不信任，是流程。
- 日报口径（给导师）：如实写"agent 辅助工程与实验执行，用户主导实验设计与分析"。

## 六、当前一手信息位置

| 什么 | 哪里 |
|---|---|
| 基线 run 日志/信标/freeze | `/root/autodl-tmp/runs/crafter_baseline_0716*` |
| 部署全程日志 | `/root/setup_dv3.log` |
| 官方论文基线曲线 | `/root/autodl-tmp/dreamerv3/scores/` |
| 两篇论文与 DreamerV3 arXiv 源码 | `/root/autodl-tmp/papers/` |
| 工作流 kit | `/root/autodl-tmp/research-agent-kit/`（README + SKILL.md） |
| 代理备忘 | `/root/PROXY_NOTE.txt` |

## 七、现场注记（2026-07-16 23:50 更新）

- 原基线 run（pid 8616）启动于旧 `/root/dreamerv3`，目录搬移后 cwd 为 `(deleted)`，但绝对 logdir 和 checkpoint 正常。成本/协议审计后已由 codex 发送 `SIGTERM` 并确认 GPU 释放；**不得因看到 `.started` 信标而自动恢复**。
- 可恢复 checkpoint：`/root/autodl-tmp/runs/crafter_baseline_0716/ckpt/20260716T233947F045849`，checkpoint step 12330；停止前最后日志 step 16200。
- 服务器执行权仍归 codex 独有（§五），Claude 负责研读、材料补充和独立验收。任何新 run 必须从 `/root/autodl-tmp/dreamerv3` 启动，并先完成 replication 卡与 freeze。

## 八、研读资料（Claude 推送）

- docs_from_claude/_研读/：研读报告(定档依据) · 代码走读452行(消融改键 file:line 地图,实验前必读) · DQN前置桥
- 论文 PDF 自取：走代理 curl arxiv.org/pdf/2301.04104 与 /pdf/1312.5602 到 docs_from_claude/_papers/

## 九、历史执行建议（已被当前 workflow gate 取代）

**当前裁决（用户 2026-07-17 最新指令）**：Claude 不掌握当前 workflow；以下内容仅保留为交接历史，不构成执行授权。实际顺序以 `research/project_state.yaml`、`CURRENT_STATE.md`、`TODO.md` 和 replication lifecycle 为准。Claude 无需裁决 DMC 协议，Codex 按 workflow 独立恢复并冻结。

**状态更新**：Crafter 基线 run（pid 8616）因 48h 过长已由用户裁决停止——§七的"绝不能动"随之失效。开新 run 前先清场验证：`nvidia-smi --query-compute-apps=pid` 应为空 + 清 `/dev/shm`（§二-3）。

**立即行动（按序）**：
1. **补依赖**（DMC 任务需要，当时精简安装未含）：`conda activate dv3 && pip install dm_control`（阿里云镜像已配；若 mujoco 相关报错再 `pip install mujoco`）。
2. **启动首个 DMC proprio 复现 run**（小时级出完整曲线——第一个"复现 paper 结果"数据点）：
   - 配置：官方默认 `--configs dmc_proprio --task dmc_walker_walk`（勿改 size/超参，复现口径）
   - logdir：`/root/autodl-tmp/runs/dmc_walker_walk_0717`；信标 + freeze 记录照 §二-5
   - 短冒烟（`--run.steps 500` 另一 logdir）确认 dm_control 通了再开正式 run
3. **跑完验收**：与 `scores/` 内官方 DMC proprio 曲线（walker_walk）画同图对比，图与结论写 `/root/autodl-tmp/artifacts/`，状态更新 `runs/STATUS.md`。
4. **Crafter 官方配置重排周末档**（48h 档期，周五晚启动周日收）——同样官方默认配置不动，复现口径。
5. 若 walker_walk 顺利，可再排 1-2 个 DMC 任务（如 cartpole_swingup / cheetah_run）加密复现证据。

**提醒**：jax 勿动（§三）；每一步"观察与解释分离"记录，后续消融矩阵（§四）等基线全部定标后与用户对齐再开。

## 十、Discussion 协作入口

- `discussion/claude/`：Claude 写研读补充、独立验收和回复；Codex 只读。
- `discussion/codex/`：Codex 写服务器审计、协议问题和运行观察；Claude 只读。
- `discussion/INDEX.md`：当前未决线程索引，不保存 canonical state。
- 经核验的事实再进入 `docs/reproduction/CLAIM_PROTOCOL_MATRIX.md`；持久决策进入 `DEVLOG.md`；run 状态进入 `/root/autodl-tmp/runs/STATUS.md`。
