# 协同对齐：research-agent-kit 精读收获 + walker_walk 复现卡（初稿）

> 2026-07-17。kit 已镜像至本地 `learn/research-agent-kit/`。本文 = 双 agent 术语/流程对齐 + 首个复现 run 的 Claim-Protocol Matrix 初稿（知识侧字段我填，执行侧留 codex 实测补）。

## 一、结论：契约与 kit 原生纪律高度同构，协同无冲突

- kit 的**服务器长跑纪律**（GPU 唯一判忙 / started 信标 / 绝对 logdir / detached / freeze / STATUS 页 / **"进程控制只能有一个执行方，其他协作者只读"**）与 AGENTS.md §二/§五 逐条对应——两套规矩本是一套。
- kit 的 `paper-reproduction.md` 就是为当前任务写的：**understanding → reproduction → exploration** 的阶段门与导师"先理解，然后复现"一字不差。
- 它还教科书式预言了 Crafter 事件："**长跑前必须用实测吞吐计算 ETA；pilot 证明不符合时间预算就调整目标，不能为了'已经开始'继续烧卡**"——我们用 36 分钟学费买到的教训，它写在第 85 行。此后一切正式 run 先跑 pilot 算 ETA。

## 二、阶段定位与缺口自查（按 kit 的 understanding 退出门）

| 退出门 | 状态 |
|---|---|
| 论文身份固定（版本/hash） | ✅ 且有增量发现（arXiv v2=Nature 口径收敛） |
| 主张拆解绑定章节/公式 | ✅ 研读报告 + WM1_f1 映射表 |
| 代码谱系（勿用"官方"抹平） | ✅ 三版本地图（2023 老码 / Nature 重写 HEAD / torch 复刻，含漂移记录）——恰是 kit 要求的姿势 |
| 协议恢复（评测 episode/聚合/预算语义） | ⚠️ **缺口①**：训练协议在走读里，评测协议细节未专门整理 → 由复现卡"unknown+取证动作"补 |
| 成本包络 | ✅（且 Crafter 实测教训已内化） |
| 首个最小复现目标 | ✅ dmc_walker_walk（代表性：proprio 域小时级可完成） |
| claim-protocol matrix | ⚠️ **缺口②** → 本文 §三 即首张卡 |

## 三、复现卡：`dmc-walker-walk-repro-01`（Claim-Protocol Matrix）

| 字段 | 内容 |
|---|---|
| Claim ID | `dmc-walker-walk-repro-01`：DreamerV3 官方默认 dmc_proprio 配置在 walker_walk 上达到论文 DMC proprio 基准曲线包络 |
| Source | Nature 2025 版 DMC proprio 汇总结果（具体图号待核 → 取证：论文 PDF 图表核对） |
| Reference artifact | 仓内 `scores/` 对应 dmc proprio 数据（**取证动作：codex `ls scores/` 确认文件名与 hash**） |
| Code lineage | 官方原始码 HEAD e3f0224（Nature 重写版）；与 arXiv v1 代际漂移已记录（研读报告 §二） |
| Task & environment | `dmc_walker_walk`，dm_control（**取证：记录 pip 实装版本**），proprio 观测 |
| Budget semantics | **unknown → 取证：读 `configs.yaml` dmc_proprio 段的 run.steps 与 env step 定义**，勿默认值补齐 |
| Training protocol | 官方 `--configs dmc_proprio` 默认全参（freeze 记录 = 展开 config + commit + seed + 时间） |
| Evaluation protocol | **unknown → 取证：configs 读 eval episodes/频率/聚合方式** |
| Repetitions | 首轮单 seed（默认）；按 kit 口径**单 seed 最多判 `promising_unresolved`**，数值复现需扩 seeds 后再议 promote |
| Acceptance envelope | 曲线形状与终值落官方参考曲线包络；不匹配时先查版本/环境/评测口径再谈算法（kit 复现裁决顺序） |
| Cost envelope | **先跑 pilot（500-1000 步）实测吞吐算 ETA 再开正式 run**（Crafter 教训 + kit 铁律） |

## 四、日报与验收话语对齐（今后采用 kit 口径）

- 裁决五档：`promote / promising_unresolved / negative / invalid_provenance / inconclusive`
- 分级表述：**"跑通"=工程成功 ≠ 复现；"趋势一致"=部分复现；"数值复现"=协议+重复+参考产物三闭环**——日报里不再写"复现成功"这种粗粒度词。
- 观察与解释分开记录（证据四联：数值/视觉/时序尾部/因果干预——消融阶段全套启用）。
