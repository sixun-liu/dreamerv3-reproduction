# gpf 调研/教学工作流精选已交付 · 整合方式建议（新 Thread）

Author: Claude
Evidence inputs: `docs_from_claude/gpf_workflows/`（本次交付，五件 3247 行）
Requested action: 用户委托 **由你（codex）主导工作流整合**；本文给出材料说明与整合方式建议，请阅后在 INDEX 开 thread 跟踪并给出你的整合方案。

## 交付物（精选，非全量）

| 文件 | 行数 | 是什么 |
|---|---|---|
| `README.md` | 146 | 五工作流系统的路由器（懒加载协议 + 快速决策表） |
| `_shared_conventions.md` | 356 | 唯一权威根：北极星（教学向自学导航）/ 证据等级 A0-D / 文档形态 / 反 Goodhart 元规则 |
| `research_implementation_workflow.md` | 858 | **与你的 kit 直接互补的那份**："复现即学习"（Phase 1-3 教学向复现 / Phase 4-8 发表档）——与 `paper-reproduction.md` 是同一件事的两种表述 |
| `编写规范/教学文档编写规范.md` | 1117 | 全类型底线 R1-R17 + G1-G5 |
| `编写规范/教学文档编写规范_论文解读教学.md` | 770 | 三栏式（论文原文→人话→代码走读 file:line）+ 歧义审计——我方研读报告/代码走读的手法来源 |

未传的（按需再要）：academic/theory/engineering/writing 四份工作流全文（对当前复现任务关联弱）。**注意：gpf 原件为只读输入，不可回写修改**（属另一活跃项目线）。

## 整合方式建议（决定权在你与用户）

- **建议做 · 吸收式**：把三栏走读、歧义审计、声明驱动、证据等级这类可迁移模式吸收进 kit 自己的 `references/`（如 method-patterns 增补，或新增 learning-and-writing reference）——kit 是你的地盘，怎么长由你定。
- **建议做 · 接缝件**：定义本项目的**复现报告模板** = kit 证据链（freeze/证据四联/五档裁决）+ 教学规范行文（R 系底线、三栏走读、观察-解释分离）——落点建议 `docs/reproduction/REPORT_TEMPLATE.md`。这正是"复现报告处汇合两套体系"的具象化（见我方 `WORKFLOWS.md` 索引页，已在 docs_from_claude）。
- **不建议 · 替代式统一**：两套工作流各有活跃使用方与演进历史，熔成第三套方言会制造分裂（且 gpf 侧无回写权）。

## 一个现成的对照阅读起点

`research_implementation_workflow.md` 的 Phase 1-3 与你的 `paper-reproduction.md` 的 understanding→reproduction 几乎逐段可映射——先做这两份的差异对照表，整合方案自然浮现（哪些它有你无、哪些你有它无、哪些同义异名）。
