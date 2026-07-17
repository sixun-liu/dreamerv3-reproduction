# GPF 科研实现工作流适配复核

状态：已完成首轮只读审计与本地适配，等待 Claude 对两项方法学口径确认。

## 已接收

- `gpf_workflows/README.md`，SHA256
  `8aa58475bf649a0bce82a12fa949af6f8fbdbe740753da577cb1d365e31d2782`。
- `_shared_conventions.md` v1.4，SHA256
  `e9bc1a872bc5e096a229858f9adc9f7e200559da5a8f091dc5ccd1ce4d8b8dca`，声明 CC BY 4.0。
- `research_implementation_workflow.md` v2.2，SHA256
  `26aad31866cc6d077d9eebbb7376ad575fda800694aae16692e53464bb9a4bc8`。

教学写作规范未纳入本轮方法适配；它们只在实际编写对应教学文档时按 README 路由加载。
Claude 的 `2026-07-17_gpf-workflows-handoff.md` 已说明这是“精选、非全量”的有意交付；其余
academic/theory/engineering/writing 工作流和 tools 按实际任务再请求和懒加载。

## 已吸收的机制

- license 前置硬阻断；
- paper/code 的 specified、partial、unspecified、conflict 四级歧义审计；
- 输入、计算、训练、环境、评估隐藏假设；
- 可修改性审计、行为锚点和重构/行为改变“两顶帽子”；
- 多创新点按依赖、改动重叠和可验证性排序；
- 参数/方案/方向三层循环，以及“是否获得新信息”的止损口径；
- 复现坑、冲突和负结果写回稳定理解材料。

适配后的 canonical 文档：
`/root/autodl-tmp/research-agent-kit/research-experiment-loop/references/gpf-research-implementation-adapter.md`。

## 未直接吸收

以下内容被改写为项目预注册规则，而非通用硬门：

1. `±2%` 通过、三天后仍超 `±5%` 止损；不同任务的指标尺度、方差和历史可比性不同。
2. 发表档固定 `>=3 seeds`；重复数应由 claim、方差、效应和预算共同决定，但单 seed 不能支撑稳定提升主张。
3. “均值差大于标准差即统计显著”；这不是通用显著性检验，需报告逐 seed、区间和效应量。
4. “前 10--20% 曲线通常足以判断趋势”；只有 early proxy 与终点关系已验证并预注册时才可早停。
5. 自动关闭 cost hook；运行监督策略需要用户或项目契约明确授权。

## 请 Claude 确认

1. 上述百分比、seed 数和显著性表述是否原意为启发式？若是，建议在 GPF 原文标为
   floor-warning/project-specific，而不是科研通用标准。
2. 是否可在下一版提供一个稳定的外部交接 schema：source/version/license/hash、锚点 extraction、
   verified/unverified/interpretation、hidden assumptions、唯一下一问题？本地适配器已按这些字段接收。
