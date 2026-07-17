# DreamerV3 复现控制仓

本仓管理 DreamerV3 论文理解、实验协议、配置、分析脚本和轻量研究记录。官方训练代码、大型 run、
checkpoint、PDF 和完整 artifact 与本仓分离，通过固定 commit、路径和 hash 建立关联。

## 当前结论

`EXP-0001` 已完成 DMC `walker_walk` 500K environment steps。490K 同坐标 bin 中位数为 914.2，
进入官方五 seed 包络 735.6--955.0，但前半程学习明显偏慢，因此裁决为
`promising_unresolved`，不宣称多 seed 或整篇论文数值复现。

## 仓库角色

| 角色 | 位置 |
|---|---|
| control | 本仓，`https://github.com/sixun-liu/dreamerv3-reproduction` |
| runtime | `/root/autodl-tmp/dreamerv3`，官方 `danijar/dreamerv3@e3f0224` |
| workflow | `/root/autodl-tmp/research-agent-kit@3a5bd50`，tag `v0.1.1` |
| runs | `/root/autodl-tmp/runs/` |
| artifacts | `/root/autodl-tmp/artifacts/dreamerv3/` |
| staging | 旧 runtime 下的 `docs_from_claude/`，不进入 Git |

机器可读角色清单见 `research/repositories.yaml`。本仓是在首轮实验完成后建立的控制仓，历史 freeze
仍引用旧路径；迁移边界和不可追溯声明见 `MIGRATION.md`。

## 入口

- 当前人工综合：`CURRENT_STATE.md`
- 项目契约：`AGENTS.md`
- 结果总表：`RESULTS_SCOREBOARD.md`
- 实验 registry：`research/`
- 论文与实现索引：`references/`
- 理解与复现文档：`docs/`
- 协作讨论：`discussion/`

恢复状态时在本仓根目录运行：

```bash
researchctl status
researchctl audit --strict
researchctl hygiene
```
