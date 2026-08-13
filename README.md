# DreamerV3 复现与跨域验证

[![Release checks](https://github.com/sixun-liu/dreamerv3-reproduction/actions/workflows/release-checks.yml/badge.svg)](https://github.com/sixun-liu/dreamerv3-reproduction/actions/workflows/release-checks.yml)

本仓保存 DreamerV3 的实验协议、冻结配置、启动与验收脚本、分析代码和研究记录。官方训练 runtime、
大型 checkpoint、Replay 和视频产物与 Git 分离，通过固定 commit、配置和哈希建立关联。

当前已经形成三条可复跑的代表路径：DMC Vision Walker、Atari100K Breakout 和 Minecraft。它们覆盖
连续控制、离散视觉决策和长时程开放世界任务，同时保留成功现象、失败案例与结论边界。

## 快速开始

```bash
./scripts/quickstart.sh dmc-vision check
./scripts/quickstart.sh dmc-vision smoke

./scripts/quickstart.sh breakout l0
./scripts/quickstart.sh breakout smoke

./scripts/quickstart.sh minecraft l0
./scripts/quickstart.sh minecraft smoke
```

启动器默认只运行短程 smoke，适合先确认环境、训练和保存链路。正式预算必须显式选择，并会在独立
输出目录中保存实际命令、配置、checkpoint、Replay、日志和 `integrity.json`。`integrity.json`
通过只表示产物结构和数值完整，不等于达到论文分数。环境安装、路径覆盖和正式命令见
[Quickstart](docs/QUICKSTART.md)。

## 已完成范围

| 路径 | 本地范围 | 结论边界 |
|---|---|---|
| DMC Vision Walker | 1M environment steps，单 seed | 形成像素控制数值对齐实例，不代表跨 seed 全域复现 |
| Atari100K Breakout | 100K decisions，独立 10 局评测 | 降规模作者重实现出现学习趋势，不是论文 200M 配置严格复现 |
| Minecraft Diamond | 100K→200K→500K，固定三局评测 | 500K 固定评测中 2/3 局达到木镐与圆石；训练 Replay 仅有少量炼铁前置信号 |

实验详细数字、证据和限制以 [结果总表](RESULTS_SCOREBOARD.md) 与各实验协议为准。

## 仓库角色

| 角色 | 位置 |
|---|---|
| control | 本仓，`https://github.com/sixun-liu/dreamerv3-reproduction` |
| runtime | Git 外的固定 `danijar/dreamerv3` checkout；跨域实验使用 2024 与 2026 两个谱系 |
| workflow | `/root/autodl-tmp/Tools/research-agent-kit@ffc2d66`，tag `v0.2.1` |
| runs | `/root/autodl-tmp/Runs/` |
| artifacts | `/root/autodl-tmp/Artifacts/dreamerv3/` |
| staging | 旧 runtime 下的 `docs_from_claude/`，不进入 Git |

机器可读角色清单见 `research/repositories.yaml`。本仓是在首轮实验完成后建立的控制仓，历史 freeze
仍引用旧路径；迁移边界和不可追溯声明见 `MIGRATION.md`。

## 入口

- 当前人工综合：`CURRENT_STATE.md`
- 快速复跑：[docs/QUICKSTART.md](docs/QUICKSTART.md)
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
