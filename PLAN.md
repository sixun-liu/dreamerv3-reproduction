# PLAN

> Updated: 2026-08-13T06:42:41Z
> Maintainer: codex
> Source of truth: research/project_state.yaml

- Stage: `writing`
- 北极星：将已经完成的 DreamerV3 跨域复现整理成可审查、可复跑、结论边界清楚的公开仓库，并把
  实验理解沉淀为后续 PPT 与教学材料。
- 当前问题：历史实验入口较分散，外部使用者难以判断应先检查什么、运行什么以及怎样确认结构性跑通。

## 发布门

- [x] 冻结 DMC Vision、Breakout 和 Minecraft 三条代表路径的实验范围与结论边界。
- [x] 提供统一 `check`、`l0`、`smoke`、`pilot/formal` 启动入口，默认不触发长实验。
- [x] 每次运行隔离保存实际命令、配置、日志、checkpoint、Replay 和完整性报告。
- [x] 提供 runtime commit、依赖、ROM、GPU 和磁盘的运行前检查。
- [x] GitHub Actions 通过 shell 语法、dry-run、Ruff 和 quickstart verifier 单测。
- [ ] 发布分支推送并审查后，再决定是否快进 `main`。
- [ ] 公开复用前确定仓库许可证。

## 当前路线

1. 完成 Quickstart、自动验收和轻量 CI 的发布前验证。
2. 核对 README、结果总表和三个实验协议之间的数字与口径。
3. 推送独立发布分支供审查；确认许可证后再更新默认分支。
4. 以 DQN 到 DreamerV3 的知识桥和三域实验为主线准备后续讲解材料。

## 暂停路线

- Minecraft 1M、5M 与论文尺度 Diamond 预算。
- `size50m/size100m` 学习效率对照及 `size200m/400m` 探针。
- 三个域的多 seed、full-suite 和更多消融。
- 仅为继续占满 GPU 而做的环境数、NUMA 或编译缓存微调。
