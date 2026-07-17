# 控制仓迁移说明

迁移时间：2026-07-17 UTC。

本仓首次 Git 提交晚于 `EXP-0001` 的运行与结案。首次提交是对既有控制材料的导入快照，不是该实验
运行前已经存在的源码 commit，也不改变历史 evidence authority。

## 导入范围

- 根控制文档：`AGENTS.md`、`CURRENT_STATE.md`、`DEVLOG.md`、`PLAN.md`、`TODO.md`、
  `RESULTS_SCOREBOARD.md`；
- `discussion/`、`docs/`、`references/`、`research/` 和 `scripts/` 中的稳定材料；
- 排除 notebook/Python cache、PDF、官方 runtime 源码、`docs_from_claude/` staging、run 和 artifact。

## 历史 provenance

- `EXP-0001` freeze 使用 runtime `/root/autodl-tmp/dreamerv3`；
- runtime commit 为 `e3f02248693a79dc8b0ebd62c93683888ddaccfe`；
- 展开配置原路径为
  `/root/autodl-tmp/dreamerv3/docs/reproduction/configs/dv3_dmcp_walker_500k_s0_0717.yaml`；
- freeze 中已经记录配置 SHA256、输出路径、seed policy 和运行命令。

旧 runtime 中的控制材料暂不删除，以保证上述绝对路径仍可解析。新控制仓中的副本从首次提交起作为
后续工作的 canonical control state；不得用新提交反向声称它是 `EXP-0001` 的运行时 commit。

未来实验应分别冻结 control、runtime 和 workflow commit。若迁移旧绝对路径，必须使用正式 path
alias/relocation event，并在迁移前后核验 hash。

