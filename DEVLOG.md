# DEVLOG

只追加持久决策、路线升降级和机制结论；原始运行输出留在 registry/artifact。

## 2026-07-16

- 将项目置于 `understanding`，先完成论文版本、代码谱系和 claim-protocol 对账，再冻结 baseline。
- `crafter_baseline_0716` 在健康运行至日志 step 16200 后经用户批准停止；step 12330 checkpoint 可恢复。该 run 作为 pre-workflow pilot，不形成论文复现结论。
- Crafter 在论文中用于 scaling 分析，但当前仓库没有 Crafter 官方 JSON 曲线；取消“单个 Crafter 长跑即可验收论文复现”的旧口径。
- 首个候选转为有逐 seed 官方曲线的 DMC proprio `walker_walk`，但论文与当前代码协议差异尚未闭环，暂不启动。

## 2026-07-17

- 固定 Nature-era DMC proprio `walker_walk` 为首个 replication：12M、repeat 2、16 env、ratio 512、
  250K agent decisions = 500K environment steps，seed 0。
- 官方参考 `scores/dmc_proprio-dreamerv3.json.gz` SHA256 为
  `8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4`；五条 seed 曲线只用于
  冻结后的同坐标比较。
- `EXP-0001` 在 2026-07-17 01:15 CST 启动，10K environment steps 是继续/停止 pilot gate。
- DQN 目标固定为 2013 arXiv Breakout 独立重实现；CleanRL `fe8d8a0` 是 2015 Nature-style
  第三方参照，不作为 2013 数值真值。
- `EXP-0001` 自然完成 500K environment-step 目标：本地 490K bin 中位 914.2，进入官方五 seed
  范围 735.6--955.0；但 250K 中位 476.1 显著低于官方范围，约 400K 后才追入包络。裁决为
  `promising_unresolved`，不从单 seed 归因早期差异，也不晋升数值复现。
- 12 小时联合窗口中的 DQN `EXP-0001` 随后自然完成 10M emulator frames：评估峰值 10.90、
  最终 2.21，高于随机参照但远低于论文平均 168。两项任务均已释放 GPU，后续不自动启动。
