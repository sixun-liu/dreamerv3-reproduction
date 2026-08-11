# EXP-0007 共同数据面板 open-loop prediction 协议

> 类型：`probe`，只产生 diagnostic/debug-only 证据，不进入正式复现 scoreboard。

## 问题

`EXP-0006` 的 P4 在两个训练 seed 中同时表现为极低 raw KL、较差重构和较差控制分数。固定数据
支持后，这种变化是否对应世界模型的实际多步预测能力下降，还是主要来自各策略访问了不同状态？

## 预注册预测

1. 主指标是共同 panel 上 horizon 15 的 prior proprio NRMSE。P4 相对 E1 若两个配对训练 seed
   均更差，则支持“低 KL 并不等于可用预测更好”。
2. P4 相对 E1 的 posterior-prior KL 预计仍更低。若 KL 更低而 prior NRMSE 更高，说明 KL
   不能脱离表征容量和预测误差单独解释。
3. teacher-forced 与 prior 必须分开：两者都变差更符合表征/decoder 信息下降；只有 prior 相对
   teacher 的额外差距扩大，才更特异地支持 imagination drift。
4. E1 相对 baseline 的方向开放，因为 `EXP-0006` 中 E1 raw KL 的两个 seed 方向不一致。

## 唯一比较与控制

- 比较变量：冻结 checkpoint 的训练 arm，`baseline`、`E1`、`P4-reconstructed`，各 seeds 0/1。
- 固定项：runtime、模型结构、共同 panel、16-step posterior warm-up、真实 replay actions、
  horizons `1/3/5/10/15`、evaluation RNG、每窗 4 个 latent draw 和指标实现。
- 每个 checkpoint 评估同一 panel；panel 从六个 replay 来源各抽 96 个窗口，共 576 个。
- 每窗 31 个 observation：前 16 个只用于从零 latent state warm-up，随后预测 15 个。
- 窗口内不得出现 `is_first/is_last/is_terminal`；同一 chunk 内入选窗口不得重叠。
- replay `action[t]` 对齐 `observation/reward[t+1]`。
- 严禁把 replay 缓存的 `dyn/deter` 或 `dyn/stoch` 输入其他 checkpoint。

## 指标

- proprio：orientations、height、velocity 的 raw RMSE/MAE、decoder symlog-space loss，以及按共同
  panel target 标准差逐维归一化的 NRMSE。
- 主 proprio NRMSE 按 24 个 raw observation 维度等权；归一化尺度只从共同 panel target 计算。
- reward：raw RMSE/MAE 与 two-hot loss。
- latent：每个未来步的 `KL(posterior || prior)`、prior entropy、posterior entropy。
- 时序/尾部：median、P95、max 与 worst-window index；按 replay 来源 arm 分层。
- 统计单位是 replay window。4 个 latent draw 先在同一 window 内平均，再跨窗口汇总；两个训练
  seed 只画 min--max，不称置信区间。

## 完整性与停止条件

- panel 必须通过来源 slice、chunk SHA256、等量来源、无边界和禁用 latent key 校验。
- 六个 checkpoint 必须是 `EXP-0006` exact step-250000 terminal checkpoint，hash 与原记录一致。
- 六次推理必须使用同一 clean runtime commit，输出全部有限且 row count 完整。
- 任一 checkpoint/config/panel hash 不匹配、NaN/Inf、OOM、重复 GPU 进程或数据盘余量低于 10 GiB
  时停止，不静默重试或替换窗口、seed、draw。
- 不按中途结果改变 panel、context、horizon、归一化、draw 数、主指标或 checkpoint。

## 结论权限

该 probe 可区分共同数据支持上的重构误差与 prior rollout drift，但不能证明闭环控制因果、跨任务
稳健性、论文 Figure 6/17 数值复现或总体效应大小。若发现稳定机制信号，后续正式实验需另开卡。
