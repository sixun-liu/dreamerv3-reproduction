# EXP-0009 Reacher Hard 学习信号三臂 pilot

> 冻结对象：arXiv:2301.04104v2 Figure 6b / Supplementary Figure 18 的定性机制顺序。
> 本轮是 1M environment-step、单 seed 的 conceptual pilot，不是 10M 正式复现。

## 问题与梯度语义

三臂只改变 replay representation 收到的梯度：

| arm | reconstruction | reward | replay value | continuation | dynamics / KL |
|---|---:|---:|---:|---:|---:|
| `baseline` | on | on | on | on | on |
| `no_reward_value` | on | stopped | stopped | on | on |
| `no_reconstruction` | stopped | on | on | on | on |

`stopped` 的含义是对 head 输入 latent tree 使用 `jax.lax.stop_gradient`。对应 decoder、reward head
或 critic 自身继续训练；把 loss scale 设为零不满足这一语义。默认 `ac_grads=none` 保持不变，
因此 imagined actor/critic 不通过 latent 反向塑形世界模型。

## 运行协议

- Runtime：作者 2024 重实现 upstream `2411f7d`，加 JAX 0.6 compatibility commit `6642b94`
  和 EXP-0009 梯度路由 commit `990123a`。
- Task：`dmc_reacher_hard`，proprio observation，`size12m`。
- Budget：每臂 500,000 agent decisions = 1,000,000 DMC environment steps，action repeat 2。
- Compute：16 envs，replay ratio 512；三臂串行，顺序预先固定为 baseline、no_reward_value、
  no_reconstruction。
- Repetition：agent seed 0；DMC 每个 env 由 `(seed, env_index)` 配对派生 seed，三臂一致。
- 禁止项：不因中间分数、曲线形状或某臂看起来明显而调参、换 seed、改预算或提前停健康 run。

论文 Figure 18 的 Reacher Hard 曲线使用 10M environment steps；本轮只有其十分之一预算。
因此主问题是早期方向是否出现，不用本轮终值对齐论文约 950 的曲线。

## 梯度 gate

先过两级验证：

1. 代数测试：route 开启时 representation/head 梯度都非零；关闭时 representation 精确为零、
   head 仍非零。
2. 真实 Dreamer 小模型 Reacher batch：对 reconstruction、reward、replay critic 分别报告
   `enc/dyn/dec/rew/critic` 参数组 grad norm，并按三臂逐项判定。

任一应为零的梯度非零、任一对应 head 梯度为零、配置或 checkpoint 不完整，都禁止启动 pilot。

## 指标与裁决

- Primary：raw episode return 按 episode 结束时的 environment step 放入固定 100K bins，计算
  0--1M 的梯形 AUC。
- Directional gate：`baseline` 和 `no_reward_value` 的 AUC 都高于 `no_reconstruction`。
- Secondary：`baseline` 与 `no_reward_value` 的顺序；1M 内未分开不判论文负例。
- Tail：末 200K environment steps 的 episode return 均值、每箱 episode 数和最差箱。
- Integrity：精确 terminal checkpoint、loss/score 有限、无 traceback/OOM、资源和磁盘记录完整。

单 seed 顺序成立只能支持“该任务该预算下出现与论文一致的机制方向”；不成立则优先区分预算不足、
seed 方差、2023/2024 代码谱系差异和消融实现问题。
