# Minecraft 多环境并行与训练吞吐讨论

> 日期：2026-08-12  
> 用途：供 AutoDL 服务器 Codex 审阅 EXP-0012 的吞吐瓶颈，并设计低成本基准。  
> 约束：本文只提出审计和基准建议，不授权直接启动 300K、1M 或更长训练。

## 1. 当前问题

EXP-0012 使用真实 MineRL/Malmo Minecraft 环境完成了 `size50m/ratio32/100K` 训练：

- 墙钟：`4053 s`，约 `1.126 h`；
- 吞吐：`100000 / 4053 = 24.67 env steps/s`；
- 平均 GPU 利用率：`8.88%`，峰值 `93%`；
- 峰值显存：`24647 MiB`；
- 当前配置：`script=train`、`run.envs=1`、`run.debug=true`、
  `run.remote_envs=false`、`jax.prealloc=true`。

若把该单环境吞吐直接线性外推到 100M，需要约 47 天。但这不能代表论文的训练效率，因为论文的
Minecraft 数据采集不是单环境串行运行。

## 2. “多个环境”是什么意思

一个环境实例就是一份独立运行的 Minecraft 游戏：它拥有自己的世界、玩家位置、画面、背包、生命值
和 episode 进度。多个环境共享同一个 DreamerV3 agent，而不是训练多个模型：

```text
多个独立 Minecraft 环境（CPU/Java/Malmo）
                  |
             批量请求动作
                  v
         同一个 DreamerV3 agent（GPU）
                  |
             各自执行动作
                  v
          经验汇入同一个 Replay
                  |
             同一个 learner 更新
```

若 8 个环境各执行一步，累计得到 8 个 environment steps。论文的 100M 是所有环境产生的累计步数，
不是每个环境分别运行 100M。

单环境时，GPU 在每次短暂推理或训练后经常等待 Java/Malmo 推进。多环境使不同游戏实例并行产出
经验，并允许策略批量推理，从而减少 GPU 和 learner 的空闲时间。

## 3. 论文口径

DreamerV3 论文附录明确说明：

- 每个 agent 使用一张 Nvidia A100；
- 默认使用 16 个环境实例；
- Minecraft 环境较慢，因此使用 **64 个远程 CPU 环境 worker** 加速数据采集；
- Minecraft Diamond 使用约 100M environment steps，单 GPU 约 9 天；
- 论文默认模型规模为 200M。

论文原文的关键句为：

> For Minecraft, we use 64 environments using remote CPU workers to speed up experiments because the environment is slower to step.

本地论文文本：`Paper/2301.04104.txt`，Environment instances 段。服务器论文：
`/root/autodl-tmp/Paper/dreamerv3_2301.04104.pdf`。

论文 100M/9 天对应约 `128.6 env steps/s`，是本轮 `24.67 env steps/s` 的约 `5.2` 倍。64 个环境
不是为了得到 64 倍净加速，而是用环境并行填补 CPU 仿真、Java 通信、策略推理和 learner 之间的
等待空隙。

## 4. 为什么显存高但 GPU 利用率低

`jax.prealloc=true` 会让 JAX/XLA 预留大块显存，模型、优化器状态和编译结果也会常驻。因此
`24.6 GiB` 显存占用不等于计算单元持续繁忙。

本轮更可能是如下节奏：

```text
Minecraft/Java 推进一步 -> GPU 短时推理或训练 -> 等待下一步环境数据
```

GPU 峰值达到 `93%`，但平均值只有 `8.88%`，与“短时计算、长时等待”的环境瓶颈相符。是否确实
如此，仍需结合 timer、CPU、RAM、policy FPS、train FPS 和 replay backlog 实测，不应只凭显存判断。

## 5. 与当前重实现的边界

EXP-0012 使用作者 2026 重实现，runtime commit：
`5168475b7a4413f9575933b4580e7073caea2114`，不是论文原始训练源码。

当前重实现的普通 `script=train` 已支持多环境：`run.debug=false` 时，Driver 会为每个环境创建独立
进程，并把多个 observation 合批送入同一个 agent。该路径仍使用 `agent/replay/step` 一体 checkpoint，
并按 `run.steps` 自然结束，因此是本机第一阶段吞吐测试的最小改动路线。

`script=parallel` 会进一步拆分 actor、learner、replay、logger 和环境进程，但 checkpoint 也拆成
`agent/replay/logger` 三部分，当前代码没有基于 `run.steps` 的自然停止路径，且尚无证据证明能从
EXP-0012 的 `train` checkpoint 完整恢复 step 与 replay。因此它只能作为第二阶段的独立系统验证，
不能直接当作 EXP-0012 的续训开关。

无论使用哪条路径，本机同机多进程都不能据此声称复原论文的 64 个远程 CPU worker 部署。

## 6. 建议的低成本吞吐基准

容器 cgroup 的实际上限约为 16 个 CPU 核、90 GiB RAM、20480 PID；宿主可见的 128 核和
754 GiB RAM 不是该容器可用配额。数据盘当前约剩 86 GiB，系统盘只剩约 3.8 GiB。不要按宿主
资源规划，也不要直接启动 16 或 64 个 Minecraft/Java 实例。

在独立 logdir 中自适应测试 `2 -> 4 -> 条件性 8` 个训练环境：

- 使用 `script=train`，不覆盖或加载 EXP-0012；
- `run.debug=false`；
- 保持 `minecraft_diamond`、`size50m`、`train_ratio=32`、batch 和环境语义不变；
- EXP-0012 的 `envs=1/debug=true` 实测 `24.67 steps/s` 作为历史实用基线；只有结果含混时才补测
  `envs=1/debug=false`，用于分离进程化本身的影响；
- 每档先做多实例 reset/step L0，再做 `5040` 累计 environment steps；该预算可被 2、4、8 环境
  及 Driver 推进量子整除；
- 先测 2 环境，通过完整性和资源门后测 4；8 环境仅在 4 环境吞吐仍增长且资源健康时放行；
- 若出现 CPU throttling、RAM 饱和、Java 崩溃、replay 堵塞或吞吐回落，立即停止扩档；不测 16。

每个 run 创建数据盘内的 `work/tmp`、`work/malmo` 和 `work/jax-cache`，显式设置 `TMPDIR/TMP/TEMP`、
`MALMO_MINECRAFT_OUTPUT_LOGDIR` 和 `JAX_COMPILATION_CACHE_DIR`。MineRL 会用 Python `tempfile`
复制每个 Minecraft 实例；未设置时会落入系统盘 `/tmp`，多环境会放大系统盘风险。

每档至少记录：

1. 总墙钟、启动和 JIT 时间、稳态 environment FPS；
2. policy FPS、train FPS、实际 replay ratio 和 backlog；
3. cgroup CPU 使用与 throttling、cgroup memory、Java/Malmo 实例数及各实例 RSS；
4. GPU 利用率、功耗、显存和 OOM/非有限值；
5. replay、checkpoint、步数和环境终止语义是否完整；
6. 环境错误、进程退出和日志中的 timer 分解。

选择“仍稳定且接近峰值吞吐的最小环境数”，而不是盲目选择最大环境数。若 2 环境相对历史基线
没有明确增益，先停止并检查进程化开销；若 4 环境相对历史基线仍小于约 `1.5x`，先定位同步 Driver、
Java、replay 或 CPU 配额瓶颈，不追加长跑。短探针只支持运行方案裁决，不支持论文性能主张。

## 7. 基准后的训练决策

只有吞吐基准稳定通过，且另行确认所选模式能正确恢复现有 checkpoint、replay 与步数后，才根据
实测吞吐、磁盘与科学价值选择增量训练预算。不得把独立基准运行混入正式训练结果，也不预先承诺
必须续训到 300K、1M 或更高预算。

可参考的剩余 200K 纯步数时间：

| 实测总吞吐 | 理想步数耗时 |
|---:|---:|
| `25 steps/s` | 约 2.22 h |
| `80 steps/s` | 约 41.7 min |
| `120 steps/s` | 约 27.8 min |
| `129 steps/s` | 约 25.8 min |

实际墙钟还包含启动、JIT、保存和评测。达到论文约 `129 steps/s` 只是吞吐参照，不等于复现论文
性能；本轮是 50M 模型，论文默认是 200M 模型，代码谱系和 CPU 部署也不同。

## 8. 请服务器 Codex 返回的内容

1. 已核实：容器实际约 16 核、90 GiB RAM；首轮现实档位为 2/4，8 条件放行；
2. 已核实：首轮使用 `script=train`；`parallel` 的停止与跨模式恢复需另开验证；
3. 先完成数据盘重定向、多实例 L0 和 `2/4/(8)` 的 5040-step 自适应探针；
4. 每档给出端到端与稳态吞吐、磁盘、CPU/RAM、Java 和 GPU 资源账；
5. 依据探针结果再冻结恢复验证和增量训练预算，不自动启动更长正式训练。
