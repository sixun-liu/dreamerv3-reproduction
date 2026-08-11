# DreamerV3 跨域 P0 审计

> 审计时间：2026-08-11 UTC
> 执行者：codex
> 性质：运行前事实审计，不是性能实验

## 结论

DMC Vision Walker 已具备 smoke 和 staged replication 条件。Atari100K 与 Minecraft 的任务配置已经从各自 runtime 展开，但当前 `dv3-2411` 环境不具备它们的依赖；两者必须等 EXP-0010 关闭后分别完成环境 smoke，不得提前并发安装或训练。Minecraft 当前授权上限为 100K environment steps。

## 仓库谱系

| 角色 | 路径 | 固定提交 | 状态 |
|---|---|---|---|
| control | `/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction` | EXP-0010 分支冻结时记录 | freeze 前须 clean 并推送 |
| DMC Vision runtime | `/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2411-crossdomain` | `6642b941f578cd72147bc2be3c3343d5bc72931c` | detached、clean |
| Atari/Minecraft runtime | `/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain` | `5168475b7a4413f9575933b4580e7073caea2114` | detached、clean |
| workflow | `/root/autodl-tmp/Tools/research-agent-kit` | `ffc2d665b477d7be0d20ec230aeb6f828552ed82` (`v0.2.1`) | clean |

`6642b94` 是作者 2024 重实现谱系上的 Blackwell/JAX 0.6 兼容提交，不是 2023 论文训练源码。EXP-0010 的复现等级因此固定为 `author_reimplementation`。

## 硬件与 DMC 环境

- Ubuntu 22.04.3 LTS；Python 3.12.13。
- NVIDIA GeForce RTX 5090，32607 MiB；driver 580.76.05；系统 CUDA toolkit 12.1。
- JAX/JAXLIB 0.6.2，NumPy 1.26.4，`dm-control` 1.0.43，MuJoCo 3.10.0。
- 数据盘 150 GB，审计时已用约 43 GB、可用约 108 GB。
- DMC 使用 `MUJOCO_GL=egl` 与 `PYOPENGL_PLATFORM=egl`，无需桌面服务。
- 训练环境 `/root/autodl-tmp/Envs/dv3-2411` 已被历史 DMC runs 验证，禁止为后续域原地升级 JAX 或 NumPy。

## 三域依赖门

| 域 | 当前事实 | 下一层动作 |
|---|---|---|
| DMC Vision | `dm_control`、MuJoCo、EGL、PyAV/Pillow 均存在 | 允许进入 EXP-0010 smoke |
| Atari100K | 当前环境缺 `ale_py`、Gym/Gymnasium 与 OpenCV | EXP-0010 结案后建立/验证 2026 runtime 环境和 ROM 链路 |
| Minecraft | 当前环境缺 Java、Xvfb、MineRL；系统也无 ffmpeg CLI（PyAV 可编码视频） | 必须使用独立环境，先完成依赖与 L0；失败可按工程阻塞结案 |

## 展开配置与参考产物

| 目标 | 配置 | SHA256 |
|---|---|---|
| EXP-0010 smoke | `configs/exp0010_dmcvision_smoke_s31415_8192_dec.yaml` | `29105b22334363c436e164f34c5c084e230720e64a7fa38b5cc438ae01184451` |
| EXP-0010 100K | `configs/exp0010_dmcvision_s000_100k_env.yaml` | `ef1068f85184f0937aa512aecf60b01e3519a34726b193edec28c88e55330588` |
| EXP-0010 1M | `configs/exp0010_dmcvision_s000_1m_env.yaml` | `ab1a663629a43e0bf197296bb34ee9301bd85926f2970ebd7c9b4b86b5e0300f` |
| Atari P0 | `configs/crossdomain_p0_atari100k_breakout.yaml` | `662e8e623e892bdeae0bd621030296ef69ae8065391f7ba8346237467b30f60b` |
| Minecraft P0 | `configs/crossdomain_p0_minecraft_reduced.yaml` | `474852d90a908df912478dcc2f3179939da42ebc0688ea970e02c8f5aa56a530` |

DMC Vision 官方参考为 runtime 内 `scores/dmc_vision-dreamerv3.json.gz`，SHA256 `68dd7e83ee6af076734e1edb843a3816cd241736c92e91837e7cf460231b7b08`。其中 `dmc_walker_walk` 有 10 条 seed 曲线；它们是参考包络，不是本地单 seed 的逐样本配对对象。

## 已知限制

1. 2024 runtime 的 DMC 配置没有 `env.dmc.use_seed`，agent seed 固定但 environment RNG 不受控。
2. 公开 score JSON 没有原始 episode 导出流水线；本地 raw episode 窗口与官方平滑曲线只能做语义接近的描述性比较。
3. JAX 显存预分配不等于活跃显存峰值；资源报告同时保留整卡采样、进程显存和 `/usr/bin/time` 数据。
4. 三个域使用不同 runtime/环境和预算语义，必须独立裁决，不能汇总为“一套代码跨域复现成功”。
