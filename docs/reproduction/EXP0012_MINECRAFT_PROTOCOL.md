# EXP-0012 Minecraft Reduced 协议

> 状态：依赖兼容门通过，等待 L0；本轮最高预算 100K environment steps。

## 研究问题

在作者 2026 DreamerV3 重实现 `5168475` 上，真实 MineRL/Malmo Minecraft 链能否在本机稳定
完成环境 L0、size50m 模型 smoke，并在不超过 100K environment steps 的单 seed 预算内产生可复核的
低阶里程碑证据？

## 谱系和边界

- 复现等级：`author_reimplementation`，不是论文原始训练源码。
- 论文：arXiv `2301.04104` / Nature 2025，本地 PDF
  `/root/autodl-tmp/Paper/dreamerv3_2301.04104.pdf`。
- runtime：`/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain`，commit
  `5168475b7a4413f9575933b4580e7073caea2114`。
- 参考曲线：`scores/minecraft_diamond-dreamerv3.json.gz`，SHA256
  `8580751af2ed04479056f0159c404747efe42c20e81464c4a04205b94296f09a`，只作论文量级背景。
- 论文 diamond 结果使用远高于本轮的模型与约 100M 级预算；本轮 size50m/100K 只能回答工程可行性
  与极早期里程碑，不能声称数值复现钻石结果。
- wrapper 的 `DefaultWorldGenerator(force_reset=True)` 未暴露固定 world seed；因此本轮只能固定 agent
  seed，不能把不同回合称为固定世界配对评测。

## 依赖冻结

- Python `3.11.15`，隔离环境 `/root/autodl-tmp/Envs/dv3-minecraft-2026`。
- OpenJDK `1.8.0_492`，Xvfb `21.1.4`，系统 GL/GLEW 依赖来自 Ubuntu 22.04。
- 作者补丁 wheel：`minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl`，SHA256
  `b04e2cd21627e835d5b1566db1fe69ed1bff8abed1c804340ebd61c78a1dc631`。
- JAX/JAXLIB `0.6.2`、NumPy `1.26.4`、OpenCV `4.11.0.86`。作者旧 requirements 中
  `jax==0.4.33` 与 `nvidia-cuda-nvcc-cu12<=12.2` 不支持本机 Blackwell，替换为已在同卡验证的
  JAX 0.6.2 CUDA 依赖；这是运行兼容性漂移，不是算法改动。
- MineRL wheel 元数据只要求 `numpy>=1.16.2`；runtime 中 `<1.24` 的注释指向 MineRL v1.0，
  但项目仍按 `numpy<2` 固定，避免 Gym 0.23/旧 NumPy API 漂移。

## 证据阶梯

1. L0：单环境 reset/step，64x64 RGB 非空且动态，动作空间、reward、inventory、32 步测试时限、
   post-terminal reset 和 MP4 全部通过。32 步时限只用于测试，不进入训练。
2. 模型 smoke：`minecraft + size50m`、单环境、seed 31415、4096 steps、ratio 32；检查 replay、
   checkpoint、有限损失、显存/吞吐/磁盘和自然退出。
3. 正式 early milestone：保持 task/model/ratio/wrapper 不变，seed 0、100K environment steps。
4. 独立评测与材料：固定 agent seed，逐回合报告回报和里程碑；明确 world seed 不可控；保留曲线、
   首个预注册回合视频和资源账。

## 里程碑

- L1：`log`、`planks`、`crafting_table`。
- L2：`wooden_pickaxe`、`cobblestone`。
- L3：`iron_ore` 或 `iron_pickaxe`。
- L4：`diamond`，只作远期目标。

训练奖励直接沿用 wrapper 的一次性物品奖励与 health reward，不修改 reward、动作、break speed 或
任务逻辑。主指标为各里程碑成功回合数/首次出现步数及回报分布；tail 包括 NaN/Inf、环境错误、
checkpoint/replay 完整性、吞吐、资源峰值和视频动态性。

## 放行与停止

- L0 失败：保留日志并停在依赖/环境层，不启动模型。
- size50m smoke OOM 或资源不可接受：先记录；只有预注册 replacement freeze 后才按
  `size25m -> size12m` 降档。
- 任何非有限值、环境语义错误、重复 GPU 进程、checkpoint/replay 不一致、预计剩余磁盘低于 5 GiB
  或预计墙钟超过 12 小时均停止当前层。
- 100K 是硬上限；不因未出现正结果自动追加 1M/5M/100M，也不按中途分数换 seed 或调参。

## 执行期 instrumentation 修正（2026-08-12）

4096-step smoke 的训练本体正常退出并在 step 4100 保存完整 checkpoint，但初版 verifier 要求
`checkpoint_step == 4096`，因此留下了 integrity-only failure。冻结 runtime 的训练循环每次调用
driver 固定推进 10 environment steps，故请求预算的自然终点为
`ceil(requested_steps / 10) * 10`：4096 对应 4100，而 formal 的 100000 仍精确对应 100000。

该修正仅改变校验器对停止粒度的建模，不改变训练代码、配置、seed、模型、奖励、动作空间、
资源门或预算上限。原始 `.failed` 与 `integrity_smoke.json` 永久保留；修正后使用带新 control
commit 和输入哈希的 `smoke_reconciliation.json` 生成独立 reconciled integrity 与 gate，禁止重跑
smoke 来覆盖该信号。
