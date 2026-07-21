# `2411f7d` runtime 兼容性与 ratio 1024 假设复核

> 2026-07-21，Codex。面向 Claude 的协作说明；属于讨论材料，不是 canonical state，未启动 GPU 实验。

## 结论

- `2411f7d` 值得作为下一条代码代际诊断路线，使用难度目前判断为**低到中等**。
- 不要在现有 `/root/autodl-tmp/dreamerv3` 中切换 commit，也不与其共享 Git worktree/object store。
  已建立完全独立的 `/root/autodl-tmp/dreamerv3-2411f7d` clone，固定 detached commit；只有需要
  保存兼容补丁时，才在旧 runtime 上建立 `fix/jax06-sm120`。
- control repo 只有在用户批准正式 probe 后，才建立 `exp/EXP-####-runtime-2411` 并做 preregistration、
  freeze 和运行。不同 runtime 不能共用同一 freeze。

## ratio 1024 假设的证据权重

当前证据更支持参考曲线使用 ratio 512，而不是 1024：

| 时间点 | `dmc_proprio` 配置 | score 文件 |
|---|---|---|
| 2023-05 `423291a` | ratio 512，repeat 2 | 首次加入参考曲线 |
| 2024-04 `2411f7d` | size12m，ratio 512，repeat 2 | 搬迁但 walker 数值不变 |
| 2024-12 `f8817c4` | size1m，ratio 1024，repeat 1，1.1M steps | 未更新参考曲线 |

因此，现默认 1024 是 score 加入约 19 个月后随模型、repeat、预算一起出现的新配置，不能反推
2023 参考曲线使用了 1024。将 `size12m + repeat2 + ratio1024 + 500K` 组合起来是合理的训练量
敏感性 probe，但不是仓库历史中出现过的论文复现配置。

单次 ratio1024 成功只能说明“额外更新能补偿当前 runtime 的缺口”，不能证明论文 Table 2 写错或
官方曲线使用了 1024；单次失败也会受当前未闭合的 DMC seed 影响。若 Claude 找到作者日志、配置
元数据或明确回复表明 1024 早于 `f8817c4`，这会改变当前证据排序。

## 已完成的无 GPU 兼容检查

独立 clone 当前状态：HEAD `2411f7d136832378c0291c587cdbf2fca6506873`，工作树 clean，作者 remote
命名为只读 `upstream`，push URL 已禁用。当前 2026 runtime 未被修改。

在 `/tmp` 解压未修改的 `2411f7d`，使用现有 `dv3` 的 Python 3.12、JAX/JAXLIB 0.6.2：

1. 全部 Python 源码通过 `compileall`。
2. 隔离安装 `pyzmq==27.1.0`、`tensorflow-probability==0.25.0` 后，旧 `dreamerv3` 可导入。
3. 原版 logger 强制创建 TensorBoard output；当前环境无 `tensorflow-cpu`，这是首次运行失败点。
4. 仅在 `/tmp` 副本禁用 TensorBoard、保留 terminal + JSONL 后，100-step CPU dummy smoke 完成，
   产生 checkpoint 和训练 metrics。
5. 同一临时补丁下，`dmc_walker_walk` CPU debug smoke 完成 100 steps；旧 DMC wrapper 可驱动
   `dm-control==1.0.43`、MuJoCo 3.10.0，并完成 JIT、replay 和训练循环。

这些 smoke 只证明 plumbing；没有测试 GPU size12m 编译、吞吐、数值曲线或论文结果。

## 预计正式兼容工作

1. 使用独立 clone `/root/autodl-tmp/dreamerv3-2411f7d`，不改当前 runtime。
2. 锁定必要依赖；避免安装体积较大的 `tensorflow-cpu`，改为默认关闭 TensorBoard 的日志兼容补丁。
3. 兼容补丁独立提交并说明只改变输出端，不改模型、损失、replay 或环境语义。
4. 先做 GPU tiny smoke，再做 size12m 短 pilot；实测通过后才评估 500K 单 seed。
5. 第一条论文对照仍用 `size12m + repeat2 + ratio512`。ratio1024 另立 diagnostic probe，不能混入
   `2411f7d` replication。

## 当前建议

优先恢复 `2411f7d/ratio512`，因为它比在 2026 runtime 上加倍训练更能区分“2024-12 重写漂移”与
“环境/score provenance 漂移”。source-only clone 保持 detached；需要补丁时建 runtime fix 分支，
需要运行时再建 control experiment 分支。
