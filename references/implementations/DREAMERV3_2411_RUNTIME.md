# DreamerV3 `2411f7d` runtime

## 谱系

- 作者仓上游：`https://github.com/danijar/dreamerv3.git`
- 冻结上游 commit：`2411f7d136832378c0291c587cdbf2fca6506873`
- 上游 tree：`eff48d403e5141b1f903d6feb54da1aa83ca18c1`
- 本地兼容 commit：`6642b941f578cd72147bc2be3c3343d5bc72931c`
- 兼容 tree：`1f916ed7c39e1e0df30487ecfcb75457ad20fd06`
- 可恢复 patch：`patches/dreamerv3-2411/0001-fix-runtime-support-jax06-compatibility-runs.patch`
- Patch SHA256：`6e4d534ec768cfae96e4cb5d7584cad4c51cf26e3728acc3185c8f91904bf8e4`

兼容提交只改变日志和 checkpoint 仪器：TensorBoard 增加开关，旧默认仍为开启；终点
checkpoint 增加默认关闭的 `run.save_at_end`，显式开启时等待异步写入完成。它不修改模型、
损失、优化器、replay、环境或训练更新。

## 环境

- 隔离前缀：`/root/autodl-tmp/envs/dv3-2411`
- Python `3.12.13`；JAX/JAXlib `0.6.2`
- dm-control `1.0.43`；MuJoCo `3.10.0`
- pyzmq `27.1.0`；tensorflow-probability `0.25.0`
- GPU：RTX 5090；正式运行关闭 TensorBoard，保留 terminal、metrics JSONL 和 scores JSONL。

`--seed 0` 控制 agent 随机性，但该版本的 DMC 构造路径没有把 seed 传给环境。因此一次成功
运行可支持旧代码谱系具有论文级表现的可能性，一次失败不能单独证明该谱系不可复现。

## 恢复

从干净上游 commit 执行：

```bash
git am patches/dreamerv3-2411/0001-fix-runtime-support-jax06-compatibility-runs.patch
git rev-parse HEAD^{tree}
```

恢复后的 tree 必须为 `1f916ed7c39e1e0df30487ecfcb75457ad20fd06`。
