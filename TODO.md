# TODO

仅保留近期可执行项；实验事实和完成历史不堆在这里。

## Now

- [x] 核验 CleanRL commit/license，并分开 2013 与 2015 DQN 谱系。
- [x] 冻结 Nature-era DMC proprio `walker_walk` 论文协议、官方 JSON 和 expanded config。
- [x] 观察 `EXP-0001` 到 10K environment steps，记录吞吐、显存、checkpoint、磁盘和 ETA。
- [x] 根据 pilot gate 完成 Dreamer 后切换 DQN，全程无两个 GPU run 并发。
- [x] 生成同坐标曲线、记录 observation/verdict，并更新日报事实路径。
- [ ] 用户人工复核 DreamerV3 与 DQN 最终曲线。
- [ ] 扩 seed 或消融前完成官方 DMC JSON 生成协议取证。

## Waiting

- 后续 GPU 实验等待用户看图和协议取证，不自动续跑。
