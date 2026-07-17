# Implementation Reference Index

## Verified

### DreamerV3 Author Public Reimplementation

- Repository: `https://github.com/danijar/dreamerv3`
- Local path: `/root/autodl-tmp/dreamerv3`
- Commit: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
- License: Apache-2.0 (`LICENSE` in the repository)
- Source class: `author`
- Caveat: repository README describes this as a reimplementation based on DreamerV2 and unrelated to Google or DeepMind; paper-version drift must be stated.

### CleanRL DQN（已核验的第三方参照）

- Repository: `https://github.com/vwxyzjn/cleanrl`
- Local path: `/root/autodl-tmp/third_party/cleanrl`
- Commit: `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`
- License: MIT（仓库 `LICENSE`）
- Source file: `cleanrl/dqn_atari.py`
- Source SHA256: `84ec363765bf3493761186eb1c7ea7ae7dcadaebed3ddddebdf4479bd2dd34f2`
- Source class: `third_party`
- Lineage: 文档明确以 Mnih et al. 2015 Nature DQN 为目标，不是 2013 arXiv DQN。
- 已知漂移：三层卷积/512 隐层、target network、Adam、life-loss terminal、max-and-skip、
  直接 resize 84x84、训练回报代替独立评估；默认 10M agent steps 等于 40M emulator frames。

本项目只将 CleanRL 用作环境封装、日志和工程结构参照。2013 DQN 结果必须标成独立重实现，不能
把 CleanRL 的 Nature-style 曲线当成 2013 论文复现。

## Pending Verification

- Google Dopamine, Stable-Baselines3/RL Zoo, Tianshou, and PFRL remain comparison leads.
- `NM512/dreamerv3-torch` is a third-party DreamerV3 implementation and is not a DQN source.

Pending entries are `lead` class and cannot define a canonical baseline.
