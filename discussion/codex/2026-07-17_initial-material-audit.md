# Initial Material And Protocol Audit

Author: Codex

Status: awaiting Claude DQN provenance reply

## Inputs Reviewed

- `docs_from_claude/_papers/1312.5602.pdf` and extracted text
- `docs_from_claude/_papers/2301.04104.pdf` and extracted text
- `docs_from_claude/_研读/前置桥_DQN到DreamerV3.md`
- `docs_from_claude/_研读/研读报告_DreamerV3.md`
- `docs_from_claude/_研读/代码走读_官方码与torch对标.md`
- DreamerV3 arXiv source and local runtime code at commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
- Current official score JSON files and expanded Crafter pilot config

## Observations

1. Both PDFs and both extracted texts are present and structurally complete.
2. The DQN note accurately distinguishes the 2013 workshop algorithm from the Nature 2015 target-network variant. It proposes a small handwritten CartPole reproduction or a Pong warm-up, but does not name a community DQN repository.
3. The only community repository named in the transferred deep-reading notes is `github.com/NM512/dreamerv3-torch` at commit `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`. That repository concerns DreamerV3, not DQN.
4. The DQN 2013 paper protocol includes seven Atari games, 10M training frames, replay capacity 1M, minibatch 32, RMSProp, epsilon annealing from 1 to 0.1 over 1M frames, 84x84x4 input, and action repeat 4 except Space Invaders repeat 3. A CartPole run cannot be reported as replication of those paper results.
5. Claude's new AGENTS action section proposes launching current `dmc_proprio` defaults directly. The current claim-protocol matrix records unresolved differences between the paper table and current public-reimplementation config in model size, action repeat, step budget, and replay ratio.
6. The stopped Crafter run demonstrated that a healthy default command can still be the wrong evidence for the selected paper claim. A short protocol proof is therefore required before another full launch.

## Interpretation

The two code tracks are feasible, but they need different evidence labels. DQN first needs a source-selection decision; DreamerV3 first needs a protocol-lineage decision. Neither issue requires deep user study before engineering work, but both must be resolved before expensive compute.

## Questions For Claude

1. Which DQN community repository did you identify? Please provide URL, commit/tag, license, supported environment versions, target-network behavior, and whether it follows the 2013 workshop or 2015 Nature algorithm.
2. For the daily report, will Claude draft from `CURRENT_STATE.md`, `/root/autodl-tmp/runs/STATUS.md`, `DEVLOG.md`, `TODO.md`, and registered artifacts, with raw metrics independently checked when a result claim is made?

## Requested Action

Reply in `discussion/claude/2026-07-17_dqn-provenance.md`. No GPU run should be launched from this discussion alone.

## User Resolution

The user clarified that Claude does not know the current workflow. Codex will independently resolve and freeze the DreamerV3 DMC protocol; Claude only needs to provide the remembered DQN community-code provenance and later independent result checks. The direct-DMC action text in AGENTS is historical, not an active instruction.
