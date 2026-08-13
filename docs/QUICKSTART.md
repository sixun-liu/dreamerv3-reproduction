# Quickstart

This repository is the lightweight control and verification layer used for the DreamerV3 reproductions. The author runtime, Python environments, Atari ROM, Minecraft dependencies, checkpoints, and large outputs stay outside Git.

The quickstart entry point reproduces three representative paths:

| Task | Runtime lineage | Default profile | Validated formal budget |
|---|---|---|---:|
| DMC Vision Walker | 2024 author reimplementation | `size12m`, ratio 512 | 1M environment steps |
| Atari100K Breakout | 2026 author reimplementation | `size50m`, ratio 256 | 100K decisions |
| Minecraft Diamond | 2026 author reimplementation | `size50m`, ratio 32 | 100K steps; later continued to 500K |

These paths validate engineering reproducibility and bounded single-seed results. They do not reproduce every table, task, model size, or seed in the paper.

## 1. Prepare the runtime checkouts

Use Linux, Python 3.11+, a CUDA-capable JAX installation, and sufficient GPU memory. The validated 50M profiles reached roughly 25 GB peak VRAM, so a GPU with at least 30 GB usable memory is the conservative starting point.

```bash
git clone https://github.com/sixun-liu/dreamerv3-reproduction.git
cd dreamerv3-reproduction

git clone https://github.com/danijar/dreamerv3.git ../dreamerv3-runtime-2411-crossdomain
git -C ../dreamerv3-runtime-2411-crossdomain checkout 6642b941f578cd72147bc2be3c3343d5bc72931c

git clone https://github.com/danijar/dreamerv3.git ../dreamerv3-runtime-2026-crossdomain
git -C ../dreamerv3-runtime-2026-crossdomain checkout 5168475b7a4413f9575933b4580e7073caea2114
```

Install JAX for the local CUDA driver first, then install the dependencies from the corresponding runtime. Do not blindly replace a working JAX build with an older pinned wheel: the Blackwell test machine required JAX 0.6.2 rather than the runtime's older CUDA pin.

Set the Python executables if they differ from the validated AutoDL layout:

```bash
export DREAMERV3_PYTHON_DMC=/path/to/dmc-env/bin/python
export DREAMERV3_PYTHON_ATARI=/path/to/atari-env/bin/python
export DREAMERV3_PYTHON_MINECRAFT=/path/to/minecraft-env/bin/python
```

Atari additionally requires a legally obtained `breakout.bin`. ROM files are not distributed by this repository:

```bash
export DREAMERV3_ROM_DIR=/path/to/atari-roms
```

Minecraft additionally requires Java 8, Xvfb, GL/GLEW system libraries, and a compatible MineRL/Malmo package. The validated environment used Python 3.11, OpenJDK 8, Xvfb, NumPy below 2, JAX 0.6.2, and the author's `minerl_mirror 0.4.4` wheel. See [EXP0012_MINECRAFT_PROTOCOL.md](reproduction/EXP0012_MINECRAFT_PROTOCOL.md) for the exact dependency boundary.

## 2. Check before running

The check validates the runtime commit, a clean runtime tree, Python imports, task-specific dependencies, GPU availability, and free disk space.

```bash
./scripts/quickstart.sh dmc-vision check
./scripts/quickstart.sh breakout check
./scripts/quickstart.sh minecraft check
```

Use `--dry-run` to inspect a resolved command without requiring the target machine to have the environment installed:

```bash
./scripts/quickstart.sh minecraft smoke --dry-run
```

## 3. Run a representative reproduction

Smoke profiles are the default and are intended to verify the complete training path, not policy quality:

```bash
./scripts/quickstart.sh dmc-vision smoke
./scripts/quickstart.sh breakout l0
./scripts/quickstart.sh breakout smoke
./scripts/quickstart.sh minecraft l0
./scripts/quickstart.sh minecraft smoke
```

Formal profiles must be selected explicitly:

```bash
./scripts/quickstart.sh dmc-vision pilot   # 100K environment steps
./scripts/quickstart.sh dmc-vision formal  # 1M environment steps
./scripts/quickstart.sh breakout formal    # 100K decisions, repeat 4
./scripts/quickstart.sh minecraft formal   # 100K environment steps
```

The public Minecraft `formal` profile intentionally starts a bounded 100K run from scratch. The reported 500K result was produced by protocol-controlled continuation from the 200K checkpoint with its optimizer, Replay, and counters; it was not a single fresh quickstart run. Use the historical EXP-0020 and EXP-0023 protocols when that lifecycle, rather than the portable entry point, is the object being reproduced.

The launcher runs in the foreground so a lost SSH connection is visible instead of being mistaken for a finished job. Use `tmux`, `screen`, or the cluster scheduler for long profiles; do not blindly rerun a command after a disconnect because a duplicate process may still own the GPU.

`dmc-vision formal` starts the final profile from scratch for portability. The original EXP-0010 used a staged 100K gate and then continued in the same log directory. Use [EXP0010_DMC_VISION_PROTOCOL.md](reproduction/EXP0010_DMC_VISION_PROTOCOL.md) and the historical runner when exact lifecycle reproduction is required.

On the validated AutoDL layout, each invocation creates a timestamped directory under `/root/autodl-tmp/Runs`. Other machines fall back to the Git-ignored `outputs/` directory. Override either default with:

```bash
export DREAMERV3_OUTPUT_ROOT=/data/dreamerv3-runs
```

## 4. Inspect the output

Each training run contains:

```text
quickstart-<task>-<stage>-<timestamp>/
  manifest.txt
  train_stdout.log
  train/
    config.yaml
    metrics.jsonl
    scores.jsonl
    replay/
    checkpoint.ckpt or ckpt/
  integrity.json
  COMPLETED
```

`integrity.json` checks the task configuration, expected checkpoint step, non-empty Replay, training losses, finite metrics, and fatal error markers. Passing it means the run is structurally valid; it does not mean the policy reproduced a paper score.

Policy evaluation is deliberately separate from this structural verifier. Use the task-specific evaluation scripts and report every preregistered episode according to the corresponding experiment protocol.

For the exact protocols, evaluation rules, and claim boundaries, read:

- [EXP0010_DMC_VISION_PROTOCOL.md](reproduction/EXP0010_DMC_VISION_PROTOCOL.md)
- [EXP0011_ATARI100K_PROTOCOL.md](reproduction/EXP0011_ATARI100K_PROTOCOL.md)
- [EXP0012_MINECRAFT_PROTOCOL.md](reproduction/EXP0012_MINECRAFT_PROTOCOL.md)
- [RESULTS_SCOREBOARD.md](../RESULTS_SCOREBOARD.md)
