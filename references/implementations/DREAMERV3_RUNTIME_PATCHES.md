# DreamerV3 runtime compatibility patches

## Lineage

- Upstream: `https://github.com/danijar/dreamerv3.git`
- Frozen upstream base: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
- Local runtime head: `5168475b7a4413f9575933b4580e7073caea2114`
- Complete local bundle: `/root/autodl-tmp/artifacts/dreamerv3/provenance/runtime-5168475.bundle`
  (SHA256 `d8b7a2acc891923c83a33f43af4de9941d9c2e4a305816bad8d36624d89442fb`)

The tracked patches make this runtime state recoverable from the remote control repository without write access to
the author's upstream repository:

| Patch | Purpose | Algorithm behavior | SHA256 |
|---|---|---|---|
| `patches/dreamerv3/0001-fix-runtime-handle-log-images-in-multi-env-evaluatio.patch` | Ignore image-valued `log/*` observations from nonzero eval workers instead of asserting scalar shape; adds regression test | No policy, environment, or score change; fixes multi-env `eval_only` logging | `3f42c193c62e281fc32c0082a5b7707ee577581788c74d7d06e6acb2b0751cc7` |
| `patches/dreamerv3/0002-fix-runtime-optionally-save-checkpoint-at-run-end.patch` | Add default-off `run.save_at_end`; explicit runs save once after natural train-loop exit | No loss/update/policy change; only writes an exact terminal checkpoint when enabled | `5317f6d6e2bb6507a0cdd9962ad820f20459309ee15919e25542125ef77f28b3` |

`MUJOCO_GL=egl` is an executor environment setting and is intentionally not a runtime source patch.

## Recovery

Starting from a clean checkout at the upstream base:

```bash
git am /path/to/dreamerv3-reproduction/patches/dreamerv3/*.patch
```

Because `git am` creates new committer metadata, the recreated commit IDs may differ. Verify the resulting tree ID
with `git rev-parse HEAD^{tree}`; it must be `1e2c715d0e3d51fd653659ea2831b9e0cc96d0a1`. The second patch remains
behaviorally inactive unless a frozen run explicitly passes `--run.save_at_end True`.
