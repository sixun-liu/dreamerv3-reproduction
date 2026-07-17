# Project References

This directory is the stable literature and understanding layer for the project. It indexes primary papers, extracted text, code-lineage references, verified reading notes, and literature surveys. It does not replace `research/`, `CURRENT_STATE.md`, `DEVLOG.md`, or formal artifacts.

## Source Classes

| Class | Meaning | Evidence use |
|---|---|---|
| `primary` | Paper, supplement, official dataset/curve, original source release | May support a claim after version and hash verification |
| `author` | Author-maintained implementation, website, talk, or later clarification | Supports lineage/context; protocol drift must be explicit |
| `third_party` | Community implementation, benchmark, survey, or reproduction | Comparative evidence only until independently verified |
| `internal_synthesis` | Claude/Codex/user reading note or code map | Navigation and hypothesis generation, not standalone claim evidence |
| `lead` | Unverified URL, memory, search result, or recommendation | Must remain in `discussion/` or `surveys/` until checked |

## Layout

- `papers/`: index and extracted text snapshots; canonical PDF binaries remain on the data disk at `/root/autodl-tmp/papers/`.
- `understanding/claude/`: stable snapshots of Claude-authored paper and code understanding.
- `implementations/`: verified code-lineage and license index.
- `surveys/`: literature-search outputs and future adapters for Claude's research workflow.
- `manifests/`: source-to-destination hashes for each ingestion batch.

## Ingestion Rules

1. `docs_from_claude/` is a staging inbox and may change while Claude works.
2. Confirm that no transfer process has the inbox open before snapshotting.
3. Ignore cache/checkpoint files and ingest only named stable outputs.
4. Record source path, destination path, timestamp, author, source class, and SHA256.
5. Keep the inbox intact until the author confirms delivery; do not use symlink compatibility layers.
6. Promote verified facts into the claim-protocol matrix or DEVLOG. Do not cite the staging path in formal results.

Literature notes must separate direct quotation/paper fact, code fact, reproduction observation, and interpretation. Unknown citation details remain `unknown`; they are never completed from memory.
