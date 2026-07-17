# Discussion Workspace

This directory is for Claude/Codex exchange, red-team notes, and unresolved drafts. It is not a second control plane and does not contain canonical project state.

## Ownership

- `claude/`: Claude-authored notes and replies. Codex reads but does not edit them.
- `codex/`: Codex-authored notes and replies. Claude reads but does not edit them.
- `INDEX.md`: Codex-maintained routing index for current unresolved threads.

Use `YYYY-MM-DD_topic.md` filenames. Each note should identify its author, evidence inputs, observations, interpretations, open questions, and requested action. Reply with a new file and link the source note instead of editing another agent's text.

## Promotion Rules

- Verified paper/code/protocol facts move to `docs/reproduction/CLAIM_PROTOCOL_MATRIX.md`.
- Durable route decisions move to `DEVLOG.md`.
- Current judgment and next decision move to `CURRENT_STATE.md`.
- Run facts move to `/root/autodl-tmp/runs/STATUS.md`.
- Formal evidence moves through `research/` and `/root/autodl-tmp/artifacts/`.

Discussion text alone cannot support a replication or method claim. Server process control remains Codex-only under `AGENTS.md`.
