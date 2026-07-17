# Ingestion Manifest 格式

本目录的 `*.sha256` 不是标准 `sha256sum -c` 两列文件，而是三列 source-to-destination ledger：

```text
source_sha256  destination  source
```

同一个 SHA256 必须同时匹配 canonical `destination` 和原始 staging `source`。迁移到独立 control
repo 后，destination 相对本仓，source 相对保留的旧 runtime。路径当前不含空格，可用：

```bash
control_root="$(git rev-parse --show-toplevel)"
source_root="${SOURCE_ROOT:-/root/autodl-tmp/dreamerv3}"
legacy_root="${LEGACY_ROOT:-/root/autodl-tmp/dreamerv3}"
while read -r expected destination source; do
  case "$expected" in ''|'#'*) continue ;; esac
  destination_path="$control_root/$destination"
  # Full paper text is license-sensitive and remains in the local legacy store.
  test -e "$destination_path" || destination_path="$legacy_root/$destination"
  for path in "$destination_path" "$source_root/$source"; do
    test "$(sha256sum "$path" | awk '{print $1}')" = "$expected" || exit 1
  done
done < references/manifests/2026-07-17_claude-snapshot.sha256
```

若未来路径可能含空格，应改用 JSONL manifest 和结构化解析器，不再扩展此空白分隔格式。
