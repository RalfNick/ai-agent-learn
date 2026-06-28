#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Checking Phase7 Hermes Feishu templates under: $ROOT"

if grep -RInE '(sk-[A-Za-z0-9_-]{20,}|cli_[A-Za-z0-9]{12,}|secret_[A-Za-z0-9_-]{12,}|xox[baprs]-|ghp_[A-Za-z0-9_]{20,})' "$ROOT"; then
  echo "Potential real secret found. Redact it before committing."
  exit 1
fi

placeholder_pattern='TB[D]|TO[D]O|'"$(printf '\345\276\205\350\241\245')"
if grep -RInE "$placeholder_pattern" "$ROOT"; then
  echo "Unresolved placeholder marker found."
  exit 1
fi

required_files=(
  "$ROOT/config/hermes.env.example"
  "$ROOT/config/hermes-config.example.yaml"
  "$ROOT/deploy/hermes-gateway.service.example"
  "$ROOT/CHECKLIST.md"
  "$ROOT/ACCEPTANCE.md"
  "$ROOT/PRACTICE_LOG.md"
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "Required file is missing or empty: $file"
    exit 1
  fi
done

grep -q 'approvals:' "$ROOT/config/hermes-config.example.yaml"
grep -q 'cron_mode: deny' "$ROOT/config/hermes-config.example.yaml"
grep -q 'backend: docker' "$ROOT/config/hermes-config.example.yaml"
grep -q 'FEISHU_CONNECTION_MODE=websocket' "$ROOT/config/hermes.env.example"
grep -q 'FEISHU_GROUP_POLICY=allowlist' "$ROOT/config/hermes.env.example"
grep -q 'HERMES_YOLO_MODE must not be set' "$ROOT/config/hermes.env.example"

echo "Templates verified."
