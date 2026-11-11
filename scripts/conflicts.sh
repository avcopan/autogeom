#!/usr/bin/env bash
# Check for unresolved merge conflict markers, and list where they are.
# Run this directly (./scripts/conflicts.sh) if pixi.toml itself has conflicts.
set -euo pipefail

if git grep -n -E '^(<<<<<<<|>>>>>>>)( |$)'; then
  echo "Error: unresolved merge conflict markers (listed above)"
  exit 1
fi
