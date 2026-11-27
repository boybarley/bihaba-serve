#!/usr/bin/env bash
# bihaba-serve clean uninstaller (NEVER touches ~/models!)
# Created by Boy Barley from Bihaba (Team Pasai Aceh)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST="${HOME}/Library/LaunchAgents/com.bihaba.serve.plist"
LEGACY_PLIST="${HOME}/Library/LaunchAgents/com.taalas.qwen35.plist"

echo "▸ Uninstalling bihaba-serve..."

# 1. Stop & remove launchd services if present
for P in "${PLIST}" "${LEGACY_PLIST}"; do
  if [[ -f "${P}" ]]; then
    launchctl unload "${P}" 2>/dev/null || true
    rm -f "${P}"
    echo "  ✓ Removed launchd service: ${P}"
  fi
done

# 2. Remove CLI & shared binaries
rm -f "${HOME}/.local/bin/bihaba-serve"
rm -rf "${HOME}/.local/share/bihaba-serve"
echo "  ✓ Removed ~/.local/bin/bihaba-serve & ~/.local/share/bihaba-serve"

# 3. Clean build artifacts inside repo
rm -rf "${REPO_DIR}/build" "${REPO_DIR}/python/bihaba_serve/__pycache__"
echo "  ✓ Cleaned build artifacts"

echo "✓ bihaba-serve uninstalled cleanly (your ~/models directory was NOT modified)."
