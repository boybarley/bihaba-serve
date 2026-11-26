#!/usr/bin/env bash
# bihaba-serve 1-command installer (macOS Apple Silicon)
# Created by Boy Barley from Bihaba (Team Pasai Aceh)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"
SHARE_DIR="${HOME}/.local/share/bihaba-serve"

echo "══════════════════════════════════════════════════════════"
echo "  bihaba-serve Installer (Apple Silicon Metal Engine)"
echo "  Created by Boy Barley — Bihaba (Team Pasai Aceh)"
echo "══════════════════════════════════════════════════════════"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: bihaba-serve requires macOS on Apple Silicon (M1/M2/M3/M4)." >&2
  exit 1
fi
if ! command -v clang++ >/dev/null 2>&1; then
  echo "ERROR: clang++ not found. Install Xcode Command Line Tools:" >&2
  echo "  xcode-select --install" >&2
  exit 1
fi
PY_BIN="$(command -v python3)"
echo "▸ Using Python: ${PY_BIN} ($(${PY_BIN} --version))"

if ! "${PY_BIN}" -c "import tokenizers" >/dev/null 2>&1; then
  echo "▸ Installing tokenizers (only Python dependency)..."
  "${PY_BIN}" -m pip install --user tokenizers
else
  echo "✓ Python tokenizers already installed"
fi

# Compile BOTH isolated Metal engines (~2 seconds total)
cd "${REPO_DIR}"
make all

mkdir -p "${BIN_DIR}" "${SHARE_DIR}"
cp "${REPO_DIR}/build/bihaba-qwen35" "${SHARE_DIR}/bihaba-qwen35"
cp "${REPO_DIR}/build/qwen35_kernels.metal" "${SHARE_DIR}/qwen35_kernels.metal"
cp "${REPO_DIR}/build/bihaba-transformer" "${SHARE_DIR}/bihaba-transformer"
cp "${REPO_DIR}/build/transformer_kernels.metal" "${SHARE_DIR}/transformer_kernels.metal"

cat <<EOF > "${BIN_DIR}/bihaba-serve"
#!/usr/bin/env bash
export BIHABA_SERVE_HOME="${REPO_DIR}"
export BIHABA_SHADER_DIR="${SHARE_DIR}"
export PYTHONPATH="${REPO_DIR}/python\${PYTHONPATH:+:\$PYTHONPATH}"
exec "${PY_BIN}" -m bihaba_serve.cli "\$@"
EOF
chmod +x "${BIN_DIR}/bihaba-serve"

echo ""
echo "✓ Installed CLI: ${BIN_DIR}/bihaba-serve (qwen35 + transformer engines ready!)"
if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
  echo "▸ Add ~/.local/bin to your PATH (run once):"
  echo "    echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.zshrc && source ~/.zshrc"
fi
echo ""
echo "Quickstart:"
echo "  bihaba-serve list ~/models"
echo "  bihaba-serve serve --models-dir ~/models --model ~/models/Bihaba-bagus-pakai --top-k 4 --port 8090"
