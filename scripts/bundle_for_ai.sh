#!/usr/bin/env bash
# Gabungkan file yang relevan untuk satu tugas jadi SATU file teks, untuk dilampirkan ke AI
# yang tidak bisa membuka GitHub. Pemakaian:  bash scripts/bundle_for_ai.sh <tugas>
#   tugas: moe-gemm | prefill-qwen35 | transformer | batching | server | semua
set -euo pipefail
cd "$(dirname "$0")/.."
task="${1:-}"
case "$task" in
  moe-gemm|prefill-qwen35) files=(docs/CATATAN_TEKNIS.md engines/qwen35/kernels.metal engines/qwen35/main.mm) ;;
  transformer)             files=(docs/CATATAN_TEKNIS.md engines/transformer/kernels.metal engines/transformer/main.mm) ;;
  batching)                files=(docs/CATATAN_TEKNIS.md engines/qwen35/main.mm engines/transformer/main.mm
                                  python/bihaba_serve/daemon.py python/bihaba_serve/server.py python/bihaba_serve/cli.py) ;;
  server)                  files=(docs/CATATAN_TEKNIS.md python/bihaba_serve/*.py python/bihaba_serve/web/index.html tests/test_offline.py) ;;
  semua)                   files=(docs/CATATAN_TEKNIS.md README.md Makefile install.sh engines/*/kernels.metal engines/*/main.mm
                                  python/bihaba_serve/*.py python/bihaba_serve/web/index.html tests/test_offline.py) ;;
  *) echo "Pakai: bash scripts/bundle_for_ai.sh {moe-gemm|prefill-qwen35|transformer|batching|server|semua}"; exit 1 ;;
esac
out="ai_bundle_${task}.txt"
{
  echo "# bihaba-serve — bundle untuk tugas: ${task}"
  echo "# commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')  tanggal: $(date '+%Y-%m-%d %H:%M')"
  echo "# Setiap file diawali baris '===== FILE: <path> ====='. Kembalikan perubahan sebagai file utuh per path."
  for f in "${files[@]}"; do
    echo; echo "===== FILE: $f ====="; cat "$f"
  done
} > "$out"
echo "✓ $out ($(wc -c < "$out" | awk '{printf "%.0f KB", $1/1024}'), ${#files[@]} file, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
