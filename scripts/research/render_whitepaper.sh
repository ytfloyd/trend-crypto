#!/usr/bin/env bash
# Render a research markdown doc to a self-contained, K2-branded HTML whitepaper.
# Requires pandoc. Open the HTML and Print -> Save as PDF for a shareable PDF.
# Usage: scripts/research/render_whitepaper.sh docs/research/medallion_lite_whitepaper.md
set -euo pipefail
SRC="${1:-docs/research/medallion_lite_whitepaper.md}"
OUT_DIR="artifacts/whitepaper"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$(basename "${SRC%.md}").html"
pandoc "$SRC" -o "$OUT" --standalone --self-contained --toc --toc-depth=2 \
  --css docs/research/assets/k2_whitepaper.css
echo "rendered -> $OUT"
