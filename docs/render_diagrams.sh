#!/usr/bin/env bash
# Re-render the D2 diagrams in source/_static/diagrams to SVG.
# Requires d2 (https://d2lang.com). Usage: docs/render_diagrams.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")/source/_static/diagrams"
if [ "$#" -gt 0 ]; then files=("${@/%/.d2}"); else files=(*.d2); fi
for f in "${files[@]}"; do
  # --scale 1 writes width/height so the docs show each diagram at its natural
  # size (capped by the theme's max-width) instead of stretching it.
  d2 --scale 1 --elk-nodeNodeBetweenLayers 30 --elk-edgeNodeBetweenLayers 16 \
     --elk-padding "[top=34,left=16,bottom=16,right=16]" \
     "$f" "${f%.d2}.svg"
done
