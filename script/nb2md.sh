#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <notebook.ipynb> <output.md>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# 1. Convert notebook → markdown
# --output sets the *filename only*
# --output-dir controls where the file is written
jupyter nbconvert --to markdown "$INPUT" \
  --output "$(basename "$OUTPUT")" \
  --output-dir "$(dirname "$OUTPUT")"

# 2. (Optional) Normalize markdown via pandoc
pandoc "$OUTPUT" -f gfm -t gfm -o "$OUTPUT"

echo "✅ Markdown documentation generated at: $OUTPUT"
