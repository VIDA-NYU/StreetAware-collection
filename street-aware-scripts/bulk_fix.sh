#!/usr/bin/env bash
set -euo pipefail

# Use current directory as base
BASE_DIR="$(pwd)"
echo "Using current directory as base: $BASE_DIR"

# Find and process .avi files safely
find "$BASE_DIR" -type f -path '*/video/*.avi' -print0 | while IFS= read -r -d '' infile; do
  # Ensure path starts with /
  if [[ "$infile" != /* ]]; then
    infile="/$infile"
  fi

  echo "Processing: $infile"

  if [[ ! -f "$infile" ]]; then
    echo "  ✗ File does not exist: $infile"
    continue
  fi

  dir=$(dirname "$infile")
  base=$(basename "$infile" .avi)
  tmpfile="$dir/${base}_tmp.avi"

  echo "  → Running ffmpeg on $infile"
  ffmpeg -y -f mjpeg -framerate 25 -i "$infile" -c:v copy -an "$tmpfile"

  if [[ -f "$tmpfile" ]]; then
    mv -f "$tmpfile" "$infile"
    echo "  → Overwritten original with fixed file."
  else
    echo "  ✗ ffmpeg failed for $infile; original untouched."
  fi
done
