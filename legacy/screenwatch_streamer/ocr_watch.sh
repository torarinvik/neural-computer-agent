#!/bin/sh
# ocr_watch.sh — eager OCR trigger for the screen orchestra (SPEC.md, the "guitar" member).
#
# Watches the batch stream and runs screenocr on a batch's full-res keyframe JPEG when the
# scene plausibly changed: ACTIVITY switching/video, or the first non-still batch after a
# still one. Calm batches never pay for OCR (attention pyramid, invariant I5).
#
# Usage: ocr_watch.sh [batch_dir] [screenocr_path]

DIR="${1:-/tmp/screen_batches}"
OCR="${2:-$(dirname "$0")/screenocr}"

last=""
prev_act="switching"   # treat the first batch as a scene change (cold start)

while :; do
  n=$(cat "$DIR/latest.txt" 2>/dev/null)
  if [ -z "$n" ] || [ "$n" = "$last" ]; then sleep 1; continue; fi
  last="$n"
  f="$DIR/batch_$n.txt"
  jpg="$DIR/batch_$n.jpg"
  out="$DIR/batch_$n.ocr.txt"
  [ -f "$f" ] || continue

  act=$(sed -n '3p' "$f" | awk '{print $2}')
  run=0
  case "$act" in
    switching|video) run=1 ;;
    *) [ "$prev_act" = "still" ] && [ "$act" != "still" ] && run=1 ;;
  esac
  prev_act="$act"

  if [ "$run" = 1 ]; then
    # the jpg conversion is async; give it a moment if it hasn't landed yet
    for _ in 1 2 3 4 5; do [ -f "$jpg" ] && break; sleep 0.3; done
    [ -f "$jpg" ] && "$OCR" "$jpg" > "$out.tmp" 2>/dev/null && mv "$out.tmp" "$out"
  fi
done
