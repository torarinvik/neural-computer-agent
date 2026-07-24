#!/bin/sh
# arch-ocr.sh — OCR an exact archived frame (Tier A ring). Resolves a story.md EVIDENCE pin or a
# zoom query to positioned text at source resolution, long after the grid batch was pruned.
#
#   arch-ocr.sh <dir> <seq> [x,y,w,h] [min-conf]
#
# Decodes archive frame <seq> to a temp PPM via arch_tool, then runs screenocr (which reads PPM
# directly). Coordinates are full-res pixels; use SPEC.md's grid->pixel formula to build the crop.
set -e
DIR=${1:?usage: arch-ocr.sh <dir> <seq> [x,y,w,h] [min-conf]}
SEQ=${2:?usage: arch-ocr.sh <dir> <seq> [x,y,w,h] [min-conf]}
CROP=$3
CONF=${4:-0.4}
HERE=$(dirname "$0")
TMP=$(mktemp -t archocr).ppm

"$HERE/arch_tool" show "$DIR" "$SEQ" "$TMP" >/dev/null
if [ -n "$CROP" ]; then
    "$HERE/screenocr" "$TMP" --crop "$CROP" --min-conf "$CONF"
else
    "$HERE/screenocr" "$TMP" --min-conf "$CONF"
fi
rm -f "$TMP"
