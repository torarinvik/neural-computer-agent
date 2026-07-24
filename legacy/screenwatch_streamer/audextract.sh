#!/usr/bin/env bash
# audextract.sh — pull a video file's audio into the cymbal's exact input format
# (orchestra_v3_plan.md V0.2). The audio twin of vidingest: 16 kHz mono s16le WAV, the format
# audiotriage/screenaud parse. Segment flags mirror vidingest so audio and video stay aligned to
# the same t0 (gold timestamps stay segment-relative, 0-based).
#
# USAGE: audextract.sh <video> <out.wav> [--t0 <s>] [--dur <s>]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: audextract.sh <video> <out.wav> [--t0 <s>] [--dur <s>]" >&2
    exit 2
fi

video="$1"; out="$2"; shift 2
t0=""; dur=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --t0)  t0="$2";  shift 2 ;;
        --dur) dur="$2"; shift 2 ;;
        *) echo "audextract.sh: unknown arg $1" >&2; exit 2 ;;
    esac
done

# Output seeking (-ss/-t after -i) is frame/sample-accurate, matching vidingest's accurate seek.
seek=(); [[ -n "$t0" ]] && seek+=(-ss "$t0")
clip=(); [[ -n "$dur" ]] && clip+=(-t "$dur")

ffmpeg -nostdin -hide_banner -loglevel error -y \
    -i "$video" ${seek[@]+"${seek[@]}"} ${clip[@]+"${clip[@]}"} \
    -ac 1 -ar 16000 -c:a pcm_s16le -f wav "$out"

echo "audextract: $out  (16 kHz mono s16le)"
