#!/bin/bash
cd "/Users/torarinvikbjarko/Documents/Machine Learning Projects/neural-computer-agent-games"
SP="/private/tmp/claude-501/-Users-torarinvikbjarko-Documents-Machine-Learning-Projects-neural-computer-agent-games/7d2d988c-c6fa-4641-b361-ceacd938889d/scratchpad"
run() {
  local tag="$1"; local seed="$2"; shift 2
  OMP_NUM_THREADS=1 PYTHONPATH="$PWD" uv run python \
    experiments/games_amodal/probes/reacher_ladder.py \
    --seed "$seed" --rung r4 --sparse --updates 400 --eval-batches 2 \
    --targets=r2,r3,r4 "$@" > "$SP/wd-$tag-$seed.json" 2>"$SP/wd-$tag-$seed.err"
}
export -f run; export SP
jobs=()
for seed in 69316 69317 69318 69319 69320; do
  jobs+=("run model $seed --model-search=10")
  jobs+=("run policy $seed --retrieval-first")
done
printf "%s\n" "${jobs[@]}" | xargs -P 5 -I{} bash -c '{}'
echo WIDEN_DONE
