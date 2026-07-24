#!/usr/bin/env bash
set -euo pipefail

# Deterministic, local validation package. VLM/GPU runs are intentionally
# separate because they depend on an installed checkpoint and accelerator.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m experiments.syllogimous_realtime.semantic_audit \
  --output experiments/syllogimous_realtime/semantic_audit.json
python3 -m experiments.syllogimous_realtime.validate_native_challenges \
  --output experiments/syllogimous_realtime/native_challenge_crosscheck.json
python3 experiments/syllogimous_realtime/validate_challenges.py \
  --count 1000000 --difficulty max \
  --report experiments/syllogimous_realtime/challenge_validation_max_1m.json
python3 -m experiments.syllogimous_realtime.run_gate_ablation \
  --packets 20 --output experiments/syllogimous_realtime/gate_ablation.json
python3 -m experiments.syllogimous_realtime.boundary_leak \
  --count 256 --output experiments/syllogimous_realtime/boundary_leak.json
for baseline in random-policy text-only-oracle vision-only vision-plus-audio full-streaming-model; do
  python3 -m experiments.syllogimous_realtime.run_baselines \
    --baseline "$baseline" --episodes 1000 --premises 6 --deadline-ms 8000 \
    --inference-ms 1 --seed 0 \
    --output "experiments/syllogimous_realtime/baseline_metrics/$baseline.json"
done
python3 -m unittest experiments.syllogimous_realtime.test_tooling \
  experiments.syllogimous_realtime.test_environment
