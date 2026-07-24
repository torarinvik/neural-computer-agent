# Neural Computer Agent

A compact research repository for building a real-time neural computer that
learns reusable cognitive primitives from sensory streams and deterministic
outcomes.

Audited model checkpoints are stored in the private Hugging Face repository:
<https://huggingface.co/torarin87/neural-computer-agent>.

The learner receives rendered vision/audio/text streams, its own opaque
actions, its own latent state and memory, and scalar verifier outcomes. It
does not receive game state, coordinates, semantic task labels, rule IDs,
correct-action labels, English chain-of-thought, or counterfactual labels for
actions it did not attempt.

## North star

Maximize verified reusable capability gained per unique interaction.

The project distinguishes:

- unique verifier/reward bits;
- unique logical lifetimes;
- replayed examples;
- optimizer updates;
- GPU and wall time;
- action latency;
- retention and forward-transfer ratios.

Final accuracy alone is not an adequate score.

## Current audited frontier

The two-decision identify-then-act task requires the agent to:

1. emit an opaque probe action;
2. observe its visible consequence;
3. infer the hidden actuator mapping;
4. observe a target;
5. emit the correct opaque action.

The current fresh predictive learner reached:

- 100% held-out accuracy at 64 unique verifier bits;
- 100% accuracy and 100% prediction flips under valid protocol rerenders;
- 100% accuracy and 100% prediction flips under target reversal;
- chance performance when the probe consequence is removed.

An incremental 8→16→32→64-bit learner reached 93.36% with 256 cumulative
optimizer updates. A 32-bit arm with 512 updates failed at 52.73%, so extra
replay does not substitute for the missing unique outcomes.

Earlier fixed-target weights caused negative transfer to the full task.
Inherited weights are therefore retained only when they improve the next
held-out learning curve.

See:

- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`
- `experiments/forward_transfer_attention/MICRO_INTERCEPT_DESIGN.md`
- `experiments/forward_transfer_attention/README.md`

## Repository map

| Path | Contents |
|---|---|
| `experiments/forward_transfer_attention/` | Main sample-efficiency, transfer, memory, binding, and causal-audit research |
| `experiments/syllogimous_neural_computer/` | Learned external-memory neural computer |
| `experiments/syllogimous_latent_agent/` | Latent real-time agent and sensory models |
| `experiments/syllogimous_bitter_lesson/` | Emergent reasoning experiments without symbolic solution machinery |
| `experiments/syllogimous_realtime/` | Real-time deterministic syllogism environment and Elisa sources |
| `experiments/sensory_codec/` | Sparse sensory stream experiments |
| `artifacts/checkpoints/` | Curated current checkpoints that are small enough for Git |
| `artifacts/manifests/` | Checksums for curated and excluded historical artifacts |
| `session_records/` | Compact historical reports and continuation notes |
| `legacy/screenwatch_streamer/` | Original sensory-streamer sources, without compiled binaries |

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA-capable PyTorch is recommended for training. CPU and Apple unified-memory
backends are useful for tests and tiny diagnostics.

## Verify

```bash
python -m pytest experiments/forward_transfer_attention -q
./scripts/verify_curated_artifacts.sh
```

The narrow current-task regression suite is:

```bash
python -m pytest \
  experiments/forward_transfer_attention/test_identify_then_act.py -q
```

## Reproduce the current 64-bit task

```bash
python -m experiments.forward_transfer_attention.train_identify_then_act \
  --report experiments/forward_transfer_attention/reports/reproduction.json \
  --checkpoint-out artifacts/checkpoints/reproduction.pt \
  --device cuda \
  --seed 211 \
  --curriculum-rung random_probe \
  --intention-width 64 \
  --pretrain-lifetimes 128 \
  --pretrain-steps 40 \
  --policy-lifetimes 64 \
  --test-lifetimes 256 \
  --fit-updates 256 \
  --batch-size 32
```

## Next experiment

The first 32-bit feature-interface tournament is complete. A seed-211 clone
reached 72.27% selection and 69.53% blind accuracy, but the family fell to
55.47% blind accuracy on seed 307. No checkpoint was promoted.

The next population should change the learning mechanism rather than continue
interface-width tuning: compare a frozen-core baseline with tiny zero-initialized
latent adapters and conservative recurrent/predictor fine-tuning. Reproduce
exact parents on a second seed before mutation. Every clone remains capped at
32 unique verifier outcomes and must pass blind causal audits.

The longer-term optimization is a gradient-trained population with
successive-halving compute allocation. Fitness is held-out learning AULC,
stable bits-to-threshold, retention, latency, and positive transfer to the next
primitive—not old-task accuracy.
