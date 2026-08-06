# Qualified: two games through one frozen amodal controller (2026-08-06)

This rung wires both games into the production N-to-M runtime: per-game
frontends emit opaque events, one `AmodalCognitiveController` performs all
recurrent computation, and per-game `KeypressDecoder` heads lower opaque
intentions to keypresses. Snake trains the core end to end; the core is then
frozen and hashed; Pong is acquired by new peripherals only, through the
frozen core; Snake is re-audited. Controls: a reward-shuffled Pong twin and a
random-core Pong twin (same peripheral budget through a never-trained frozen
controller).

Command (per seed):

```bash
uv run python -m experiments.games_amodal.shared_controller \
  --seed <seed> --updates 600 --batch-size 64 --steps 64 \
  --gamma 0.95 --learning-rate 1e-3 \
  --event-width 64 --intent-width 32 --feedback-width 16 --hidden 32 \
  --eval-seeds 8
```

## Result: one seed promoted, one rejected on transfer

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake through controller | 0.8555 | 0.9297 |
| Snake retained after Pong (score and core hash) | exact | exact |
| Pong through trained frozen core | 0.5195 | 0.9082 |
| Pong through random frozen core | 0.9336 | 0.9570 |
| core transfer margin | -0.4141 | -0.0488 |
| updates to half mastery, trained vs random core | 221 vs 203 | 119 vs 215 |
| Pong reward-shuffled twin | 0.0371 | 0.0684 |
| replayed examples | 0 | 0 |

Structural gates pass on both seeds: exact Snake retention, bit-for-bit
frozen core, causal null near chance, zero replay. Seed 69317 additionally
passes all acquisition gates and shows nearly twice-as-fast Pong acquisition
through the trained core (half mastery at update 119 versus 215). Seed 69316
shows genuine negative transfer: the same budget that takes a random core to
0.9336 plateaus at 0.5195 through the Snake-shaped core.

The low-budget control (`*_lowbudget_control.json`, 300 updates at batch 32)
is preserved: there the trained core beat the random core on both seeds
(+0.53, +0.20) only because the random core was under-trained, which is
exactly why endpoint-only transfer claims are unreliable.

## Claim boundary

Promoted: the N-encoder/one-controller/M-decoder composition works for real
games with exact retention, a frozen core, and zero replay; games are cheap
enough that fresh peripherals can learn them through an arbitrary fixed
recurrent map. Not promoted: reliable positive transfer from a game-trained
core — it is seed-sensitive and can be strongly negative. This replicates
the parent repository's standing bottleneck (safe reuse works; consistent
positive transfer does not) on a real-game substrate. The next levers are
controller growth registers or prior calibration rather than more budget:
budget escalation was already the applied repair between the control and
this run.
