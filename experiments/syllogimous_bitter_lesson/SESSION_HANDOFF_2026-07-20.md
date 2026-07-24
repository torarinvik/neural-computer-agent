# Session handoff — 2026-07-20

## Safe resume point

- Checkpoint: `small_mixed_75_25_block2.pt`
- SHA-256: `6275346965c5b0b3c6b4edd684a48a05c50e8a5125c4e5c4f0d18347319193b1`
- Model: 2,350,200 parameters, event Transformer memory, gated-residual recurrent thought cell,
  eight public actions, eight maximum thought steps.
- Sealed parity evaluation: 99.6429% overall. Per length 2/4/8/16/17/18/19:
  100/100/100/99.5/99.5/98.5/100%.
- Sealed hard-attention evaluation: 58.0357%. Per choice count 2–8:
  71.75/78.5/74.5/57.5/46.5/42.75/34.75%.
- The latency bonus is competence-gated and active at this checkpoint.

Do not resume from `small_mixed_mod4_small.pt`: it improved attention to 61.43% but
regressed long parity to 69.5–80.5% and failed to learn cyclic logic.

## What was added

- `cyclic_transfer.py`: deterministic cyclic relation composition with moduli 2, 4, and 8.
- Independent `--cyclic-premises` and ordinary parity premise curricula.
- `--reasoning-family mixed` plus `--cyclic-fraction`, retaining explicit parity replay.
- Unequal deterministic dataset interleaving.
- Reproducible small-run scripts for mixed modulus-4 and isolated learnability controls.
- One-premise cyclic perception controls are now legal.
- All model inputs remain raw RGB frames and PCM. Bookkeeping fields are verifier-only.
- Test state at handoff: 43 tests passing.

## Negative results worth preserving

- Zero-shot modulus-4 at 2/4/8 premises: 50.33% (chance).
- Isolated modulus-4, two premises, 12,000 training examples: 49.6%.
- Isolated modulus-2, two premises, 12,000 training examples: 50.4%.
- The first mixed modulus-4 run replayed parity only at 2/4/8 and caused catastrophic
  forgetting at 16–19. It is rejected, but its reports/checkpoint are retained.

These results say the current `SHIFT0`/`SHIFT1` sensory vocabulary must pass a simpler
perception control before composition is tested. Do not interpret them as evidence that
the architecture cannot learn richer composition.

## Exact next experiment

1. Render visually distinct, compact relation glyphs that remain human-readable.
2. Train/evaluate a one-premise modulus-2 matching control and require near-100% held-out
   accuracy. This diagnoses perception without multi-step composition.
3. If it passes, train two-premise modulus-2 while replaying parity at every mastered length
   2/4/8/16/17/18/19 and hard attention.
4. Require sealed parity, cyclic, and attention audits after every short block.
5. Advance one variable at a time: cyclic premises 1→2→3→4, then modulus 2→4, and only
   later modulus 8. Never accept a blended metric by itself.

## Cloud state at handoff

- Vast SSH used: `ssh -p 52562 root@38.49.42.46 -L 8080:localhost:8080`
- SSH identity: `/Users/torarinvikbjarko/.ssh/id_ed25519_vast_ai`
- Remote workspace: `/root/elisa-screenwatch`
- The instance has no persistent volume. All important new checkpoints, reports, code, and
  scripts were downloaded locally before this handoff.
- The instance may still be live and billable; stopping/destroying it is a separate user action.
