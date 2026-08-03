# Complement margin-loss frontier — 2026-08-03

The binary complement learner was given a constant-gradient margin objective
in addition to its outcome-only binary loss. This is still verifier-side
diagnostic training: the learner receives only controller-visible latents,
opaque attempted actions, and scalar attempted-action outcomes. No correct
unattempted action or operation label enters the buffer.

## 512-lifetime acquisition

| Seed | Margin | Independent complement | Zeroed slot | Causal gain | Span 9 Δ | Span 10 Δ | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 93775 | 1.0 | 63.52% | 50.48% | +13.04 pp | −1.55 | −1.54 | safe |
| 93776 | 1.0 | 61.59% | 50.71% | +10.88 pp | −0.15 | −0.30 | safe |

Both seeds pass the +5-point causal bar and the two-point old-skill retention
gate. A shuffled-outcome control (seed `93782`) scored **50.09%** on the
independent complement audit, while the truthful seed `93775` scored **63.70%**
in the matched audit. Blank-cue and complete-memory-reset controls remained at
chance. This is a replicated, retention-safe 512-lifetime recipe and is the
best current complement acquisition arm.

## 1,024-lifetime escalation

| Seed | Margin | Complement | Causal gain | Span 9 Δ | Span 10 Δ | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 93777 | 1.0 | 66.73% | +16.02 pp | −3.23 | −2.14 | reject |
| 93779 | 0.25 | 66.87% | +16.35 pp | −0.39 | −0.43 | safe, single seed |
| 93780 | 0.25 | 67.20% | +16.36 pp | −2.96 | −2.20 | reject |
| 93781 | 0.1 | 68.06% | +17.58 pp | −2.02 | −1.58 | reject |

Lowering the margin improves the new-task score but does not remove
seed-sensitive interference. No 1,024 arm is promoted. The one safe seed is
kept as a diagnostic candidate only.

## Protected continuation

Continuing an already learned 512-lifetime slot with 256 or 512 fresh
complement lifetimes preserved span nine/span ten, but improved the complement
by only **+0.33/+0.76 points** for the ordinary loss. A margin-loss continuation
actually fell **−1.89 points** on the complement. Thus the current slot learns
the adjacent primitive efficiently once, but does not yet produce a strong
compounding gain from more of the same experience.

## Decision and next frontier

Use the replicated margin-1.0 512-lifetime arm as the current partial
capability checkpoint. Do not claim mastery or scale blindly past 1,024. The
remaining frontier is separating a new-slot residual's useful contexts from
old-task contexts without relying on parent-action logits, probabilities, or
simple provenance gates. The next candidate should be a genuinely selective
promotion/rejection population or an interference-aware, task-agnostic gate,
with the same causal, shuffled, blank, reset, and retention audits.
