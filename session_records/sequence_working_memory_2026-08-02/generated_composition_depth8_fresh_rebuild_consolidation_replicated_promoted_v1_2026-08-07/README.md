# Fresh-rebuild neural consolidation of depth-eight procedures (2026-08-07)

Status: replicated promoted bounded external-memory result.

The audit trained two independently acquired eight-step procedures, then
compared an inherited three-slot external student with a fresh three-slot
student. The inherited controller stayed frozen. The fresh student was the
stable-prefix winner in both seeds and was admitted only because the audit was
run with `--allow-fresh-consolidation`; the memory verifier independently
checked every protected source alias before replacing the two rows.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| source behavior after acquisition | `1.0000/1.0000` | `1.0000/1.0000` |
| fresh-rebuild behavior after reload | `1.0000/1.0000` | `1.0000/1.0000` |
| fresh retention probes | `1.0000` on all 8 | `1.0000` on all 8 |
| physical rows | `2 -> 1` | `2 -> 1` |
| payload ratio | `0.7392` | `0.7392` |
| target reload behavior | `0.9961` | not required/attempted |
| controller digest | unchanged | unchanged |
| replayed examples | `0` | `0` |

Each replica consumed `202,752` unique verifier bits, `68,096` unique logical
lifetimes, and `2,176` optimizer updates. Wall time was `249.2s` and `251.3s`.

The accepted candidate is a fresh-outcome rebuild, not inherited positive
transfer. It proves that isolated external memory can behavior-verify a
smaller shared executable artifact and preserve both protected depth-eight
capabilities without replaying old examples. Target transfer is therefore
reported as unqualified for this result; the seed-69316 target arm was a
successful diagnostic, while seed-69317 did not require it for promotion.

The rejected controls are retained here because they identify the boundary:

- Three depth-eight sources failed shared retention at both 256 and 512
  consolidation updates, even after expanding from two to three slots.
- Two depth-eight sources passed behavior but were rejected under the default
  inherited-only admission policy when the fresh winner was not allowed.

The positive mechanism is therefore verified fresh rebuild plus independent
retention gating, not unrestricted continual learning, arbitrary program
induction, or positive transfer from inherited weights.
