# Robust three-appearance relation acquisition

## Breakthrough

The previously seed-sensitive third-appearance result is now a fixed,
replicated training recipe.  The unchanged controller architecture learned
disconnected dot pairs from the two-appearance bars-and-diamonds parent on all
three fresh preregistered seeds:

| seed | bars | diamonds | dot pairs | binary | visible | XOR |
|---:|---:|---:|---:|---:|---:|---:|
| 9671 | 99.76% | 96.68% | 97.71% | 92.81% | 91.08% | 91.13% |
| 9672 | 99.65% | 97.54% | 97.73% | 94.43% | 92.57% | 91.10% |
| 9673 | 99.41% | 95.63% | 97.99% | 96.18% | 92.97% | 92.45% |
| mean | 99.61% | 96.62% | 97.81% | 94.47% | 92.21% | 91.56% |

Every required capability and retention gate passed on every seed.  This
replaces the earlier lucky 1,792-lifetime endpoint with a robust threshold:

- 64 acquisition updates followed by 32 consolidation updates;
- 3,072 unique new-appearance lifetimes;
- 18,432 new-appearance verifier bits;
- 64,512 total verifier bits including five balanced rehearsal streams;
- 12.47 seconds mean end-to-end training and evaluation time.

The preceding diamond bridge used 10,240 new lifetimes and 184,320 total
verifier bits.  The robust third appearance therefore still costs 3.33 times
fewer new lifetimes and 2.86 times fewer total verifier outcomes than the
second appearance.

The promoted checkpoint is
`artifacts/checkpoints/unified_pair_relation_robust_three_appearance_seed9672.pt`,
SHA-256
`1ff5d38258a8c683fbc7dcfd6a1098e20c18ae35d5372d377fd8874e29544f54`.

## Independent audit

The promoted seed-9672 controller received a fresh 8,192-lifetime audit:

| appearance | normal accuracy | missing-second-object control |
|---|---:|---:|
| bars | 99.65% | 49.41% |
| diamonds | 97.85% | 49.81% |
| dot pairs | 97.78% | 50.01% |

Blank vision returned to chance, and valid rendered counterfactuals,
prediction flips, reversed rules, active-state reset, and feedback controls all
passed.  The controller still masters all three appearances with zero optional
thought passes: one controller evaluation per sensory event, the physical
minimum.

## What actually caused the gain

The experiment began as a search for an adaptive population selector and then
as a test of a new additive gate extension.  The controls corrected both
stories:

1. A held-out population race did not reliably select a valid candidate and
   spent substantially more evidence than the fixed recipe.
2. A zero-output 64-unit additive gate extension replicated on all three
   seeds, but so did a matched run that trained the whole slot.
3. Most decisively, the old architecture with no extension replicated on all
   three seeds under the same 64+32 schedule.

The breakthrough is therefore the longer acquire-then-consolidate curriculum,
not added model capacity.  The extension is a safe experimental mechanism, but
it is neither necessary nor promoted.

## Rejected interventions

- Strong global retention during acquisition suppressed dot learning.
- Larger replay batches also suppressed acquisition.
- Static and loss-prioritized replay weights optimized the wrong short-run
  proxy and did not produce a better retained repertoire.
- Refiner-only acquisition learned dots quickly but damaged binary mapping and
  diamonds.
- Population selection was costlier and less reliable than moving the fixed
  schedule past the measured ignition valley.

These are bounded negatives for the tested budgets, not impossibility claims.

## Next frontier

Execution is already compiled to minimum latency for this primitive.  The next
high-return test alternates the curriculum axis: acquire a genuinely new
relation, such as larger/smaller, on the already familiar bars, diamonds, and
dot-pair appearances.  Measure experience-to-threshold first, then optimize
optional thought only if the new relation actually needs it.

The required compounding comparison is:

1. the full retained three-appearance parent;
2. a matched parent with the pair-relation slot reset;
3. identical unique lifetimes, verifier bits, seeds, and stopping rule;
4. retention gates after every promoted rung.

That separates reuse-driven sample efficiency from mere additional capacity.
