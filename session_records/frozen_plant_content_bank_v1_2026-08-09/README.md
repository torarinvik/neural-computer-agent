# Frozen structural plant + external content bank (F75)

Leave-one-out over the four disjoint families, 3 seeds. Plant pre-trained on
48 families sampled from the schema (each with its own 16-token bank entry),
then FROZEN; the held-out family is learned by fitting a fresh entry alone.

    bank_plant.py --held-out <family> --seed <s> --pretrain-families 48 \
                  --bank-tokens 16 --pretrain-updates 3000 --updates 600

## Result: one gate passed, one gate failed

**Forgetting solved.** Retention delta after fitting the held-out entry, over
96 measurements: min +0.0000, max +0.0000. True by construction (frozen plant,
separate entries); measured to prove the interface does not leak content into
weights.

**Structure transfer real and causal.** Held-out accuracy:

| held out | schema-pretrained | scrambled-pretrained | random plant |
| --- | ---: | ---: | ---: |
| line | 1.000 | 1.000 | 0.312 |
| dial | 0.994 | 0.684 | 0.005 |
| toggle | 1.000 | 0.557 | 0.007 |
| perm | 0.898 | 0.264 | 0.009 |

`line` (8 states, 2 actions) is too easy to discriminate and is not evidence.

**Cost gate FAILED.** Bank 123 updates mean vs cold 62; cheaper than cold in
only 2/12 runs; 11/12 reach 0.98. Pre-training (2350 updates over 48 families)
is on top and cannot be amortised into a win, because per-task cost already
exceeds cold. The bank arm trains 1,024 parameters against cold's 68,936.

## Honesty notes

The single smoke run (perm/69316: bank 50 vs cold 75) looked like a clean pass
and was one of the two wins out of twelve. Sixth single-seed signal in this
project to fail replication.

Two bugs were found by reading this probe's own output before recording any
result: the scrambled control redrew permutations inside its row loop and
collided seeds on same-length names; retention was an absolute score rather
than a delta. Fixing the control moved scrambled `toggle` from 1.000 to 0.557.

## Next

Amortised entry inference: an encoder mapping a few (state, action, next state)
triples directly to an entry, so acquisition is forward passes not gradient
steps. Prediction recorded in advance in MEMORY_BANK_DESIGN.md.
