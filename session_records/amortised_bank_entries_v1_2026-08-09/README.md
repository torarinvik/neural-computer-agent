# Amortised bank entries: reading beats fitting, but not scratch (F76)

An encoder maps observed (state, action, next state) triples directly to a bank
entry. Acquiring a family is one forward pass: zero gradient steps, zero
weights moved. Plant and encoder trained together over 256 schema-sampled
families, 20000 updates, then both frozen.

    amortised_bank.py --pool 256 --train-updates 20000 --query 256 --context 128
    nulls: --scramble  --random-plant

## Reading is real

| | in-distribution read accuracy |
| --- | ---: |
| trained (seeds 69316/17/18) | 0.918 / 0.903 / 0.937 |
| scrambled-dynamics null | 0.210 |
| random-plant null | 0.040 |

Wrong-context null (encoder fed a DIFFERENT family's transitions): 0.000-0.065.
The entry carries content; nothing is memorised.

## Novel families, same generator, never trained on

| arm | read acc | mastered by reading | fine-tune cost | cold cost |
| --- | ---: | ---: | ---: | ---: |
| trained | 0.682 | 2.3/16 | 83.8 | 49.5 |
| scrambled | 0.147 | 0/16 | 273.4 | 40.6 |
| random plant | 0.026 | 0/16 | 600.0 | 40.6 |

68% of a novel family's dynamics correct at zero gradient cost, both nulls dead.

## The gate still fails

Fine-tuning from a partially correct entry costs 84 against cold's 50, and only
2.3/16 novel families reach 0.98 by reading. The frozen plant caps repair:
1,024 entry parameters against 68,936 free ones. Freezing is what makes
retention perfect (delta 0.0000 everywhere) and what caps expressivity — one
mechanism, not two.

Total accounting: 20000 pre-training updates over 256 families against ~50 per
family cold. Never breaks even.

## Distribution boundary, measured

Hand-made families: line 0.958, dial 0.373, perm 0.255, toggle 0.043. The
generator has no paired-flip op and builds only product state spaces, so
`toggle` and `perm` are outside its SUPPORT rather than unseen instances of it.

## Next

Binding constraint is entry EXPRESSIVITY, not reading. (a) let the entry
modulate the plant's computation (per-family gains/biases) instead of only
prepending tokens; (b) widen the generator's support. Predictions recorded in
advance in MEMORY_BANK_DESIGN.md, including the falsifier: if (a) degrades
retention, any channel wide enough to be expressive is wide enough to
interfere, and that is a real wall.
