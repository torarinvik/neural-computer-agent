# The read path does not accumulate: one readable ancestor helps, a second hurts

The read path restored transfer at ancestry 3 → 4 (+0.0242, p = 2.5e-3, with a
capacity control and a retention audit). The compounding claim needs the
advantage to survive another step. It does not.

## What the two cells actually compare

This is the part that determines what the numbers mean.

At **3 → 4**, the shallow arm is the three-skill parent, which holds no skill
slots at all. Turning the read path on does nothing to it. So that cell compares
**one readable ancestor against none**.

At **4 → 5**, the shallow arm is the four-skill parent, which holds one slot —
and with the read path on it reads that slot too. So this cell compares **two
readable ancestors against one**. It is the marginal value of a second, not the
value of reading at all.

## Result

Ancestry 4 → 5, eight seeds, budgets 96/160/256, everything else identical.

| condition | pooled paired delta | sign test |
|---|---:|---|
| no read path | −0.0144 | 29W/54L, p = 0.008 (12 seeds, wider grid) |
| raw read | −0.0304 | 7W/17L, p = 0.064 |
| read compressed to 16 | −0.0199 | 10W/10L, p = 1.00 |
| read compressed to 32 | **−0.0503** | 4W/16L, **p = 0.012** |

Every variant is negative. **A second readable ancestor does not add to the
first; it subtracts.**

## The compression hypothesis was wrong

Three earlier observations suggested dilution: reading one prior slot (+64
inputs) helped, while reading two (+128) and reading the legacy pair (+128) both
hurt, the latter badly enough to drop absolute accuracy from 0.9899 to 0.8228. A
slot has only 64 hidden units, so a wide read plausibly swamped it.

Compressing the read to 16 or 32 dimensions tests that directly, and it fails.
Compression does not recover the advantage, and at 32 it is the worst arm
measured. Whatever goes wrong with a second ancestor is not input width.

## What this leaves standing

The 3 → 4 result is unaffected: reading one consolidated ancestor, where
previously none was visible, turns −0.0002 into +0.0242 at p = 2.5e-3, with the
capacity control at −0.0014 and every retention gate intact. The diagnosis
behind it — that exact-zero gating makes ancestry bit-identical to the next slot,
and that transfer and interference travel one channel — is unchanged and is what
the fix acts on.

What is now bounded is the reach of the fix. It buys the first step out of the
first-composition regime and not the second. A ladder built on it would gain
once and then stop, which is a smaller claim than compounding.

## Open

Why a second ancestor hurts is unexplained. It is not input width. Candidates
worth separating: the two readable slots may be mutually redundant, so the
second adds noise without information; the shallow arm at this cell already
reads one ancestor and may capture most of what is available; or the specific
task pair at 4 → 5 may be less related than the pair at 3 → 4, which the cue
separability probe could test directly.

The cleanest next measurement is a cell where the shallow arm reads nothing and
the deep arm reads two, which separates "a second ancestor" from "reading at
all" — the current design confounds them.
