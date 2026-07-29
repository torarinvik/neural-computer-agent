# Diversity-preserving memory consolidation

## Question

Does collapsing every behaviorally equivalent memory to one prototype discard
variation that becomes useful under a future distribution shift? Can a small
within-class diversity reserve recover transfer accuracy while retaining most
of the compression?

## Why this rung

The previous milestone compressed 16 independently acquired memories to one
row for each of two hidden behaviors. That reached about 99% on held-out
versions of the trained bar-shaped inputs. A zero-shot probe then changed only
the object geometry:

- one representative per behavior retained 99.3–99.6% on bars;
- it retained about 99.1% on unseen diamonds;
- it fell to about 97.1% on disconnected dot pairs;
- the uncompressed bank still reached about 99.5% on dot pairs.

The stored skills and action path therefore remained valid. The missing
ingredient was representative diversity under a stronger appearance shift.
A preliminary jump to four-rule memories was rejected: although the relation
carried partial signal, independently stored four-rule values executed the
same full mapping only 36.5% of the time. That would have confounded memory
writing with consolidation.

## Mechanism

The online consolidator still uses only the learned pairwise relation between
controller-created latent memories. It receives no task name, rule bit,
appearance name, or target memory action.

For each discovered equivalence class it now reserves a bounded number of
representatives:

1. a novel relation occupies a free row;
2. an equivalent memory also occupies a free row until the class reaches its
   diversity allowance;
3. later equivalent memories merge into their closest representative;
4. genuinely novel memories still replace the least-used row when full.

The promoted operating point keeps two representatives per behavior:
four rows for the two-skill stream. No parameter is trained or changed.

## Formal zero-shot results

Both runs start from separate accepted 64-bit consolidation checkpoints.
Each audit contains 4,096 streams of 16 bar memories. Query geometry changes
only at evaluation.

| bank | rows | seed 20621 dot pairs | seed 20622 dot pairs |
|---|---:|---:|---:|
| first four memories | 4 | 91.99% | 92.13% |
| one representative / behavior | 2 | 97.17% | 97.47% |
| **two representatives / behavior** | **4** | **98.36%** | **98.57%** |
| three representatives / behavior | 6 | 98.96% | 99.05% |
| uncompressed | 16 | 99.50% | 99.52% |

At the promoted four-row operating point:

- bars remained 100%;
- unseen diamonds reached 99.72% and 99.69%;
- both behaviors were present in 100% of banks;
- no new training verifier bits or optimizer updates were used;
- binary-mapping and four-rule retention gates passed;
- every model tensor remained bit-identical.

This is a 4× logical-row reduction. Serialized bytes were 41.56% of the
16-row files, a 2.41× file reduction, because fixed metadata dominates these
small banks.

## Adversarial and physical audits

Across 2,048 physical capacity-four banks:

- every save/reload was tensor-exact;
- dot-pair accuracy was 98.88% and 98.05%;
- both behaviors were retained in every bank.

Zeroing the stored values reduced dot-pair behavior to 49.50% and 50.60%.
The same-capacity first-four control stayed near 92%, showing that capacity
alone is insufficient.

A valid counterfactual kept every bank and RGB frame fixed while reversing
only the verifier rule. Across 4,096 lifetimes per seed:

- ordinary accuracy was 98.46% and 98.54%;
- reversed accuracy was 98.49% and 98.46%;
- selected physical rows flipped in 98.07% and 98.36%;
- fresh feedback-derived probes changed in 100%.

## Conclusion

Behavioral equivalence is not the same as universal interchangeability. One
prototype was enough inside the original distribution but erased benign
appearance variation that became useful on disconnected objects. Reserving
two examples per learned skill recovered 1.19–1.46 points of zero-shot
dot-pair accuracy and beat naive same-capacity storage by more than six
points, while retaining a 4× logical compression.

The new frontier is to make the diversity allowance itself adaptive. The
controller should spend extra rows only when verifier history shows that a
class's representatives are not interchangeable under the current stream,
with accuracy sovereign and row/read cost secondary.
