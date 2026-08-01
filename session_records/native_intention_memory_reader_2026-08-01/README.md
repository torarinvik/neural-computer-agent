# Native intention-bus memory reader — 2026-08-01

## The interface fork

The direct `retrieved_memory` channel was tested first with a learned 96-D
translator. It stayed at 56.4% held-out accuracy and about 15% reversal flips.
Even a disposable probe trained with private query labels stayed at the same
level. Retrieval was exact and the controller was unchanged, so the failure is
an interface-capacity result: this checkpoint does not turn an arbitrary
retrieved vector into a useful action through its existing hidden path.

The smallest architecture-aligned repair was a generic memory-to-intention
reader. It receives the frozen query intention and a retrieved generic memory
row, emits a 24-D intention residual, and then uses the unchanged frozen
protocol decoder. The reader is initialized as an exact no-op, trained once
from scalar support outcomes plus the controller's own query action, and frozen
before adaptation. Only serialized memory rows change during the audit.

## Replicated result

Each run used 256 support verifier bits, 2,048 held-out contexts, real disk
save/reload, and MPS. The controller digest was identical before and after.

| seed | disk | reversed | prediction flips | no memory | shuffled | corrupted |
|---:|---:|---:|---:|---:|---:|---:|
| 29201 | 99.56% | 99.90% | 99.66% | 50.00% | 51.81% | 51.07% |
| 29202 | 99.46% | 99.71% | 99.27% | 50.00% | 49.02% | 55.66% |
| 29203 | 99.95% | 100.00% | 99.95% | 50.00% | 50.39% | 56.40% |

All pre-registered gates passed for all three seeds. The corruption arm is
not required to fall exactly to chance; its required causal separation from
intact memory passed in every run.

## Interpretation

This closes the next interface question: a frozen controller can use learned
external memory to alter its **amodal intention** rather than merely retrieve a
protocol action. The result is still an adapter qualification, not evidence
that the controller's recurrent core internally learned a new weight-free
operator. The next frontier is to expose this reader through the standardized
runtime memory/intention bus and measure whether its inherited capability gives
a fresh learner a lower stable bits-to-threshold on a genuinely new primitive.
