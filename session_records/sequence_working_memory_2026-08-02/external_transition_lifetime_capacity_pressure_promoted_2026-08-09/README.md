# Bank-owned lifetime telemetry under capacity pressure — promoted

This two-seed audit puts the verifier-trained lifetime policy on the real
bounded external transition-model bank. Each fresh three-slot bank contains
three independently trained affine factual transition models, and the bank
itself records usage, logical age, and held-out prediction-error telemetry.
The policy reads that persisted telemetry through the bank boundary; the
fixture does not pass synthetic feature tensors to the policy.

A hidden verifier derives a generic disposal rule from the same opaque keys
and bank-owned telemetry, then checks retained models on held-out transition
rows after copy-on-write eviction. Matched random and recency controls use the
same fresh banks and verifier. Both seeds pass:

- learned held-out selection: `0.575` and `0.555`;
- random control: `0.340` and `0.280`;
- recency control: `0.015` and `0.020`;
- held-out retained-model behavior, protected-slot, stable-address, and exact
  policy persistence gates;
- zero controller updates and zero replayed transition examples.

This promotes bank-owned lifetime telemetry and verifier-gated selection under
bounded capacity pressure. It does not promote unrestricted growth, learned
verifier replacement, consolidation/compression selection, or general
continual learning. The verifier remains authoritative and the policy remains
an independently replaceable external memory-management component.
