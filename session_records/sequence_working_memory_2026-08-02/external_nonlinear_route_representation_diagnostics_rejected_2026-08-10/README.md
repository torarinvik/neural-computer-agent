# Nonlinear route-representation diagnostics (2026-08-10)

These are one-seed diagnostic controls for the route-identity bottleneck,
not promoted experiments. They compare source-only context pretraining,
direct context-key route features, and source/meta-pretrained recurrent
trajectory features against the same nonlinear open-world fixture.

| arm | route proposals matching factual winner | revisit identity | held-out acquisition |
| --- | ---: | ---: | ---: |
| source-pretrained trajectory feature | 2/6 | 0/6 | pass |
| source-pretrained context key | 1/6 | 0/6 | pass |
| meta-pretrained trajectory feature | 2/6 | 0/6 | pass |

All arms kept the controller frozen, retained no raw provisional rows, used
no old-regime replay, and restored the external router exactly. The factual
model's acquisition error was unchanged from the four-step control. Training
the route feature path and expanding the pretraining distribution therefore
did not solve route identity in this fixture.

The diagnostics are rejected as capability gains but retain two useful
infrastructure changes: route queries can explicitly consume the trained
context-key feature, and recurrent trajectory features can be included in
source-only pretraining. The next route experiment needs a verifier-grounded
factual signature or a stronger meta-training family; more cosine prototypes
are not sufficient.
