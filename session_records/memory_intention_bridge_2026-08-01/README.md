# Frozen-controller memory-intention bridge — 2026-08-01

## Question

Can a frozen controller acquire a new context-private relation rule through a
generic learned memory-code bridge, rather than storing protocol action
prototypes directly?

## Scope and mechanism

The controller, vision/relation representation, and output protocol remain
frozen during adaptation. A small bridge is trained once on support episodes,
then frozen. It maps a controller-created support memory value to a two-state
latent code; a separate composer combines that code with the frozen query
intention and emits the opaque action protocol. At adaptation time, only
content-addressed RAM/disk rows are mutable.

Training uses no hand labels or private query answers. The only learner-visible
signals are the attempted opaque support action, its scalar verifier
success/failure, and the frozen controller's own query action. The verifier's
true query target is retained only for held-out audits.

To prevent a visual shortcut, each context receives a unique random RGB patch
used as its memory key. The patch is independent of the hidden relation-to-
action remapping, so recognizing the key cannot reveal the answer rule.

## Three independent runs

Each run used 256 support contexts (256 scalar verifier bits), 200 bridge
updates, 2,048 held-out contexts, real disk save/reload, and MPS. The frozen
controller digest was identical before and after every run.

| seed | disk | reversed | prediction flips | no memory | shuffled | corrupted |
|---:|---:|---:|---:|---:|---:|---:|
| 29001 | 100.00% | 100.00% | 100.00% | 49.95% | 50.98% | 49.27% |
| 29002 | 100.00% | 100.00% | 100.00% | 50.00% | 49.12% | 50.39% |
| 29003 | 99.95% | 99.95% | 100.00% | 50.00% | 50.15% | 50.68% |

All ten pre-registered gates passed for all three seeds, including exact
content-addressed retrieval and freezing of both bridge and composer during
adaptation. The reports and promoted bridge checkpoints are kept beside this
README.

## Interpretation

This is a stronger result than the non-parametric action-cache Gate 1: a
learned, two-dimensional latent memory code can be translated into an opaque
action intention while the controller itself remains immutable. It is a
sample-efficient proof that external memory plus a small reusable adapter can
support new context-conditioned behavior without weight updates.

It is not yet the final native-memory claim. The experiment uses a fixed visual
key encoder and an external bridge/composer; it does not make the controller's
native `retrieved_memory` hidden vector path learn the rule by itself. The next
test is to replace this fixed key/bridge boundary incrementally with the
standardized amodal event and intention buses, preserving the same controls and
the zero-weight-update invariant.
