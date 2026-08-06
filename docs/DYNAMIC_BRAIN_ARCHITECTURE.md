# The Dynamic Brain: canonical vision-to-architecture mapping

This document is the authoritative statement of what this repository is
building and where every kind of knowledge is allowed to live. When any
experiment, doc, or result appears to conflict with this document, this
document wins, and the conflict must be recorded as a transitional
violation (see "Current violations" below).

## The problem being solved

Ordinary models have a fixed parameter count and therefore a fixed capacity,
like a brain of fixed size. Everything they know is burned into weights, so
knowledge cannot grow without growing (and retraining) the model, and new
learning competes destructively with old learning inside the same finite
weight budget.

This architecture solves that with two decouplings:

1. **I/O decoupling.** Encoders and decoders detach the brain from any
   particular input or output. The controller consumes only opaque event
   tensors and emits only opaque intentions. Any modality, game, sensor, or
   actuator can be attached without touching the brain.
2. **Knowledge decoupling.** Skills and information detach from the brain's
   weights. They live in an external memory bank that can grow without
   bound, be combined into new abilities, and be audited, protected, or
   evicted item by item. The brain does not have to *contain* what it
   knows — only to *operate* it.

## The component roles (the CPU analogy)

| Role | Component | Size discipline | What it may store |
| --- | --- | --- | --- |
| CPU / frontal lobe | `AmodalCognitiveController` | **Fixed forever** | Only the general ability to compute, learn, address, and combine. Never a specific skill or fact. |
| Peripheral drivers | encoders / decoders | Fixed per peripheral | Only I/O translation for their device (board → events, intention → keypresses). Never strategy or facts. |
| Sketchpad / RAM | controller recurrent state, workspace slots, event window | Fixed width | Transient working contents: current events, fetched skills and info in use. Cleared without loss — everything durable lives in the bank. |
| Long-term memory bank / disk | `ContentAddressedMemory`, artifact memory, external capability slots + retention ledger | **Grows without bound** | All skills and all information. Content-addressed: fetched by similarity of the current situation, never by task label. |
| Fetch path | growth/candidate routers | Grows with the bank | The learned mapping from opaque situations to bank entries. |

The one sentence that must never be violated in a promoted claim:

> **Skills and information are stored in the memory bank. They are not
> burned into the weights of the controller, the encoders, or the
> decoders.**

Whether the bank, the sketchpad, or the fetch path is implemented with
neural modules, tensors on disk, or anything else is an implementation
choice — use whatever measures best. The non-negotiable property is that
the brain's knowledge is **dynamic and unbounded**, while every computing
component stays fixed-size.

## What each discipline buys

- **Fixed CPU**: catastrophic forgetting stops being a capacity war. The
  controller's weights only hold meta-ability (how to learn, fetch,
  combine), so protecting them (e.g. Fisher consolidation) protects one
  thing, not an ever-growing pile of skills.
- **Unbounded bank**: knowledge never competes for space with other
  knowledge. A full bank grows instead of silently forgetting (the
  promoted grow-when-full / protected-eviction results).
- **Content addressing**: transfer becomes fetch. If two games contain
  maze-like situations, the maze events address the same bank entries; the
  second game reuses the first game's navigation fragments because they
  *match*, not because anything was labeled "maze."
- **Item-wise audit**: bank entries can be individually verified,
  protected, consolidated, or evicted — impossible for knowledge smeared
  across weights.

## Current violations (transitional, to be retired)

Honesty requires stating where today's promoted results still break the
rule. As of 2026-08-06:

1. **Game skills live in core weights.** The promoted two-game EWC rung
   (`ewc_consolidation_plastic_core_v1`) stores Snake and Pong ability in
   the controller's weights, protected by Fisher consolidation. This
   proved the core can learn continually without replay — but it is a
   violation of the storage rule, kept as scaffolding until skill
   externalization lands.
2. **Peripheral weights carry some skill.** The per-game frontends and
   decoders were trained jointly with play, so strategy fragments likely
   leak into them. Externalization must squeeze skills out of the
   peripherals as well: they should converge toward pure format
   translation.
3. **No memory bank is wired into the games runtime.** `memory=None` in
   the games rungs. The bank machinery is promoted in the parent line but
   not yet operating on games.

The skill-externalization rung (below) is the designed retirement of all
three.

## The skill-externalization rung (design)

**Goal.** Move acquired game skill out of weights and into fetchable bank
artifacts, then demonstrate compositional transfer: fragments fetched from
earlier games make a later, overlapping game cheaper to acquire.

**Setup.** The shared-controller games runtime with a memory bank attached.
Skills become *artifacts*: opaque tensors written to the bank through the
existing consolidation/verification gates, addressed by content.

**Phase 1 — externalize.** Acquire Snake as today, then consolidate the
skill into bank artifacts and *reset or regress the core toward its
pre-acquisition state*. Gate: with the artifact fetched into the sketchpad,
play recovers to mastery; with the bank withheld, play collapses to
chance. That double dissociation is the proof that the skill is in the
bank, not the weights. Repeat for Pong.

**Phase 2 — fetch is causal and content-addressed.** Gates: fetch accuracy
from opaque events alone (no labels); a shuffled-bank null (fetching the
wrong artifact must not work); permutation invariance of the bank; the
retention ledger protects both artifacts; a full bank grows rather than
evicts.

**Phase 3 — compositional transfer.** Introduce a third game sharing
structure with an earlier one (e.g. maze-navigation shared between two
maze games, or ball-interception shared with Pong). Train it once with the
bank available and once with the bank withheld, matched budgets. Gate:
the bank-available acquisition curve is steeper (fewer updates to
threshold), and a fetch audit shows earlier-game artifacts were actually
fetched during the improvement (causality: withholding just those
artifacts removes the speedup). This is the transfer-learning claim in
your vision, measured causally.

**Standing gates for every phase:** two seeds, reward-shuffled nulls, zero
replay, fixed controller/peripheral parameter counts (any growth must be
in the bank), and the storage-rule audit: after externalization, core and
peripheral weights alone must not suffice to play.

**Claim boundary discipline.** Until Phase 1's double dissociation passes,
no result may claim skills live in the bank. Until Phase 3's causal fetch
audit passes, no result may claim transfer via reuse.
