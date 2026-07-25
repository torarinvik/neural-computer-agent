# Unified cognitive controller

This isolated experiment begins the transition from separately trained
primitive modules to one controller that learns within its recurrent state and
external memory.

## Boundary

The learner receives only:

- rendered RGB frames;
- its own previous opaque action;
- the scalar outcome of that attempted action;
- its own latent RAM/VRAM workspace;
- later, controller-created rows retrieved from disk memory.

It never receives stimulus identities, hidden rule bits, correct actions,
unattempted-action labels, semantic task IDs, or symbolic solution steps.

## First rung

The smallest rung chooses one private correct opaque action for each lifetime.
Trial zero is necessarily chance; after one attempted action and outcome, the
answer is fully identifiable. The controller must retain that evidence across
later query trials without a weight update or further learner-visible feedback.
Visible stimuli vary but are irrelevant at this rung, preventing the first
experiment from conflating active-state learning with visual rule binding.

The second rung chooses a private bijection between two visible stimuli and two
opaque actions. The first pilot learned a predictable second-trial shortcut but
stayed near chance on later randomized trials. It was correctly rejected and
motivated the smaller constant-action atom. The corrected generator samples
every query identity independently, eliminating that positional shortcut.

Before binding the two atoms, a visible-identity rung trains the same controller
to ground the two rendered identities in opaque actions using only attempted
actions and outcomes. Acceptance uses true pixel-level stimulus reversal,
required prediction flips, and blank-vision collapse. The discarded diagnostic
probe established that identity is already perfectly decodable from the parent
vision features; this rung tests whether behavioral reward can route it.

Naive sequential visual training erased the constant-action memory atom. A
combined parent is therefore admitted only when balanced rehearsal preserves
the old blind causal gate while the new pixel-grounding gate passes:

```bash
python -m experiments.unified_cognitive_controller.train \
  --device cuda \
  --task visible_identity \
  --rehearsal-task constant_action \
  --rehearsal-every 2 \
  --retention-task constant_action \
  --checkpoint-in artifacts/checkpoints/unified_constant_action_v1_seed2501.pt \
  --steps 600 \
  --batch-size 256 \
  --trials 6 \
  --feedback-trials 1 \
  --report experiments/unified_cognitive_controller/reports/combined_atoms.json \
  --checkpoint-out artifacts/checkpoints/unified_combined_atoms_v1.pt
```

The separate constant and visible tasks are ambiguous when mixed without a task
identifier: one demonstration can be consistent with both while they demand
different next actions. The unified composition rung therefore samples all
four binary functions—constant zero, constant one, identity, and flipped. Two
feedback-bearing support events cover both visible identities, making the
private function uniquely inferable. Later queries carry no feedback. This is
human-solvable without a task label and directly tests memory × perception.

The same vision encoder, recurrent controller, generic workspace operations,
latent intention, and replaceable actuator adapter process every trial. The
long-term memory interface is serializable but deliberately inactive until
RAM-based few-shot behavior passes.

Acceptance requires:

- held-out post-feedback accuracy of at least 85%;
- the same under a true private-rule reversal with identical pixels;
- at least 80% paired prediction flips;
- at least a 15-point advantage over blank vision;
- near-chance zero-shot behavior;
- material degradation when workspace is reset;
- material degradation when outcome history is shuffled.

## Audited milestone

The gradual curriculum produced a repeatable forward-transfer result:

| 600-step four-rule initialization | Seed 2502 | Seed 2503 |
|---|---:|---:|
| mastered visual primitive | 99.90% | 99.85% |
| fresh controller | 75.05% | 74.85% |

The fresh controllers learned a feedback-only shortcut: constant rules can be
solved perfectly and identity-dependent rules guessed, giving a 75% ceiling.
Blank vision reproduced that ceiling. Primitive-initialized controllers learned
the full composition instead. On a 1,024-lifetime blind audit the selected
parent reached 99.85% normal and 99.90% reversed-rule query accuracy, 99.76%
paired query flips, 48.78% with shuffled outcomes, 55.66% with complete active
state resets, and 74.46% with blank vision.

A disposable supervised probe—not used by the agent—found visual identity
remained 100% decodable after compound training, versus 50.68% with shuffled
labels. This localized an apparent old-protocol regression to incompatible
behavioral prefixes rather than erased perception.

The next gradual rung restricts the function family to identity versus flipped,
so one attempted action and one scalar outcome uniquely identify the mapping.
Starting from the four-rule parent reached 100% query accuracy after 600 steps;
the matched fresh controller remained at 49.26%. The exact checkpoint repeated
at 100% on 1,024 blind lifetimes with 100% rule-counterfactual flips and chance
under shuffled feedback, active-state reset, or blank vision.

Training that one-support rung alone erased the broader four-rule behavior.
Balanced rehearsal fixed this without a task ID: bijection lifetimes retained
one support outcome while four-rule rehearsal lifetimes retained two. The
admitted checkpoint passed both 2,048-lifetime blind audits:

| Capability | Normal | Reversed | Paired flips |
|---|---:|---:|---:|
| one-support bijection | 99.98% | 99.95% | 99.93% |
| retained two-support four-rule | 100% | 100% | 100% |

The learner saw only RGB, its attempted opaque action, scalar outcome, and
latent active state. It received no task ID, rule bit, correct-action label,
unattempted-action label, or within-lifetime weight update. The promoted
checkpoint is
`artifacts/checkpoints/unified_compound_rehearsal_seed2505.pt`, SHA-256
`8e346fd2863f595fd0ee96b5f4a8353cae48e96e7739ad0febd11345a60f9099`.

This demonstrates fast within-lifetime binding, repeatable positive transfer,
and rehearsal-based behavioral retention inside one compact controller. It
does not yet demonstrate persistent disk-memory growth, transfer to a new
modality, or broad reasoning.

Run the sub-minute GPU experiment:

```bash
python -m experiments.unified_cognitive_controller.train \
  --device cuda \
  --task constant_action \
  --steps 600 \
  --batch-size 256 \
  --trials 6 \
  --report experiments/unified_cognitive_controller/reports/hidden_rule_smoke.json \
  --checkpoint-out artifacts/checkpoints/unified_hidden_rule_v1.pt
```

Only a passing run may save a checkpoint.

Blind-audit a saved parent without changing its weights:

```bash
python -m experiments.unified_cognitive_controller.audit_checkpoint \
  --device cuda \
  --task constant_action \
  --checkpoint artifacts/checkpoints/unified_constant_action_v1_seed2501.pt \
  --seed 2707 \
  --report experiments/unified_cognitive_controller/reports/constant_action_blind_seed2707.json
```
