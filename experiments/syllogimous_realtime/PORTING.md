# Syllogimous v3 Elisa parity inventory

Upstream reference: `../../../Syllogimous-v3-upstream` at
`01238d0b1a9b508257e6b5580063b1f76ad3eeb3` (CC BY-NC 3.0).

This is a behavioral port, not a DOM translation. Private truth and generator
state remain behind the evaluator boundary; an agent sees only rendered RGB and
PCM emitted in real time.

## Reasoning families

| Upstream generator | Elisa family | Required behavior |
|---|---|---|
| `createSameOpposite` | Distinction | same/opposite chains, negated wording, optional meta-relations |
| `createMoreLess` | Comparison | ordered magnitude chains, reversed wording, negation/meta |
| `createBeforeAfter` | Temporal | before/after chains, reversed wording, negation/meta |
| `createDirectionQuestion` | Direction2D | eight compass relations and coordinate-derived conclusion |
| `createDirectionQuestion3D` | Direction3D | 26 spatial relations |
| `createDirectionQuestion4D` | Direction4D | 3D spatial relation plus earlier/same/later time |
| `createSyllogism` | Categorical | valid moods/figures, invalid-rule sampling, distractor premises |
| `createSameDifferent` | Analogy | compare the truth values of two non-syllogism questions |
| `createBinaryQuestion` | Binary | AND/NAND/OR/NOR/XOR/XNOR over two question classes |
| `createNestedBinaryQuestion` | NestedBinary | recursively composed Boolean questions to configured depth |
| `generateCards` | Sorting | WCST-like multi-property lexicographic sorting |

## Current parity status

- Native generator/solver seed sweeps cover distinction, comparison, temporal,
  2D/3D/4D direction, categorical syllogisms, Boolean composition, and sorting.
- Boolean leaves now draw from all seven upstream base families and honor the
  six independently enabled operator flags.
- Sorting uses the exact upstream vocabularies and its width-selection behavior,
  including the original `minWidth=2` choice between two and three attributes.
- Negated surface forms preserve logical meaning for every relation family and
  categorical syllogisms and render with the configured negated ink in both
  carousel and compact display-all scenes.
- Analogy selects four unique entities, independently infers both relations from
  the shuffled premise graph, and preserves the upstream directional retry
  balancing between same-relation and different-relation cases.
- Meta-relations use an explicit `Relation.Meta` dependency rather than a hidden
  side field. Disjoint source/target pairs, surface negation, and solver
  resolution are covered by the native seed sweep.
- All twelve paradox IDs and twelve logic-puzzle IDs have typed rare-override
  selection and exact upstream premise/conclusion text tables. Paradoxes retain
  an explicit `Undetermined` answer and the second logic roll overrides the
  first paradox roll exactly as upstream. Checked text access uses an error
  union; the renderer consumes only cursor-validated line indices.
- The meaningful lexicon contains all 446 upstream nouns verbatim. The 1,530
  abstract consonant-vowel-consonant labels are decoded from their exact ordered
  Cartesian construction, excluding equal first/last consonants.
- Typed presentation state preserves persistent button ordering, persistent
  color inversion while Stroop is enabled, color reset when disabled, the
  independent red/white text Stroop flip, and structural negation-explainer
  detection without consulting private truth. When enabled, the explainer is a
  real first carousel card and synchronized audio event; display-all scenes show
  the same instruction.
- `syllogimous_public_view.elisa` is now the compulsory renderer boundary. Its
  ADTs contain no truth, answer, expected sorting order, or hidden sorting key.
  Inactive sorting attributes are removed rather than merely marked hidden.
- `syllogimous_render_stream.elisa` emits premise/conclusion cards for all six
  public question variants, including Boolean leaf traversal, analogy pairs,
  active sorting cards, and all fixed-puzzle line counts.
- Native RGB rasterization consumes the audited public stream for every family,
  including per-node Boolean frames, active sorting attributes, timer state,
  answer buttons, and wrapped fixed-puzzle text. The smoke suite currently emits
  42 causally ordered PPM frames, including terminal feedback.
- `public_question` and `next_render_card` form one exhaustive private-to-public
  route shared by every sensory renderer. The synchronized audio renderer uses a
  documented dual-tone vocabulary for card kind, entities, relations, operators,
  sorting attributes, fixed-puzzle text, and feedback. It is derived exclusively
  from `RenderCard`; no private truth or expected sorting key enters PCM.
- The original browser game has no audio output. Tone-coded PCM is therefore an
  experiment-specific accessibility/modality extension rather than an upstream
  behavior claim.
- Cross-round score, millisecond response history, submitted-answer kind, and
  right/wrong/timeout feedback are retained in a typed `GameLedger`. Unlike the
  browser's localStorage record, the compact ledger deliberately excludes private
  question objects and hidden evaluator state. Lifetime score and round totals
  are unbounded; the 1,024 public records form a ring buffer, so long training
  runs cannot fail when diagnostic history fills. Four-digit feedback fields
  saturate visually while the exact counters remain evaluator state.

## Cross-cutting mechanics

- meaningful noun vocabulary or abstract-symbol vocabulary;
- shuffled premise order and configurable premise count;
- negated spans and optional negation explainer;
- meta-relations that relate one premise relation to another;
- Stroop color inversion and randomized true/false colors;
- randomized true/false button positions;
- display-all and premise-by-premise carousel presentation;
- independent per-family real-time deadlines expressed in milliseconds;
- score, response-time history, right/wrong/timeout feedback;
- configurable Boolean operator set and nesting depth;
- analogy-only, binary-only, and sorting-only selection rules;
- rare paradox and logic-puzzle override questions;
- sorting cards with color, material, shape, size and ascending/descending keys.

Browser-only persistence, telemetry, dialogs, and decorative Metal Gear assets
are not model tasks. Their configuration semantics and visible feedback are
ported; network telemetry and `localStorage` are deliberately excluded from the
training runtime.

## Elisa architecture

- `syllogimous_types.elisa`: algebraic domain model and refinements.
- `syllogimous_generators.elisa`: deterministic family generators and solvers.
- `syllogimous_game.elisa`: `machine from` episode protocol and reward clock.
- `syllogimous_render_stream.elisa`: exhaustive answer-free card stream.
- `syllogimous_raster.elisa`: RGB rasterization only.
- `syllogimous_audio.elisa`: public-card tone vocabulary and `machine over` PCM synthesis.
- `syllogimous_sensory.elisa`: synchronized RGB/PCM/timestamp boundary.
- `syllogimous_ledger.elisa`: typed score, latency, and feedback history.
- `syllogimous_actions.elisa`: checked error-union parser for model-emitted action text.
- `syllogimous_driver.elisa`: monotonic episode coordinator and public scene ADT.
- `syllogimous_scene.elisa`: carousel, display-all, sorting, and feedback scene rendering.
- `syllogimous_runtime.elisa`: cross-round selection, feedback, and restart lifecycle.
- `syllogimous_host.elisa`: real wall-clock stdin/stdout sensory transport.

`machine from` owns protocol states such as presenting, awaiting an answer,
feedback and termination. `machine over` owns bounded stream consumers such as
premise traversal and expression evaluation. `when` is reserved for genuinely
order-independent truth tables; `match` handles algebraic variants. Expected
failures use error unions rather than sentinels.

The native `RealtimeEpisode` now owns carousel/display-all navigation, answer
eligibility, monotonic millisecond clock checks, terminal outcome/reward,
sorting-card swaps, and sort submission. `EpisodeTask` keeps expected truth or
expected card order private; `EpisodeObservation` exposes only cursor, visible
card count, conclusion visibility, and public phase. The clock-before-action
ordering is an explicit `machine from EpisodeDecision.CheckClock` transition.

The host feeds the parser through a bounded `u8&` action buffer rather than an
unbounded C-string cast. Newline framing, polling, and RGB/PCM writes are kept
outside evaluator state; a blocked stdout consumer therefore consumes real
episode time just like a slow model transport would.

`select_question_plan` ports the upstream class-selection rules without eagerly
exposing or serializing generated questions. It distinguishes the six analogy
base families from the seven binary base families, validates analogy/binary/
sorting-only modes with typed errors, selects Binary versus NestedBinary from
the configured depth, performs the two independent rare-puzzle rolls in their
original override order, and lets sorting compete after them with probability
`1 / (1 + ordinary_choice_count)`. Every plan carries the deadline of the
finally visible family; a rare fixed puzzle intentionally retains the ordinary
question's already-selected timer, matching the browser.

The upstream one-second countdown tick is not preserved. Deadlines and response
times use a monotonic millisecond clock; the timer bar is derived continuously
from `remaining_ms / deadline_ms`, and the small speed reward uses the exact
millisecond remainder. Rendering, streamer work, routing and inference all occur
inside that same clock interval.

`RealtimeDriver` is now the host-side coordinator. It reconstructs carousel
cards for forward/back navigation, exposes display-all and player-ordered sorting
as answer-free scene variants, parses model output, advances the episode clock,
records terminal feedback exactly once, and provides a `tick_driver` path for
continuous wall-clock polling. `observe_driver_sensory` is the only model-facing
observation entry point and emits pixels plus PCM/timestamp data, never the scene
or evaluator ADTs.

`GameRuntime` extends that contract across rounds. It retains RNG and selection
state, keeps terminal feedback visible for the upstream-compatible 1200 ms, then
generates the next private episode at the exact transition timestamp. During
feedback, model actions are rejected and observation remains audiovisual; no
question, truth value, expected ordering, or scene ADT crosses the model boundary.

The host envelope is 36 bytes of framing followed by raw RGB and PCM payloads;
the only scalar metadata are timestamp, dimensions, sample count, and sample
rate. Its generated hard profile enables every upstream family, with negation,
meta-relations, Stroop, all Boolean operators, and a 30-second deadline. The
host target is compiled and validated at `-O3`; the exact hard profile and typed
runtime also pass at `-O3` in isolation.
