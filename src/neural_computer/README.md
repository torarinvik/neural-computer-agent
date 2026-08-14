# `neural_computer`

The production package implements the modality-independent machine. It must not
import experiment code.

## Main boundaries

- `interface.py`: typed amodal events, intentions, feedback, and collections.
- `runtime.py` / `controller.py`: variable N-to-M execution and fixed shared
  controller state.
- `memory.py`, `temporal_memory.py`, `temporal_index.py`: external factual and
  temporal storage with fail-closed reads.
- `artifact_memory.py`, `state_store.py`, `retention.py`, `lifecycle.py`:
  integrity, persistence, protection, admission, eviction, and replacement.
- `program.py`, `recipe_*`, `control_flow*`, `register.py`: verified external
  operators, programs, and composition.
- `executive.py`: persistent typed workspace and the minimal external
  `RECEIVE` / `READ` / `WRITE` / `COPY` / `CALL` / three-way `BRANCH` /
  `WAIT` / `EMIT` / `HALT` interpreter. Every `CALL` uses explicit,
  interface-versioned, transactionally replaced operator state; state is never
  hidden inside a shared operator object.
  The public `tick()` path remains fully defensive. Live internal execution may
  use an owner-bound, non-serializable sealed state lease that keeps the same
  event and operator-result validation while avoiding repeated full state scans.
- `executive_memory.py`: generic stateful executive operators, beginning with
  an opaque positive-relative value delay. Missing temporal history remains
  absent rather than becoming zero evidence.
- `executive_operators.py`: allow-listed generic singleton-event, equality,
  and binary-intention operators for persisted programs.
- `executive_bank.py`: self-contained instruction/operator artifacts and
  append-only, checksum-protected `.bank` admission and reload. Operator
  manifests select only built-in constructors and cannot import arbitrary code;
  `compose_executive_artifacts` safely rebases slots, handles, and control-flow
  targets across admitted files.
- `agent_brain_bank.py`: the canonical heterogeneous `AgentBrain.bank`
  container. It combines executive artifacts with legacy temporal route banks,
  preserves each family's ABI and evidence, encodes tensors without pickle in
  the new JSON format, and requires explicit validated migration for old torch
  banks. It also searches opaque ordered parent pairs, requires reachable
  sequential handoff, verifier-gates composed children with staged fresh-bit
  accounting, caches immutable execution metadata, and recomputes
  parent-to-child provenance during reload.
- `executive_route.py`: opaque context-to-slot selection over admitted
  executive artifacts. Route evidence is persisted with the `.bank`, records
  exact behavior propensities, and can use lifetime-aggregate outcomes so a
  lucky action streak cannot promote the wrong skill.
- `episodic.py`: working-memory and episodic external computation.
- `world_model.py`, `online_transition.py`, `factored_transition.py`: factual
  transition learning and model-based execution.
- `keypress.py`: one replaceable opaque protocol boundary.
- `live.py`: variable-port `INPUT`, queued reward adapters, monotonic cognitive
  ticks, authenticated action receipts, exact-once outcome resolution, device
  dispatch, latency accounting, and `ExternalExecutiveLiveMachine`, which runs
  a reloaded external skill through the sealed executive into a replaceable
  intention decoder.
- `human_io.py`: public-screen capture, evidence-bound outcomes, visible pulse
  segmentation, allow-listed macOS windows, and replaceable key transport.
- `credit.py`, `plasticity.py`, `promotion.py`: trainer-only outcome credit,
  protected updates, and evidence gates.

The broad top-level export surface in `__init__.py` is retained for the current
canonical experiments. New work should prefer direct module imports so obsolete
compatibility exports can be retired without another repository-wide migration.

## Architectural rule

The package provides general computation and storage mechanisms. Durable
task-specific content belongs in external artifacts. Reward is received as a
trusted input event; it is not an opcode. Boolean logic is a library-level
operator, while the kernel is responsible for typed I/O, workspace access,
program calls, and control flow.

`LiveInputInstruction` preserves a fixed event ABI while accepting a variable
number of sensory and verifier devices. `QueuedOutcomeInputDevice` lets an
environment provide reward with an emitted action receipt and scalar evidence;
`TemporalProgramOutcomeObserver` routes that input to the selected external
program slot without mutating its instruction tensor.

`ExternalAmodalExecutive` is deliberately outside the neural controller. Its
operator handles carry independently versioned interfaces, quiet input remains
absent, workspace writes are detached copy-on-write values, and incompatible
types or divergent batched branches fail closed. The v1 interpreter has one
shared instruction pointer; physical/live execution is batch one. Autonomous
program proposal, durable admission, and controller-selected execution remain
separate lifecycle work rather than hidden interpreter behavior.
The sealed fast path is an internal runtime optimization only: leases are bound
to their creating executive, reject structural corruption at each tick, cannot
be serialized, and do not replace the defensive public restoration path.

`ExternalExecutiveLiveMachine` is the explicit bridge from a verified external
program artifact to `CognitiveTickRuntime`. It does not train the executive or
decoder (the decoder is frozen by default), and it routes delayed outcomes
through the ordinary receipt/observer boundary. A synchronous device may retain
an input event while the program is in `WAIT`; the input is not discarded or
silently replaced with a fabricated zero event.

`ExternalExecutiveRouterLiveMachine` extends that bridge to a bank of frozen
skills. A replaceable context encoder maps learned event tensors to an opaque
route key at episode start; no task ID or semantic rule field crosses the
boundary. Action-level outcomes remain ordinary resolved live events, while the
default route ledger commits one mean verifier outcome per lifetime through
`finish_episode()`. `per_outcome` feedback is available when an environment's
verifier contract explicitly calls for it. For aggregate feedback, the default
reversal threshold is the configured mastery threshold, so a partially correct
skill cannot remain protected merely because it beats chance; environments may
still provide an explicit stricter or looser reversal threshold.
