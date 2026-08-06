# Episodic context and causal credit

This experiment targets the current continual-learning bottleneck directly.
The memory-side learner receives only ordered learned event tokens, opaque
action vectors, scalar outcomes, and presence. It first learns a reusable
episode context with paired augmented views, then learns opaque artifact
addressing from common-random attempted-row outcomes. A newly appended
external route is trained from fresh outcomes only; the old router and
context encoder are frozen.

The verifier privately generates three temporal procedures whose single-event
statistics are identical but whose ordered event patterns differ. Candidate
artifact keys are random opaque vectors. No task ID, correct row, or semantic
operation name enters the encoder or router.

The report compares the recurrent context representation against a pooled-event
baseline, checks candidate permutation, trains an event-credit head from
paired write-intervention utilities, and audits two sequential fresh route
additions. Each appended capability receives isolated credit state; the shared
context encoder and earlier credit heads remain frozen. Old-route retention,
prior-extension attempts, reward-shuffled extension selection, and per-new-
artifact ablations are measured.

This is a bounded representation-and-credit result. It does not claim
unrestricted memory growth, arbitrary program induction, or general
continual learning.

The harness also supports sequential families through `--new-families`. The
four-step audit acquires families `2,3,4,5` after freezing the base context
and router:

```bash
PYTHONPATH=src uv run python -m experiments.episodic_context_credit_amodal.train \
  --new-families 2,3,4,5 --external-credit-updates 128 \
  --seed 69316 --report-out /tmp/episodic-credit-four-step.json
```

This is a promoted bounded four-step replay-free growth result when both
seeds pass all gates. The family list is explicit so each extension is
credited only against earlier extensions, never future inactive slots.

The extended pattern bank pressure-tests eight sequential additions:

```bash
PYTHONPATH=src uv run python -m experiments.episodic_context_credit_amodal.train \
  --new-families 2,3,4,5,6,7,8,9 \
  --context-updates 512 --credit-updates 256 --route-updates 512 \
  --external-credit-updates 128 --seed 69316 \
  --report-out /tmp/episodic-credit-eight-step.json
```

The short context budget is retained as a rejected control: the longer
episode requires more frozen-context acquisition before external growth can
preserve the old route.

The pattern bank can also be generated for a longer episode:

```bash
PYTHONPATH=src uv run python -m experiments.episodic_context_credit_amodal.train \
  --episode-length 6 --new-families 2,3,4,5,6,7,8,9 \
  --context-updates 1024 --credit-updates 512 --route-updates 1024 \
  --external-credit-updates 128 --seed 69316 \
  --report-out /tmp/episodic-credit-generated-len6.json
```

This generated-bank audit passes both seeds and retains the under-budget
seed-unstable control as rejected evidence.

The same length-six sequence is now composed with the external retention
ledger and reversal transaction. The promoted protocol uses the ten opaque
capability keys across the old route and eight additions, requires a fully
protected bank to refuse eviction, applies four sustained low outcomes to
only the newest capability, and then verifies fresh re-protection after four
successful outcomes. Both seeds passed route, causal, isolated-credit,
zero-replay, retention, reversal, and recovery gates. This is a bounded
retention-safe growth result; learned consolidation, unrestricted memory
growth, and general continual learning remain open.

The next twelve-addition length-six pressure test is retained as a rejected
cross-seed control. Seed 69316 passes, but seed 69317 leaves the final
capability at `0.8125` fresh route selection, below stable initial protection;
the downstream fully-protected-bank and recovery gates consequently fail.
This localizes the next bottleneck to confidence/retention calibration as the
candidate bank grows. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_retention_rejected_v1_2026-08-06/`.

The intermediate ten-addition rung passes both seeds with the original
budgets. It is now the highest promoted generated length-six boundary: twelve
opaque capabilities retain their routes and isolated credit state, and the
fully protected bank still refuses eviction before the final capability is
reversed and recovered. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_ten_step_retention_v1_2026-08-06/`.

A twelve-addition control with every extension budget doubled to 256 updates
also failed cross-seed retention: seed 69317 fell to `0.75` on the final
route. This rejects longer per-extension training as a sufficient repair and
keeps route-interference/confidence calibration as the next target. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_ext256_rejected_v1_2026-08-06/`.

The final calibrated twelve-addition rung now passes both seeds. Each new
extension receives fresh positive outcomes for its own context and fresh
negative outcomes for already-acquired contexts, and its loss is aligned to
the `>1.0` activation boundary. This removes the earlier route-interference
and score-calibration failure without changing the retention gates. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_calibrated_v1_2026-08-06/`.

The calibrated full-bank follow-up fills all 20 generated pattern families:
two frozen old capabilities and 18 sequential additions. Across two seeds,
route selection remains at least `81.25%`, old-route and permutation controls
are perfect, reversal releases only the failing target, recovery re-protects
it, and replay remains zero. This is the current bounded ceiling, not
unbounded memory or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_full_bank_v1_2026-08-06/`.

The first temporal distribution-shift audit freezes length-six capabilities
and appends fresh length-seven capabilities without replay. Positive route,
credit, retention, and reversal gates survive, but one seed selects a shifted
family under reward-shuffled outcomes. It is rejected and retained as the
current credit-calibration control. See
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_episode6_to7_rejected_v1_2026-08-06/`.

The repair replaces finite-batch random shuffling in the control with
antithetic contradictory outcomes for each identical query. The corrected
length-six→length-seven audit passes both seeds: all old and new routes,
shifted credit, reversal/recovery, and the null control pass with zero replay.
See
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_episode6_to7_v2_2026-08-06/`.

Filling the full bank after that shift also passes both seeds: 18 new
length-seven capabilities are added to the two protected length-six ones,
with minimum route selection `92.19%`, old-route/permutation gates intact,
and zero replay. This is the strongest bounded stress result, not a claim of
unbounded or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_v1_2026-08-06/`.

The larger length-six→length-eight full-bank shift also passes both seeds,
with minimum new-route selection `82.81%`, perfect old-route/permutation
controls, and zero replay. The weakest route declines as the shift grows, so
repeated shifts remain the next test. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to8_v1_2026-08-06/`.

The length-six→length-ten stress rung is rejected: seed 69317 fails initial
protection for one shifted family even though routing and credit remain
strong. This is the current confidence-aware retention bottleneck; the
threshold is not being weakened. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to10_rejected_v1_2026-08-06/`.

The repair uses 256 updates per shifted extension and a zero-centered
antithetic null objective. The 6→10 full-bank rung then passes both seeds with
minimum route selection `84.38%`, full retention/reversal/recovery, and zero
replay. The 128-update rejection remains a useful acquisition-depth control.
See
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to10_v2_2026-08-06/`.

The repeated-shift harness now passes length six → eight → ten in one run:
eight capabilities arrive in the first shift and ten in the second, with
phase minima of `89.06%` and `89.06%` across the weaker seed, full-bank
protection, and zero replay. This is the current strongest bounded result;
dynamic expansion beyond the 20-family bank remains open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10_v1_2026-08-06/`.

The next rung passes length six → eight → ten → twelve in one frozen run:
eight, ten, and twelve capabilities arrive in three sequential shifts, growing
the bank to 32 capabilities. Phase minima across seeds are
`92.19%/89.06%`, `89.06%/90.63%`, and `92.19%/85.94%`. Old routes and
permutation are perfect; all-shift credit, causal acquisition, full-bank
protection, isolated reversal/recovery, antithetic null, and zero replay pass.
This is dynamic growth beyond the prior 20-family ceiling, not unbounded or
general continual learning: the generator remains finite and route/credit
acquisition remains externally trained. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_v1_2026-08-06/`.

The repeated-shift harness also supports a copy-on-write average prior for
external route adapters. At 256 updates per family, both seeds preserve every
hard route, causal-credit, retention, reversal, null-control, and zero-replay
gate across the 32-capability schedule. This promotes safe reuse of external
growth state, but not a reliable sample-efficiency gain: one seed improves a
late-shift floor while the other loses a small amount on the final shift. The matched
128-update control fails both seeds in the final phase, so late-shift
acquisition depth remains a real bottleneck. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_growth_prior_v1_2026-08-06/` and
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_growth_prior_ext128_rejected_v1_2026-08-06/`.

The shared growth-router rung replaces those capability-local extensions with
one permutation-equivariant candidate scorer per shift. Its learned query is
the concatenation of context, final recurrent state, mean recurrent state,
and max recurrent state; candidate keys remain random opaque vectors. Across
two seeds, the 6→8→10→12 schedule reaches phase floors
`0.9844/0.9375`, `0.9844/0.9688`, and `0.9531/0.9375` with 32 total
capabilities, zero replay, causal credit, retention, reversal, and null gates
all passing. The direct candidate permutation audit and corrected stricter
sequential operational route-permutation audit are exact to rounding at
`0.9906/0.9911`. The earlier `0.4932/0.4943` reading was a harness false
negative caused by comparing a remapped physical row to its unpermuted family
index. The 16,384-update shared-router budget is still high, so this promotes
reusable bounded growth rather than general continual learning or
sample-efficient transfer. Evidence is in
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_v1_2026-08-06/`.

The same shared router then passes 6→8→10→12 at 8,192 updates per expansion.
Across both seeds, phase floors are `0.9844/0.9844`, `0.9688/0.9063`, and
`0.9219/0.9063`; operational permutation is `0.9875/0.9802`, all causal,
retention, reversal, null, and zero-replay gates pass, and the bank still
reaches 32 capabilities. Total optimizer updates fall by 46.9% versus the
16,384-update rung. Copy-on-write router priors and prototype-address
controls did not replicate and are not promoted. Evidence is in
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_8192_v1_2026-08-06/`.
