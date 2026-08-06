# Parent-conditioned external capability bank — 2026-08-06

Status: promoted narrow two-seed external-capability-bank result.

This audit freezes a span-2 parent controller and acquires two independent
memory-side recurrent programs from fresh rendered events, opaque actions, and
scalar verifier outcomes:

- `reverse4`
- `forward4`

Each artifact contains only the replaceable external recurrent context program
and its capability-local output decoder. The shared controller, frontend, and
parent output path remain unchanged. A learned opaque route selects one file;
the artifact bank verifies and reloads it before execution.

Across seeds `69316` and `69317`, all promotion gates passed:

- both programs reached stable-prefix mastery;
- selected accuracy was `0.895` / `0.973` and `0.934` / `0.961`;
- wrong-program controls fell to `0.566` / `0.535` and `0.539` / `0.559`;
- route accuracy, permutation accuracy, and reload route accuracy were all
  `1.000` on both seeds;
- reward-shuffled routing was `0.500` on both seeds;
- parent retention remained `1.000` for every selected and reloaded artifact;
- the frozen parent digest was unchanged, corruption was rejected, and replay
  count was zero.

The archived `artifact_bank_seed69317/` and `router_seed69317.pt` provide one
verified persistent payload. The two JSON reports contain the complete
histories, accounting, stable-prefix bits, controls, and exact gate results.

This promotes a reusable controller-as-CPU / memory-as-files boundary with
replaceable external computation. It does not establish general continual
learning, unbounded memory growth, arbitrary program induction, or broad
multimodal transfer. The next pressure test is sequentially appending more
than two capability files, with capacity pressure, eviction/consolidation,
and nonstationary route reversal while retaining earlier programs.
