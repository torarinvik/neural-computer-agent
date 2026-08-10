# Promoted external intention repertoire

Three seeds (`85101`, `85102`, `85103`) pass the candidate-formation pressure
test. A four-entry opaque experience stream is written once to an append-only
external repertoire. The policy-free runtime omits `candidate_intentions`
entirely; it retrieves the verified repertoire entries and sends them to
factual goal-conditioned search. An empty fresh repertoire uses only the
ephemeral controller seed as a safe fallback.

All promoted runs reach `1.0` mastery on four novel goals, while the matched
fresh empty-repertoire controls reach `0.0`; goal-shuffled controls reach
`0.0`, and random floors range from `0.0176` to `0.0273`. The controller and
factual model remain byte-stable during search, the original four vectors are
retained, persistence is exact, and replay/controller optimizer updates are
zero. Mean search latency is `3.28–4.90 ms` with `336` expansions per seed.

The safety result matters: an unverified controller seed is not mixed into a
verified repertoire by default because it can poison factual beam search. It
remains an explicit exploration option and the fallback when no verified
candidate exists.

This promotes external opaque candidate retrieval and safe exploration
plumbing only. It is not outcome-trained policy learning, arbitrary new
computation, unrestricted memory growth, or general continual learning.
