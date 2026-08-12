# Remediated persistent growth to 100 capabilities (2026-08-07)

Status: promoted replicated bounded replay-free growth result.

The frozen episodic context encoder and base route survive seven sequential
temporal distribution shifts. The bank grows from two length-six capabilities
through new length `8, 10, 12, 14, 16, 18, 20` families, reaching 100 total
capabilities. Each new capability owns isolated external route and credit
state; the persistent route/credit state is reloaded and checksum-verified
after growth.

| gate | seed 69316 | seed 69317 |
| --- | :---: | :---: |
| minimum shift route selection | 0.8125 | 0.8594 |
| old route retained | pass | pass |
| causal new routes | pass | pass |
| reward-shuffled null | pass | pass |
| full-bank protection/reversal/recovery | pass | pass |
| route/credit reload and corruption rejection | pass | pass |
| replayed examples | 0 | 0 |

The acquisition policy remains selective. Fresh outcome probes identify weak
rows before admission, and remediation updates are spent only on those rows.
Seed 69316 required two remediation rounds; seed 69317 required two rounds as
well. Protected rows were not replayed or updated. The full bank refused
eviction, then released and recovered only the deliberately reversed target.

Accounting totals were `8,072,280` verifier bits, `1,195,560` logical
lifetimes, `73,216` optimizer updates, and `197.66` seconds for seed 69316;
`8,121,432` verifier bits, `1,220,136` logical lifetimes, `74,752` optimizer
updates, and `149.81` seconds for seed 69317. Both runs used zero replayed
examples and kept the controller frozen.

This promotes a 100-capability, seven-shift external-memory boundary. It
remains bounded generated growth: it does not establish unbounded memory,
arbitrary program induction, robust positive transfer against a fresh learner,
open-ended compression, or general continual learning.

The exact command was:

```text
PYTHONPATH=src uv run python -m experiments.episodic_context_credit_amodal.repeated_shift_growth \
  --seed {69316,69317} \
  --base-episode-length 6 \
  --shift-episode-lengths 8,10,12,14,16,18,20 \
  --families-per-shift 8,10,12,14,16,18,20 \
  --context-updates 1024 --credit-updates 512 --external-credit-updates 128 \
  --route-updates 8192 --extension-updates 256 \
  --remediation-updates 256 --remediation-rounds 4 \
  --remediation-probe-repetitions 8 --remediation-threshold 0.8 \
  --batch-size 16 --audit-batch-size 64 --growth-initialization fresh
```
