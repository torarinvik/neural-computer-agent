# Learned transition-evidence admission

This pressure test makes the online context boundary tolerant to noisy learned
representations. An external evidence evaluator is trained from deterministic
scalar consistency outcomes. During the frozen-controller target phase it
accepts small next-state noise as reuse, keeps reuse read-only so noisy values
cannot overwrite mastered facts, and admits genuinely contradictory evidence
under a new address.

```text
.venv/bin/python experiments/external_learned_evidence_admission/train.py \
  --seed 69701 \
  --report-out /tmp/external-learned-evidence-admission.json
```

This remains a bounded learned verifier over opaque transition tensors. It
does not yet learn context from raw modalities, adapt thresholds online, or
compress unbounded external memory.
