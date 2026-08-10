# Outcome-only executable-file admission

This pressure test exercises the next CPU/files rung after the runtime seam.
The shared register interpreter and the frozen amodal controller are fixed.
Two opaque executable files are mastered first; a third file is held outside
the live bank and is admitted only after a stable run of scalar verifier
outcomes. A corrupted candidate is rejected without changing the bank, while
the accepted file is protected and used by a warm and a matched fresh route
learner.

The warm route is stored in a separate opaque external cell from the source
route. This preserves the source context after a prior single-policy append
experiment exposed route interference. Cell separation is memory-side state;
it does not add a controller branch or expose a program index to the learner.

The learner receives no relation, program index, task name, or verifier
target. It sees only opaque event tensors, sampled program choices, exact
propensities, and terminal scalar outcomes. The source route state is never
replayed during target acquisition, and the controller/interpreter remain
frozen.

Run a seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_file_admission/train.py \
  --seed 23001 \
  --report-out /tmp/external-program-file-admission-23001.json
```

This promotes, if all gates pass, bounded verifier-gated admission of a
portable executable file plus context-separated warm/fresh route comparison.
It does not prove
program synthesis, arbitrary new computation, unrestricted memory growth, or
general continual learning.
