# Immediate audiovisual reflex arena

This isolated experiment tests event-driven sensory learning with one-step credit
assignment. Every balanced trial presents a target or hazard above, right, below, or
left of a central agent. Targets require moving toward the cue; hazards require the
opposite action. Trials rotate among visual-only, audio-only, and agreeing audiovisual
evidence.

The adapter receives only raw pixel sequences and mono PCM. Static lead-in duration,
cue onset, kind, direction, and visual theme vary. Correct and incorrect actions earn
+1 and -1 immediately; training uses their exact expected one-step reward. A very
small normalized emission cost favors efficiency without overpowering correctness.
SmolVLM2 remains completely frozen.

```sh
python -m unittest experiments.event_stream_reflex.test_reflex

python -m experiments.event_stream_reflex.train \
  --mode learned --local-files-only --device cuda
```

Dense, fixed-gate, and trainable-threshold modes share the same listener, dataset, and
optimization budget. Results are reported overall and separately for vision, audio,
audiovisual, target, and hazard trials.

The confirmed three-seed result is summarized in [RESULTS.md](RESULTS.md). Sparse
fixed and reward-trained gates preserve perfect reflex accuracy while discarding
about 90.3% of dense sensory tokens. End-to-end latency does not improve because
the frozen 500M listener dominates this implementation's runtime.
