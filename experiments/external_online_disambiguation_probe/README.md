# End-to-end online active probe resolution

This audit exercises the runtime seam end to end. An ambiguous opaque
evidence window requests a read-only probe from the online router; the caller
executes the returned intention in a hidden regime; the noisy consequence is
then submitted to the ordinary factual router. A random-intention control is
scored with the same factual resolution tolerance.

The default configuration uses eight candidate intentions, one informative
intention, two hidden regimes, and outcome noise with standard deviation
`0.1`. The controller remains frozen, the factual bank is not mutated during
queries, and router persistence is checked. This is a narrow integration
boundary, not a claim of learned multimodal probe selection or general
continual learning.

```text
uv run python experiments/external_online_disambiguation_probe/train.py \
  --seed 83201 \
  --report-out /tmp/external-online-disambiguation-probe.json
```
