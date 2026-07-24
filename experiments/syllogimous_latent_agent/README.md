# Syllogimous latent agent

This experiment replaces autoregressive VLM control with a small recurrent latent
agent. At inference, the model accepts only RGB frames, PCM samples, and a padding
mask. Private premises, relations, answers, seeds, and game state are never model
inputs.

Three cores share the same learned sensory front end:

- `gru`: two-layer persistent recurrent memory.
- `graph`: causal self-attention over previously observed card embeddings.
- `recursive`: recurrent memory plus shared-weight latent refinement.

Training uses deterministic solver answers as labels. This is privileged
supervision, not privileged inference: the labels are passed only to the loss.

Run a smoke comparison:

```sh
python -m unittest experiments.syllogimous_latent_agent.test_latent_agent
python -m experiments.syllogimous_latent_agent.train \
  --core recursive --train-samples 1000 --eval-samples 100 --epochs 1 \
  --checkpoint latent_recursive.pt --report latent_recursive.json
```

The held-out evaluation changes the visible symbol prefix and includes chain
lengths absent from training. Reports include final-answer accuracy, all-action
accuracy, parameter count, and end-to-end milliseconds per episode.

## First H100 result

All models were trained on 2--6 premises with six training-only symbol prefixes
and evaluated on the reserved `Z` prefix. At trained lengths all three exceed
99% final-answer accuracy. Extrapolation is the unresolved problem: graph memory
retains 28.5% accuracy at 16 premises, while the GRU and recursive cores collapse.

The rejected shortcut report is retained deliberately. In the first generator,
the conclusion relation word revealed the answer; 99.8% accuracy from that run
is invalid evidence of reasoning. The balanced generator test now proves that
every relation phrase occurs with both answer values.

## Cached graph curriculum

The second H100 run expands the visual vocabulary to 128 entities, trains a
5.37M cached graph model on a length curriculum through 16 premises, and reserves
24, 32, and 64 premises for extrapolation. It reaches 99.4% across trained
lengths, but not algorithmic length generalization: a 6,000-episode audit scores
18.85%, 62.4%, and 52.45% respectively. The 24-premise failure is a strong
length-conditioned `FALSE` bias, not answer imbalance.

The incremental API encodes each card exactly once and caches its state. On an
H100, batch-one sensory-to-action latency stays between 1.38 and 1.41 ms per new
event from 16 through 64 premises. The next architectural test should remove
learned absolute position embeddings, which are trained only through position 17
and are a plausible source of the extrapolation cliff.

That controlled ablation has now been run. Removing positions changes unseen
24/32/64-premise accuracy from 18.85/62.4/52.45% to 22.1/53.85/48.9% over
6,000 episodes. Overall accuracy decreases from 44.57% to 41.62%. Absolute
positions therefore are not the principal cause of the cliff. The stronger
hypothesis is that the network learns length-conditioned heuristics instead of
an invariant relation-composition operation; the next model should explicitly
apply the same local edge-composition rule until convergence.

## Neural transitive closure

The 2.97M closure agent learns subject, relation, object, and conclusion markers
from RGB while keeping exact solver labels on the loss side only. Its discrete
predictions build a directed graph, and seven shared squaring/composition steps
compute transitive closure for up to 128 entities. Unstable answer gradients are
stopped at the parser boundary; perception retains direct auxiliary supervision.

In the matched marked-card comparison, the learned cached graph scores
65.35/76.95/57.95% at 24/32/64 premises. Neural closure scores
83.8/83.65/85.0%. The closure result is effectively length-invariant, while its
remaining errors track visual object recognition. Incremental batch-one latency
is about 1.04 ms per event for closure versus 1.36 ms for the learned graph.
