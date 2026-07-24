# Real-time Syllogimous contract v1

This experiment is an attributed, noncommercial research adaptation inspired by
[Syllogimous v3](https://github.com/4skinskywalker/Syllogimous-v3), licensed
CC BY-NC 3.0. The original project is not endorsed by this experiment.

## Agent boundary

An agent may receive only:

- timestamped RGB frames already presented by the renderer;
- timestamped mono PCM samples already played by the renderer;
- its own recurrent state.

It may emit only the textual action vocabulary `WAIT`, `NEXT`, `PREVIOUS`,
`TRUE`, `FALSE`, `MOVE_CARD source destination`, or `SUBMIT_SORT`. The evaluator
parses that output; the model has no callable game API. It never receives premise
strings, parsed relations, the correct answer, exact remaining milliseconds, DOM
state, OCR, generator state, question type metadata, seed, difficulty, or reward
internals. Information visibly printed or audibly presented by the renderer is,
of course, observable through raw pixels or PCM.

The evaluator alone owns logical state and reward. Renderer plans and evaluator
files must live outside the model process and must never enter a prompt or model
input tensor.

## Causality and time

- The episode clock begins when the first premise becomes visible.
- Video is emitted at 30 or 60 FPS; audio is continuous 16 kHz PCM.
- Observations contain no future frames or samples.
- `NEXT` changes the visible card but does not pause time.
- `TRUE` and `FALSE` are rejected until the conclusion is visible.
- Inference, routing, sensory encoding, expert switching, and action transport
  consume real wall-clock time.
- Deadlines, response times, and reward inputs are integer milliseconds. The
  visible timer is computed from actual elapsed milliseconds rather than a
  one-second countdown tick.
- An action completed after the deadline is a timeout.
- Training may run many environments concurrently, but no individual environment
  may pause its clock for inference.

## Added auditory modality

The upstream browser game is visual-only. This experiment adds a deterministic,
documented dual-tone vocabulary so an audio-only streamer can learn the same
public card facts. Card kind, entities, relations and their public parameters,
operators, sorting attributes, fixed-puzzle text, and feedback are encoded. The
audio renderer accepts only the answer-free `RenderCard` ADT. It has no function
that accepts private `Question`, `EpisodeTask`, truth, or expected sorting order.

This tone vocabulary is part of the rendered environment, analogous to an
accessibility code learned by a human player; it is not a hidden side channel to
the model. Silence suppression belongs to the learned streamer after PCM capture,
not to the game renderer, so the streamer cannot query semantic game state.

## Reward

For deadline `D`, correct completion time `t`, and `0 <= t <= D`:

```text
correct = 1 + 0.05 * (D - t) / D
wrong   = -1
timeout = -1
```

The speed bonus exists only for a correct answer. Router rewards may additionally
subtract small measured compute, sensory-byte/token, switching, and lateness costs.
Correctness remains the dominant term.

## Split discipline

Development seeds are below 100000. Final-evaluation seeds are 100000 and above
and require an explicit final-evaluation flag. Symbol alphabets, surface layouts,
fonts, colors, relation combinations, and difficulty combinations are held out in
addition to RNG seeds.

## Difficulty tiers

The native selection config records `DifficultyTier` explicitly. The max host
profile uses 16 premises, depth-4 Boolean composition (287 worst-case visible
cards), four-key sorting, negation, Stroop, meta-relations, challenge families,
and a 10,000 ms deadline. The renderer's 1,023-card bound is enforced rather
than silently exceeded. Python tooling exposes matching intro/standard/hard/max
profiles for sweeps and reports the selected tier in every manifest.
