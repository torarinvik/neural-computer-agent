# Deeper n-back growth with a domain-general event window

This promotion pressure-tests whether the outcome-only external-compute
mechanism scales to harder working-memory depths without changing the frozen
controller or adding n-back-specific reasoning code.

## Change

The private verifier now generates `nbackN` targets generically from the
family depth. The external event-window size is a versioned construction
parameter. The learner still receives only rendered symbol events, opaque
keypresses, and scalar verifier outcomes; the family name, depth, target bit,
and correct action remain private.

The four-event window supports n-back-3. A direct capacity probe shows that it
does not support n-back-4 reliably. Widening the same generic window to five
events makes n-back-4 learnable without a new reasoning branch. The full
five-event promotion also assigns distinct rendered cue symbols to every
file, avoiding an invalid same-key route collision.

## Results

| Protocol | Seed | Files admitted | Minimum direct accuracy | Minimum routed accuracy | Shuffled-control maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-event window, through n-back-3 | 17 | 7 / 7 | 0.8636 | 1.0000 | 0.4479 |
| Five-event window, through n-back-4 | 17 | 8 / 8 | 1.0000 | 1.0000 | 0.2760 |
| Five-event window, through n-back-4 | 18 | 8 / 8 | 1.0000 | 1.0000 | 0.4063 |

All promoted runs passed direct stable mastery, route selection, same-context
reversal, old-file retention, route reload, shuffled-feedback rejection,
frozen-controller/frontend, unchanged admitted files, and zero-replay gates.
The n-back-3 file reached 1.0000 on every fresh direct lifetime in the
four-event run. The n-back-4 file reached 1.0000 on every fresh direct lifetime
in both five-event runs.

The four-event n-back-4 capacity probe was intentionally not promoted: after
192 attempted-outcome updates, seed 17 evaluated at
`[0.7781, 0.7531, 0.7594, 0.7250]` and seed 18 at
`[0.7531, 0.7594, 0.7250, 0.7906]`. This is the expected information-window
boundary, not evidence for a missing n-back-specific learner branch.

Primary accounting was kept separate from controls. The five-event runs each
used 1,279,488 unique primary verifier bits, 30,464 audit bits, 107,136
primary logical lifetimes, 1,536 primary optimizer updates, 1,896 route-memory
updates, 73,728 matched-control bits, 6,144 control logical lifetimes, 192
control optimizer updates, and zero replayed examples. The four-event run used
1,136,128 primary bits, 28,288 audit bits, 92,800 primary logical lifetimes,
1,344 optimizer updates, 1,628 route-memory updates, and zero replay.

## Claim boundary

This promotes replicated deeper working-memory acquisition and shows that a
small, domain-general representation change extends the learned external
compute boundary from n-back-3 to n-back-4. It does not establish unrestricted
history, learned compression, arbitrary program induction, or general
continual learning. The next bottleneck is replacing the bounded event window
with a scalable external history/memory contract while preserving
information, route identity, and no-replay retention under capacity pressure.

Raw reports are `window4-seed17.json`, `window5-seed17.json`, and
`window5-seed18.json`; the non-promoted capacity audit is in
`window4-nback4-capacity-probe.md`.
