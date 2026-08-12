# Multi-candidate fresh-verified compute reuse (2026-08-07)

The admission policy now trials a new binding against every existing physical
compute module. Each candidate receives an isolated adapter/state and fresh
outcome training; the highest worst-case probe score is selected only if it
clears the floor. Otherwise all trial bindings are discarded and a new
physical module is grown.

Three opaque procedures promote at both seeds:

- seed `69316`: procedure 2 rejects the only existing candidate, grows a
  second physical module, and procedure 3 selects candidate 0 after trying
  both modules; final behavior is `1.0000/0.9258/1.0000`;
- seed `69317`: both later procedures reuse the only physical module; final
  behavior is `1.0000/0.7695/1.0000`;
- all old bindings retain after reload, checksum recovery passes, the frozen
  controller is unchanged, and replay is zero.

The resulting physical-to-logical ratios are `2:3` and `1:3`, with payload
ratios `0.5936` and `0.5133` versus independent full programs. This is the
first candidate-selection result rather than first-slot reuse. It remains
bounded continual external memory; larger candidate banks, routing latency,
and open-ended learned compression remain to be tested.
