# Factorized external register composition: mechanistic signal

Date: 2026-08-07  
Status: positive mechanistic signal; not promoted  
Schema: `neural-computer.external-register-factorized-composition.v1`

The default register interpreter now applies a shared low-rank factorized
register update. This follow-up used the same two-bit opaque event-codebook
pressure test as the rejected nonlinear-transition diagnostic. Instruction
zero learned complement-first-bit and instruction one learned
complement-second-bit with separate capability decoders. The interpreter and
instruction data were then frozen; a fresh decoder learned only the serial
composition from the final register.

Both primitive decoders reached `1.0000`, the old primitive decoder remained
at `1.0000` after adding the second instruction, and the fresh composition
decoder reached `1.0000`. This is evidence that the factorized register path
can preserve a reusable intermediate when output decoders are genuinely
external. It is not yet a capability gain or a promotion: the event source is
a fixed random codebook rather than valid rendered Brain Workshop events, and
the run did not include fresh/reward-shuffled/missing-evidence/reload or
memory-corruption controls.

The next audit must use the public `step_register` interface on rendered
amodal events, retain the old decoder and old primitive behavior, and compare
against both the current whole-program bank and a matched fresh learner.
