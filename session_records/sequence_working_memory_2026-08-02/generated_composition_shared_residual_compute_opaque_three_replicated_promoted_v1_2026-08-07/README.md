# Three opaque procedures with append-only residual compute (2026-08-07)

This is the next scaling rung after the two-procedure residual-compute
promotion. The frozen controller acquires three runtime-generated opaque
procedures sequentially. Slot one trains the shared context basis; each later
slot trains only its compact local recurrent compute and opaque decoder. The
shared basis and all older slots are frozen before every addition.

Both seeds `69316` and `69317` promote. Reloaded behavior is:

- seed `69316`: `1.0000 / 0.8906 / 1.0000`;
- seed `69317`: `1.0000 / 0.9023 / 1.0000`.

Both runs pass fresh retention during growth, exact reload, all-slot digest
protection, shared-base immutability, deliberate corruption recovery,
frozen-core equality, and zero replay. The shared bank plus decoders use
`0.6740` of three independent full program-plus-decoder payloads.

This is verified bounded external compute growth, not general continual
learning. Each unrelated procedure still receives a local compute slot, and
the next bottleneck is capacity-aware reuse/compression of those local slots.
