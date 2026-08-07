# Four opaque procedures with append-only residual compute (2026-08-07)

The append-only residual-compute bank is tested through a fourth sequential
runtime-generated opaque procedure. Slot one trains the shared context basis;
each later slot trains only compact local recurrent compute and its opaque
decoder. The shared basis and all prior slots are frozen before each addition.

Both seeds promote. Reloaded behavior is:

- seed `69316`: `1.0000 / 0.8906 / 1.0000 / 0.9766`;
- seed `69317`: `1.0000 / 0.9023 / 1.0000 / 0.9453`.

All four slots retain fresh probe behavior during growth and after exact
reload. Shared-base and all old-slot digests remain unchanged. The external
memory checksum detects deliberate corruption and exact restoration recovers
the clean digest and behavior. The controller remains frozen and replay is
zero. The four-slot payload is `0.5999` of four independent full
program-plus-decoder payloads.

This is bounded continual external compute growth, not unrestricted memory
growth or general continual learning. The remaining bottleneck is scaling the
local compute bank while reusing/compressing procedures without sacrificing
fresh acquisition or retention.
