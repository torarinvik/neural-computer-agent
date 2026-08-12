# Reusable physical compute with isolated capability bindings (2026-08-07)

The residual-compute bank is split into two memory objects:

- physical compact recurrent compute modules;
- logical capability bindings with their own intention adapters and external
  recurrent states.

Two related registry procedures are bound to the same physical compute module
and trained sequentially. The shared basis, physical compute, and first
binding are frozen before the second binding is trained. Both seeds promote:
reloaded behavior is `1.0000/0.8906` and `1.0000`; exact reload, independent
binding state, all retention, checksum recovery, frozen-core, and zero-replay
gates pass. The library has one physical compute module for two logical
bindings and a `0.7016` payload ratio versus two independent full programs.

The matched opaque-rule control is rejected at `0.6367` when forced to reuse
the physical module. This is the intended safety boundary: reuse is admitted
only for fresh-verified compatible procedures; unrelated procedures must grow
new compute rather than silently sharing an insufficient module.

This is verified compute reuse, not general continual learning. The next step
is a content-addressed reuse policy that chooses among multiple physical
modules, then grows a new one when all candidates fail fresh verification.
