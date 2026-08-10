# Copy-on-write transition-model consolidation

Three seeds (`70111`, `70112`, `70113`) passed a stronger consolidation
lifecycle audit. Two equivalent opaque transition slots were verified on
held-out evidence and shared one physical model, reducing storage from three
models to two while retaining both context addresses. A later update to the
second logical slot automatically detached it copy-on-write; the source model
remained byte-stable and all three contexts became physically independent.

Distinct source/target functions were rejected without mutation. The frozen
controller received zero updates, consolidation itself used zero optimizer
updates, replayed examples during the consolidation transaction were zero, and
bank persistence preserved the post-detachment state exactly.

This promotes safe parameter-sharing lifecycle behavior. It does not claim
semantic merging, unrestricted memory growth, or general continual learning.
