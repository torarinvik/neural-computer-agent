# Overlap-safe provisional candidate retention

The online router previously deleted an isolated candidate whenever a
committed slot matched a later aggregate transition bundle. That was an
unsafe lifecycle shortcut: two nonstationary regimes can share a prefix, so a
committed match is not proof that the uncommitted regime is obsolete. The
candidate is now retained until explicit promotion, eviction, or capacity
policy; its controller-independent state and replay-free evidence remain
isolated.

An exact same-seed control on rendered active discovery used seeds `80–103`,
masked-window state, tight route matching, and the existing multi-lifetime
held-out/recursive/retention/fresh-challenger gate. Complete gates improved
from `15/24` to `16/24` on n-back-3, `14/24` to `15/24` on n-back-4, and
`12/24` to `14/24` on n-back-5: `41/72` to `45/72` overall. The matched
passive retention arm completed `12/24`, `13/24`, and `13/24` respectively.
The different-cue n-back-5 active arm completed `13/24` versus `11/24`
passive.

All active runs retained the source slot byte-for-byte, left the controller
unchanged, used replay-free external updates, and replayed zero examples.
This promotes a bounded candidate-lifecycle and retention boundary, not
general continual learning or unrestricted memory growth. The remaining
failure modes are held-out family verification, capacity under unresolved
candidates, and later target-route recovery.
