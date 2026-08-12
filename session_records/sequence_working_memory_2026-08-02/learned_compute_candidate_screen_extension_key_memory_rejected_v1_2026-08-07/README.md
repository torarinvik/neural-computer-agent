# Isolated extension key-memory control rejected (2026-08-07)

This is the safe version of learned address adaptation: the mastered base
screen and its keys stay fixed, while each appended extension owns a separate
trainable opaque key group. It is evaluated at the promoted 1024-update
source/full-prior boundary.

The control passes strict promotion on both seeds, but it does not improve the
baseline and slightly hurts the easy seed: unseen routing is `0.9792/1.0000`
versus the static-key baseline's `1.0000/1.0000`; one target falls to `0.7778`
on seed `69316`. The memory contract is retained for future behavioral update
rules, but this optimizer policy is not promoted.
