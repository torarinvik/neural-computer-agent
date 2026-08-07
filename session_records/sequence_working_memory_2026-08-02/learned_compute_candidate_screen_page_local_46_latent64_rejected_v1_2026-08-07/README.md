# Page-local 46-candidate latent-width control rejected (2026-08-07)

This control keeps the promoted 46-candidate page-local budget and
representation assignment but increases the learned screen latent width from
32 to 64. It does not repair source retention: seed `69316` reaches `0.8750`
with a `0.0000` per-target floor, and seed `69317` reaches `0.9375` with a
`0.4000` floor. All unseen candidates still pass at `1.0000`.

Increasing the query/key latent width alone is therefore rejected. The next
capacity axis is router hidden width or an interference-isolating source
screen, not another signature transform.
