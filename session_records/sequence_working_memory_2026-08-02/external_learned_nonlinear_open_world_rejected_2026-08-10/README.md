# Learned nonlinear open-world factual memory — rejected

This three-seed diagnostic replaced the promoted fixed random-feature factual
basis with a trainable nonlinear MLP and used the new one-pass
`streaming_gradient` candidate protocol. Four regimes exposed only `48/64`
training rows. Current-window local optimization used four optimizer updates
per four-row window; old-regime replay and raw provisional-row retention were
zero.

| seed | max held-out MSE | revisit result | quality gate | verdict |
| ---: | ---: | --- | :--- | :--- |
| 82601 | 0.0733 | no old-slot revisit matched | pass | reject |
| 82602 | 0.1593 | no old-slot revisit matched reliably | fail | reject |
| 82603 | 0.0378 | only a subset matched | pass | reject |

The failure is architectural and specific: the MLP can sometimes fit a
partial current regime, but its factual prediction error is not stable enough
for the router to identify old regimes. Tight routing thresholds turn revisits
into capacity pressure; loose thresholds admit wrong matches. The fixed
random-feature sufficient-statistics family remains the promoted factual
baseline.

This rejects the naive learned-MLP substitution, not learned nonlinear memory
as a research direction. The next candidate is a representation-stable or
meta-learned nonlinear model with an independently verified route query.
