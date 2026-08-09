# Promoted learned opaque transition-context rung

Two seeds (`69911`, `69912`) test whether an external context encoder can
replace supplied regime/context labels at the model-bank boundary. The
encoder was trained only from paired noisy views of opaque transition bundles;
the learner received no regime labels, task IDs, or privileged simulator fields.

The encoder produced stable same-bundle keys, separated the base, auxiliary,
and held-out target dynamics, and automatically allocated a new target model
slot. The target slot reached `1.0` measured mastery in both seeds after 21
and 19 optimizer updates, versus 38 and 26 for matched fresh target models.
Both prior slots retained `1.0` measured mastery, remained byte-stable, and
received no target updates. The controller was frozen and received zero
updates; old prior examples were not replayed during target adaptation.
Wrong-context, corruption, fresh-model, and exact persistence controls passed.

This promotes learned context formation for a bounded transition bundle, not
general continual learning. Context formation is trained before deployment,
the bundle is finite, the bank is append-only, current target examples are
reused during optimization and accounted for, and there is no alternating
unbounded stream, consolidation, or compression result yet. The next pressure
test is online context identity under alternating regimes and capacity
pressure.
