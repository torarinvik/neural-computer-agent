# Promoted policy-free model-bank continual rung

Two seeds (`69811`, `69812`) test the exported session's strongest
architectural prediction on genuinely different transition rules. A source
opaque transition model was trained first. A target model slot was then
appended outside the controller and initialized from the source model. Only
that target slot was adapted; behavior was derived by opaque model search,
not by a stored policy.

The target reached the fixed transition-loss threshold in 23 and 29 optimizer
updates, versus 35 and 36 for matched fresh target models. Both achieved
`1.0` target mastery in both seeds. Source mastery remained `1.0` after
target adaptation. The source slot was byte-stable, wrong-context and
corrupted controls failed, persistence was exact, and the frozen controller
received zero updates.

The target phase replayed zero old-source examples. It did reuse its current
target transition batch, which is accounted for explicitly in each report;
this is not a claim of zero replay of all target examples. Source pretraining
repeated its source batch and is separately accounted for.

This promotes a bounded model-bank result and a replicated acquisition-speed
signal, not general continual learning. Context vectors are supplied, slots
grow append-only, and there is no learned context discovery, consolidation,
compression, or long alternating-stream test yet.
