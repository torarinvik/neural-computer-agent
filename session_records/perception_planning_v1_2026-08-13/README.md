# Perception by planning (F224)

pp-*: per-plane vocabulary only. Parity on single-object worlds
(t=-0.83), incumbent edge on multi-object (-0.4288, t=-4.04) --
residue localised to the relational nearest-object operation.

pp2-*: plus one generic relational operator nearest(channel, ref) with
searched anchor. vocab - handwritten = +0.0703, t=+1.31 overall;
multi-object edge erased (+0.026, t=+0.41). Selection picks non-human
anchors (object-plane anchoring) that beat slot_state on collect1/2 and
navigate1. slot_state is subsumed: one point (rel_anchor0) in a
generated space, no longer load-bearing.

Reproduce: python -m experiments.games_amodal.probes.perception_planning --seed S
