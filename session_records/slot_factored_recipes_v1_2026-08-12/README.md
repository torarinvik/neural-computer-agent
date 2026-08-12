# Slot-factored recipes (F204-F207)

The wall F202 and F203 both stopped at is not arity, it is SEQUENTIAL
SEMANTICS. A parallel recipe assigns each slot one instruction
`(op, j, m)`, all reading the pre-state.

## F204 language + plant

Held out, fit on 32 transitions, scored on a different 128-row draw,
ground-truth execution on both sides:

    domain            arity  parallel        seq d1        seq d2
    held-out games    4.125  0.8982 @ 1119   0.7821 @ 158  0.8687 @ 26398
    seen games        3.167  0.9014 @  910   0.8260 @ 125  0.8975 @ 16933
    rule families     0.917  1.0000 @  160   1.0000 @  21  1.0000 @    21

Parallel cost is FLAT in arity, sequential is exponential. Overfit gap
(in-sample minus held-out) parallel +0.0114 vs depth-2 -0.0005.

Plant, 12k updates:  summed code 1/3/6 steps 0.3747 / 0.3809 / 0.4055;
one residual step per SLOT 1.0000. Ablation: dropping the pre-state
re-feed still scores 1.0000, so per-slot decomposition is the whole
effect (my first explanation was wrong). At 40k: 0.9973-1.0000 random
states, 0.9605-0.9919 on real-world states.

## F205 reader, 3 seeds, paired per world-action

    section              n    floor   READER  search   search-reader
    held-out games     120   0.4820   0.7226  0.8370   +0.1144 t=9.18
    seen game shapes   180   0.4437   0.8382  0.8359   -0.0023 t=-0.77
    held-out families  299   0.1698   0.8847  0.9411   +0.0564 t=5.26

Controls: identity floor; mode program; shuffled labels (scores the
mode EXACTLY); wrong world same action; wrong world AND action.

## F206 wake mixture, interior optimum at 0.30

    share  held-out  reader-wrongworld        search-reader  families
    0.00     0.7226  +0.0221 t=1.55 (ns)          +0.1144     0.8847
    0.15     0.7388  +0.0322 t=2.57               +0.0982     0.8907
    0.30     0.7803  +0.0440 t=4.15               +0.0567     0.8818
    0.45     0.7779  +0.0511 t=3.65               +0.0591     0.8501
    0.60     0.7334  +0.0500 t=3.08               +0.1035     0.8052

## F207 control loop, 12 seeds, 10 held-out compound worlds

    arm                 share 0.00   share 0.30
    mode-program           -0.7897      -0.7897
    random                 -0.6652      -0.6652
    read from WRONG world   0.0186      -0.0288
    READ (1 forward pass)   0.0566       0.0310
    oracle                  0.0466       0.0466
    searched (1119/action)  0.1197       0.1197

    read - oracle       t=0.27        t=-0.37     (indistinguishable)
    read - random       t=20.5 12/12  t=17.5 12/12
    read - WRONG world  t=1.41  8/12  t=3.59 11/12   <- resolves at 0.30
    searched - read     t=1.42        t=2.14

Reproduce: `python -m experiments.games_amodal.probes.factored
--train-source mixed --wake-games 15 --synthetic-share 0.3 --seed S`
and `...probes.factored_control --train-source mixed --wake-games 15
--synthetic-share 0.3 --game-from 15 --game-to 25 --seed S`.
Single-threaded (`torch.set_num_threads(1)` is pinned in-script).
