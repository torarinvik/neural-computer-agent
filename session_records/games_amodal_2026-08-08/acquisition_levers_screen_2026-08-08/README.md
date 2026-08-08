# Acquisition-lever screen: negative

Mastery-only screen (no ignorance rollouts) of the literature's on-policy
reliability levers on the corrected co-trained loop, 4 fresh seeds
(101-104) x 4 configs, local macOS torch.

    config     both-twins mastered
    base       0/4
    headinit   0/4   (--head-init 0.1; several total failures)
    normadv    1/4   (--normalize-advantage)
    both       0/4   (worst: repeated double failure)

Pre-stated bar: a lever must lift the both-mastered rate above control.
None did convincingly; head-init x0.1 looks actively harmful at this
budget (3000 updates may be too few once early entropy is that high).

Confound to keep in mind before comparing with F55's remote 8/16: these
runs differ from the F55 cohort in THREE ways at once (platform, seeds,
and ignorance off). In particular the possibility that the ignorance
objective itself aids acquisition (entropy regularisation via the
uniform-KL pressure) is untested and would reconcile the rates.

Consequence per the pre-registration: the twins' acquisition instability
is not the generic on-policy variance these levers target. The
goal-factored redesign (docs/GOAL_FACTORED_DESIGN.md) is now the main
path, not a refinement.
