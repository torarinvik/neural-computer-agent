# The model-proxy gap dissolves (F223)

rg-*: reward-derived goals (ridge on contact features, zero rollouts).
Sane oracle>=bank ordering everywhere; exact twin weight sign-flips;
capability mixed (collect parity, choice collinearity-collapse,
intercept as registered). Two evidence bugs fixed pre-finding
(single-step evidence; after-avatar filter dropping all contact rows).

sb-*: the four-cell crossover. Goal chosen by bank rollouts -> bank
beats oracle (+0.9102 t=+2.45); chosen by oracle rollouts -> oracle
beats bank (+0.2669). The ordering follows the SELECTOR. The
bank-beats-oracle anomaly of F203/F216/F221/F222 was goal-selection
bias, not model error. F203's principle gains a clause: the oracle
ceiling is valid for fixed objectives only.

Reproduce: python -m experiments.games_amodal.probes.reward_goals /
selection_bias --seed S
