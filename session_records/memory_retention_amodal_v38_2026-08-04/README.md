# Hard-retention write-critic control rejection

This control retained the differentiable transaction for parent learning but
used hard sampled writes during retention, together with the v37 critic. It
reproduced the last-write shortcut (`0.522` target-first versus `0.997`
target-last), showing that the critic failure is not caused by the
straight-through memory gradient.
