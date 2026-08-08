# Four-instruction bounded residual operator — rejected

This audit tests a new external-register operator,
`factorized_bounded_residual`. It normalizes the register, produces a
low-rank instruction-conditioned proposal, bounds it with `tanh`, and applies
an opaque feature-wise residual gate. The controller remains frozen and the
instruction chain still executes only on the external register snapshot.

The matched baseline variant passed every safety control on seeds `69316` and
`69317`: primitive retention, reward-shuffled rejection, missing-evidence
rejection, exact reload, checksum-corruption rejection, frozen parent, and
zero replay. Inherited final composition was `0.8867` on both seeds, but
stable composition required `20,480` and `24,576` verifier bits versus
`16,384` and `8,192` for matched fresh learners. The positive-transfer gate
therefore failed.

The fresh-outcome shared-blueprint pretraining variant improved inherited
composition to `0.9883` and `0.9688`. All safety controls still passed, but
both inherited paths required `8,192` bits while fresh learners required only
`4,096`. This also failed positive transfer.

Conclusion: bounded state updates improve execution stability but do not yet
produce a reusable blueprint for genuinely new computation. The next test
must measure held-out computation transfer, not only deeper execution of the
same known instruction chain.
