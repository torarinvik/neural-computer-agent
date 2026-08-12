# Shared external register interpreter: rejected composition diagnostic

Date: 2026-08-07  
Status: rejected; diagnostic only  
Schema: `neural-computer.external-register-composition-diagnostic.v1`

This pressure test exercised the newly committed
`ExternalCapabilityRegisterMachine` with two opaque instruction vectors and
one shared interpreter. The verifier privately generated two-bit samples from
a fixed random event codebook. Instruction zero was trained to complement the
first bit and instruction one to complement the second bit. After primitive
training, both instructions and the interpreter were frozen and evaluated in
the held-out serial composition, which should complement both bits.

The primitive instructions each reached `1.0000` accuracy, but the frozen
serial composition reached `0.5000`, exactly chance for the two-bit output.
This rejects the current unconstrained MLP transition as a reusable operator
algebra. It does not reject the external-state boundary: the failure is that
the shared interpreter permits instruction-specific latent codes that solve
their individual objectives without forming a composable register language.

No capability was promoted. No checkpoint was curated. The next experiment
must add a factorized or otherwise explicitly compositional transition and
re-run the same primitive, composition, fresh, reward-shuffled, instruction
ablation, reload, corruption, and frozen-core controls on valid rendered
amodal events.
