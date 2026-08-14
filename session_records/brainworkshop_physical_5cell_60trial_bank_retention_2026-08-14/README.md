# Physical five-cell 60-trial bank retention (2026-08-14)

Status: **promoted as bounded five-cell retention and route learning**.

Exactly one public difficulty axis advanced from four to five visible cells.
Session length changed separately before the experiment from twelve to sixty
ordinary one-second Brain Workshop trials to amortize sensory warm-up, capture
initialization, and ready-screen transitions. The Position 1-Back rule,
scoring, public pixels, and ordinary keyboard actions were unchanged.

An isolated one-lifetime probation started from the promoted four-cell bank
with a fresh frontend seed. It consumed 60/60 visible stimulus events, emitted
57 actions after three no-action warm-up events, and received 15/15 positive
public verifier outcomes. Its 15 route observations were not inherited by the
promotion candidate.

A new transactional copy then ran three fresh sixty-trial lifetimes with
another frontend seed. It consumed 180/180 events, emitted 171 actions after
the three warm-ups per lifetime, and received 56/56 positive verifier outcomes.
The cumulative public accuracy remained 1.0 from verifier bit 1 through every
later measured prefix. Configured live time was 192.16 seconds and wall time
was 194.65 seconds.

The controller and admitted program remained immutable. Controller optimizer
updates, external-program optimizer updates, and replay were all zero. Only
the promotion run's 56 live reward-input observations advanced the bank from
version 88/twenty-six contexts to version 144/twenty-nine contexts. The single
program digest remained
`90e20193a50fdfa22b75fe722e6a9e131d9ba05d7f7e7d0aedbce9fc1f3c5749`.

This establishes bounded retention/generalization of one Position 1-Back
temporal program across five visible cells. It does not add another program or
establish 2-back. Six cells must begin with a separate sixty-trial probation.
The promoted `AgentBrain.bank` SHA-256 is
`45c1bbb1431fef3094c761beab52042768760062ba300af8a5913e2bd0b6e94b`.
