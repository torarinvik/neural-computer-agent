# Fifth-slot saturation and zero-shot transfer — 2026-08-03

The promoted fourth-slot complement checkpoint was given one more appended
slot with the same 1,536-lifetime protected recipe. The candidate reached
72.74% complement accuracy versus 72.29% for its parent: only **+0.46 points**
causal gain, below the +5 promotion bar. It is retained only as
`artifacts/checkpoints/complement_population_fifth_slot_seed93880_unpromoted.pt`.

This is the stopping signal for repeating the same complement primitive. More
slots and more same-task data are no longer a high-ROI path at this frontier.

As a transfer check, the three-slot parent and promoted fourth-slot child
both scored **73.44%** on the same zero-shot span-11 audit seed, with blank and
reset controls near chance. The fifth-slot candidate scored **73.43%**. The
new complement skill therefore does not measurably improve span-11 transfer;
the next task must supply a genuinely new relation or a better credit path.

The next frontier is a gradual new primitive (or a carefully isolated span-11
credit-assignment experiment), not a sixth complement slot. Preserve the same
causal, shuffled, blank, reset, and old-span retention audits.
