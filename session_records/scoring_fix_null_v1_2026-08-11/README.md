# Scoring fix is a null (F152)

Probe 250, 3 seeds paired against F143. Terminal scoring (counts the
horizon once) vs summed scoring (over-counts r_3 by 4x):

    +0.1175 / +0.1247 / +0.1371   pooled +0.1264
    against summed                pooled +0.1229

Beam search consumes only the argmax, so a distortion that preserves
plan ORDERING is invisible. The objective was arithmetically wrong and
ordinally right.

Fifth eliminated candidate for the games residual.
