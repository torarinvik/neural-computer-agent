# Active discovery transfer — changed target regime

The active-evidence seam was tested on a target regime different from the
original pressure test: rendered n-back-4 with opaque cue symbol 5. The
matched passive arm received the same extra lifetime and candidate exposure;
only active selected its probe using isolated model disagreement. Source rows
were not replayed or rewritten.

Across seeds `80–103`, active discovery passed `16/24` complete target gates;
passive matched exposure passed `14/24`. The paired outcomes were five active
wins, two active losses, and seventeen ties. Both arms retained the source
slot in `24/24` runs, left the controller unchanged, replayed zero examples,
and consumed `714` transition rows once. Active used `600` unique verifier
bits and `138` probe rows; passive used `592` bits and the same probe-row
budget.

This promotes a narrow positive transfer result for active evidence selection
under one changed regime. It does not promote universal transfer, unrestricted
memory growth, arbitrary computation, or general continual learning. The next
pressure test should vary the regime and evidence budget again while measuring
stable bits-to-threshold and failure-mode changes.
