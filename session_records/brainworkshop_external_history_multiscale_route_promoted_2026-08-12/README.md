# Mixed-depth external-history retention and routing (2026-08-12)

This audit tests whether one fixed controller/interpreter can retain and route
external computation files with different active temporal depths. Each file
stores its complete lifetime in append-only external history, while its
private active query profile is `(4, 4, 4, 4, 5)`: the first four files read
three preceding events plus the current event, and the final `nback4` file
reads four preceding events plus the current event. The shared machine window
is five records wide; inactive padding is explicitly masked.

The five files are `symbol_parity`, `triplet_parity`, `parity2`,
`switch_binary`, and `nback4`. The controller, learned event frontend, and
generic register interpreter are frozen after the first file. New file
parameters are trained from attempted action outcomes only; replayed examples
are zero.

Seeds 17 and 18 both reached `1.0000` on all four fresh lifetimes for every
file. The route bank then selected every file at `1.0000`, a same-context
replacement at slot four at `1.0000`, and retained the original source file at
`1.0000`. Unknown-context controls stayed near chance (`0.4875` and
`0.5117`), and all file digests remained unchanged during route learning.

This promotes replay-free multi-file retention and routing across mixed
bounded temporal depths. The query-depth profile is externally configured,
not learned; this does not establish learned depth selection, learned
compression, unrestricted memory growth, arbitrary program induction, or
general continual learning.

Detailed accounting is in `sample_efficiency_ledger.json` and the per-seed
summaries.
