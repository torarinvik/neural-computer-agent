# Signed entry pathway (F113)

Probe 213. `game_slots.py --signed-entry` added to the F111 config:
the value head gains `tanh(polarity(entry)) * salience(state)`, making
promote/suppress one sign apart. 2 seeds (69316, 69317).

Held-out +0.0357 / +0.1027 (F111: +0.0069; oracle-value target +0.1234;
floor -0.0470). Entry effect +0.2230. top=food normal 0.667 vs inverted
0.073 — all gain on normal polarity; the tanh turns salience UP on
normal worlds and only OFF on inverted ones. Next: log the polarity
scalar per world; if never negative, fix the reader (diff-entries).
