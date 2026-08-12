# scale_up.py is goal_language.py with constants changed; this is the
# exact derivation (run from the repo root).
import pathlib, re
src = pathlib.Path("experiments/games_amodal/probes/goal_language.py").read_text()
s = src.replace("SLOTS, VALUES = 6, 8", "SLOTS, VALUES = 6, 12")
s = s.replace("HEIGHT = WIDTH = 8", "HEIGHT = WIDTH = 12")
s = re.sub(r"FamilyVerifier\((.*?)\)",
           lambda m: f"FamilyVerifier({m.group(1)}, height=HEIGHT, width=WIDTH)",
           s, flags=re.S)
# (docstring replaced; see scale_up.py)
