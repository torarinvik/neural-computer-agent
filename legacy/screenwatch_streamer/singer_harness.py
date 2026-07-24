#!/usr/bin/env python3
"""singer_harness.py — thin context preparer + applicator for the Screen Watcher Singer.

This is the start of the executable "singer" side. It does the mechanical work
(dirs, polling skeleton, slicing) so the LLM (you or another model) only does
the neural perception + memory update.

It never invents observations — it only prepares faithful slices and applies
what the model emits.

Core flow (per watcher_protocol.md + singer_prompt.md):
- Inherit goal / state / log / story
- Triage on header only (cheap)
- For full perceive: OCR first (if present), then keyframe ASCII (full or relevant), deltas summary
- The model returns:
  - New state.md content (full)
  - Log lines to append (if any)
- This script can apply them atomically.

Current status: context preparation + apply supported. Live poll loop is stubbed
(works when recorder is running and you have permissions).

Usage:
  python singer_harness.py prepare \
      --batch-dir /tmp/counter_skip_run \
      --watch-dir /tmp/counter_skip_run \
      --batch 3 \
      --out /tmp/singer_input_b3.md

  # Then feed /tmp/singer_input_b3.md (+ singer_prompt.md) to the model.
  # Model outputs the new state block and any log lines.

  python singer_harness.py apply \
      --watch-dir /tmp/counter_skip_run \
      --state /tmp/new_state.md \
      --log-append /tmp/new_events.txt

  # Or for quick one-off on current latest:
  python singer_harness.py prepare --batch-dir /tmp/screen_batches --watch-dir /tmp/screen_watch --latest
"""

import argparse
import os
import re
import sys
import shutil
from pathlib import Path
from typing import Optional

BATCH_DIR_DEFAULT = "/tmp/screen_batches"
WATCH_DIR_DEFAULT = "/tmp/screen_watch"


def read_text(p: Path):
    if not p.exists():
        return ""
    return p.read_text()


def write_text_atomic(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, p)


def parse_latest(batch_dir: Path) -> int:
    lt = batch_dir / "latest.txt"
    if lt.exists():
        try:
            return int(lt.read_text().strip())
        except Exception:
            pass
    # fallback: find highest batch_N.txt
    nums = []
    for f in batch_dir.glob("batch_*.txt"):
        m = re.search(r"batch_(\d+)\.txt$", f.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else -1


def read_batch_header_and_ascii(batch_path: Path, max_ascii_rows: int = 80):
    """Return (header_lines, activity, ascii_sample, full_header_text)."""
    if not batch_path.exists():
        return [], "unknown", "", ""
    text = batch_path.read_text()
    lines = text.splitlines()

    header = []
    activity = "unknown"
    ascii_lines = []
    in_ascii = False
    in_color = False

    for ln in lines:
        if ln.startswith("BATCH "):
            header.append(ln)
        elif ln.startswith("STATS "):
            header.append(ln)
        elif ln.startswith("ACTIVITY "):
            header.append(ln)
            activity = ln.split()[1] if len(ln.split()) > 1 else "unknown"
        elif ln.startswith("KEYFRAME"):
            header.append(ln)
            in_ascii = True
            continue
        elif in_ascii:
            if ln.startswith("COLOR") or ln.startswith("FRAME ") or ln.strip() == "BATCH_END":
                in_ascii = False
                in_color = ln.startswith("COLOR")
                continue
            if len(ascii_lines) < max_ascii_rows:
                ascii_lines.append(ln)
        elif ln.startswith("FRAME ") and not in_ascii:
            header.append(ln)

    header_text = "\n".join(header)
    ascii_sample = "\n".join(ascii_lines)
    return header, activity, ascii_sample, header_text


def find_ocr(batch_dir: Path, b: int) -> str:
    ocr_p = batch_dir / f"batch_{b}.ocr.txt"
    if ocr_p.exists():
        return ocr_p.read_text()
    return ""


def get_batch_path(batch_dir: Path, b: int) -> Path:
    return batch_dir / f"batch_{b}.txt"


def prepare_context(batch_dir: Path, watch_dir: Path, batch_num: int, out_path: Optional[Path] = None):
    batch_path = get_batch_path(batch_dir, batch_num)
    header_lines, activity, ascii_sample, header_text = read_batch_header_and_ascii(batch_path)
    ocr_text = find_ocr(batch_dir, batch_num)

    goal = read_text(watch_dir / "goal.md").strip() or "(no goal.md)"
    state = read_text(watch_dir / "state.md").strip() or "(no prior state — cold start)"
    log_tail = "\n".join(read_text(watch_dir / "log.md").strip().splitlines()[-8:]) or "(empty log)"
    story = read_text(watch_dir / "story.md").strip() or "(no story.md yet)"

    # Triage note
    triage = "FULL PERCEPTION (non-still or first)" if activity != "still" else "MINIMAL (still high-conf — consider cursor bump only)"

    bundle = f"""# Singer Context Bundle — batch {batch_num}

You are the singer. Follow the rules in `singer_prompt.md` (and watcher_protocol.md + SPEC.md) exactly.

## Goal
{goal}

## Inherited working memory (state.md)
{state}

## Recent episodic trace (tail of log.md)
{log_tail}

## Consolidated story (story.md)
{story}

## Current batch to process
BATCH {batch_num}
Triage recommendation: {triage}

### Batch header (first lines)
{header_text}

### OCR (if present — read this first for real text)
{ocr_text if ocr_text else "(no OCR for this batch)"}

### Keyframe ASCII grid (luminance; first ~80 rows shown for token control)
Use the grid for layout and shape. Quote distinctive row patterns when describing.
```
{ascii_sample}
```
(If more rows or COLOR section or deltas are needed for this task, say so in OPEN_QUESTIONS or escalate.)

## Instructions
1. Inherit first.
2. Apply triage gate.
3. If full perceive: OCR → keyframe shapes → motion from header stats.
4. Produce:
   - Full new `state.md` content (valid markdown, ≤ ~2 KB, with evidence pointers like [batch {batch_num} f0]).
   - Zero or more lines to APPEND to log.md (only on real event boundaries, in the exact format `t=<ms> [<obj>] <sentence>   [batch {batch_num} fX]`).
5. Be conservative. Use stable object IDs. Never invent. When the grid is ambiguous, prefer explicit shape description + lower conf or OPEN_QUESTIONS.
6. After you respond with the new state and log lines, the harness (or operator) will apply them.

Return ONLY:
- A fenced block ```state
<complete new state.md>
```
- Then zero or more plain log lines (or a ```log
t=... lines
``` block).

Do not add extra commentary outside those blocks unless you are escalating.
"""

    if out_path:
        out_path.write_text(bundle)
        print(f"wrote context bundle → {out_path}")
    else:
        print(bundle)
    return bundle


def apply_updates(watch_dir: Path, new_state_path: Optional[Path], log_append_path: Optional[Path]):
    """Apply a produced state.md and/or log append lines atomically where possible."""
    watch_dir.mkdir(parents=True, exist_ok=True)

    if new_state_path and new_state_path.exists():
        content = new_state_path.read_text()
        # basic sanity: must look like state.md
        if "# Working memory" not in content and "state" not in content.lower():
            print("WARNING: new state content does not look like state.md header", file=sys.stderr)
        target = watch_dir / "state.md"
        write_text_atomic(target, content)
        print(f"applied → {target}")

    if log_append_path and log_append_path.exists():
        append_text = log_append_path.read_text().strip()
        if append_text:
            log_path = watch_dir / "log.md"
            with open(log_path, "a") as f:
                if not append_text.endswith("\n"):
                    append_text += "\n"
                f.write(append_text)
            print(f"appended events to {log_path}")

    print("apply complete.")


def main():
    ap = argparse.ArgumentParser(description="Singer harness for elisa-screenwatch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Build LLM input bundle for one batch")
    p_prep.add_argument("--batch-dir", default=BATCH_DIR_DEFAULT)
    p_prep.add_argument("--watch-dir", default=WATCH_DIR_DEFAULT)
    p_prep.add_argument("--batch", type=int, help="Specific batch number")
    p_prep.add_argument("--latest", action="store_true", help="Use latest.txt")
    p_prep.add_argument("--out", help="Write bundle to this path instead of stdout")
    p_prep.add_argument("--max-rows", type=int, default=80, help="Max ASCII rows to include")

    p_apply = sub.add_parser("apply", help="Apply produced state and/or log lines")
    p_apply.add_argument("--watch-dir", default=WATCH_DIR_DEFAULT)
    p_apply.add_argument("--state", help="Path to new state.md content")
    p_apply.add_argument("--log-append", help="Path to file containing lines to append to log.md")

    p_poll = sub.add_parser("poll", help="Stub for future live polling loop")
    p_poll.add_argument("--batch-dir", default=BATCH_DIR_DEFAULT)
    p_poll.add_argument("--watch-dir", default=WATCH_DIR_DEFAULT)
    p_poll.add_argument("--interval", type=float, default=2.0)

    args = ap.parse_args()

    if args.cmd == "prepare":
        bdir = Path(args.batch_dir)
        wdir = Path(args.watch_dir)
        if args.latest:
            b = parse_latest(bdir)
            if b < 0:
                print("no batches found", file=sys.stderr)
                sys.exit(1)
        else:
            if args.batch is None:
                print("--batch N or --latest required", file=sys.stderr)
                sys.exit(1)
            b = args.batch
        out = Path(args.out) if args.out else None
        prepare_context(bdir, wdir, b, out)

    elif args.cmd == "apply":
        wdir = Path(args.watch_dir)
        state_p = Path(args.state) if args.state else None
        log_p = Path(args.log_append) if args.log_append else None
        apply_updates(wdir, state_p, log_p)

    elif args.cmd == "poll":
        print("poll loop not fully implemented yet — use prepare + apply in a manual or scripted loop for now.")
        print("Future: will read latest, triage, prepare bundle, invoke model (or wait for input), apply.")
        sys.exit(0)


if __name__ == "__main__":
    main()
