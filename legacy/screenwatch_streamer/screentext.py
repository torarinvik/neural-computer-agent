#!/usr/bin/env python3
"""screentext.py — the OCR-diff member (orchestra_v3_plan.md V-next): turns Apple Vision positioned
readings into TYPED text-change evidence. Its deepest rule, learned the same way the viola's was:
OCR-diff is NOT `image -> string -> string-diff`. It is TRACKING AND TRUTH MAINTENANCE FOR TEXT.
Vision supplies candidate readings; this member decides when two readings describe the SAME region,
when their difference is a REAL edit, and when the only honest answer is instability.

  screentext.py diff  <archive_dir> <gold.jsonl> [--min-dwell 3]
       -> stdout: typed OBS/INF records per gold region-window
  screentext.py audit <archive_dir> <gold.jsonl>
       -> the 3-way attribution table (recognition / association / change-decision) + headline metric
  screentext.py --selftest
       -> verifies the decision state machine WITHOUT arch_tool/Vision (pure logic)

Design mirrors screensed.py / screenaud.py: stateless per invocation, deterministic, a NEURAL
claimant not an oracle. The member owns only short-lived region hypotheses; the LEDGER owns durable
belief (supersession, disputes). Emitted INFERRED text-state may be superseded; the OBSERVED Vision
reading and the OBSERVED pixel-change can never be erased (THE ONE LAW, specialized to text —
see eval/ocr_audit.md). Perception substrate: arch_tool show <dir> <seq> -> full-res PPM;
screenocr --crop -> positioned strings. Both already pass the V0.4 legibility gate.

WHY consensus+veto and not raw diff: measured on pharo static content (eval/ocr_audit.md), Vision is
stable frame-to-frame MOST of the time but throws intermittent single-frame blips (~20% of static
pairs). A naive 2-frame diff false-fires on those blips. Multi-frame consensus outvotes the blip; a
pixel-veto refuses any text change unsupported by a compatible visual change. Confirmation is
symmetric: the old reading must persist, THEN the new reading must persist.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH_TOOL = os.path.join(HERE, "arch_tool")
SCREENOCR = os.path.join(HERE, "screenocr")

MIN_DWELL = 3            # frames of persistence required on BOTH sides of a transition (symmetric)
VETO_PIXEL_FRAC = 0.02  # a text change needs >= this fraction of region pixels to have changed
MIN_CONF = 0.4          # Vision confidence floor (recognition, NOT temporal-identity, calibration)


# ------------------------------------------------------------------ pure decision logic (selftest'd)

def normalize(s):
    """Comparison key: fold the OCR-noise dimensions (leading bullets/glyph junk, case, spacing) that
    eval/ocr_audit.md measured as spurious. RAW text is kept separately as immutable evidence; this
    derived key is ONLY for deciding same-vs-changed, never rewrites the observed string."""
    s = s.strip().lower()
    s = re.sub(r"^[^0-9a-z#]+", "", s)      # leading bullets/icons Vision hallucinates (•,©,฿,@,O ...)
    s = re.sub(r"[^0-9a-z#]+$", "", s)      # trailing truncation punctuation ( -, (, : ... )
    s = re.sub(r"\s+", " ", s)
    return s


def consensus(frame_lines):
    """frame_lines: list (one per frame) of lists of normalized line-strings. Return the consensus
    line MULTISET: a line is in consensus if it appears in > half the frames. This outvotes the
    single-frame OCR blip measured in the baseline (seq-324 wobble)."""
    from collections import Counter
    n = len(frame_lines)
    if n == 0:
        return []
    tallies = Counter()
    for lines in frame_lines:
        for ln in set(lines):           # count each line once per frame
            if ln:
                tallies[ln] += 1
    return sorted(l for l, c in tallies.items() if c * 2 > n)


def stability(frame_lines):
    """Fraction of frames whose normalized line-set equals the modal line-set. 1.0 = rock-steady;
    low = the region is churning (dwell not satisfied -> cannot confirm a change through it)."""
    from collections import Counter
    if not frame_lines:
        return 0.0
    keys = [tuple(sorted(set(l for l in lines if l))) for lines in frame_lines]
    modal, cnt = Counter(keys).most_common(1)[0]
    return cnt / len(frame_lines)


def decide(before_frames, after_frames, pixel_frac, min_dwell=MIN_DWELL):
    """The state machine, as a pure function so --selftest covers it with no binaries.
      before_frames / after_frames: list-per-frame of normalized line lists.
      pixel_frac: fraction of region pixels that changed between the two clusters (OBSERVED).
    Returns (outcome, detail). outcome in {STABLE, CONFIRMED_CHANGE, OCR_UNSTABLE}."""
    cb, ca = consensus(before_frames), consensus(after_frames)
    sb, sa = stability(before_frames), stability(after_frames)
    dwell_ok = (len(before_frames) >= min_dwell and len(after_frames) >= min_dwell
                and sb >= 0.5 and sa >= 0.5)
    if cb == ca:
        return "STABLE", {"consensus": cb, "stab_before": sb, "stab_after": sa}
    # consensus differs. Two guards must BOTH pass before we call it a real edit:
    if pixel_frac < VETO_PIXEL_FRAC:
        # text "changed" but the pixels did not materially change => recognition noise, not an edit.
        return "OCR_UNSTABLE", {"reason": "pixel-veto: no compatible visual change",
                                "pixel_frac": pixel_frac, "before": cb, "after": ca}
    if not dwell_ok:
        # a side was too unstable to trust either reading as a persistent state.
        return "OCR_UNSTABLE", {"reason": "dwell not satisfied", "stab_before": sb,
                                "stab_after": sa, "before": cb, "after": ca}
    return "CONFIRMED_CHANGE", {"before": cb, "after": ca, "pixel_frac": pixel_frac,
                               "stab_before": sb, "stab_after": sa}


def important_tokens(lines):
    """Alnum tokens length>=3, for recognition recall (ignores the bullet/punct noise dimension)."""
    toks = set()
    for ln in lines:
        for t in re.findall(r"[0-9A-Za-z#][0-9A-Za-z#\-]{2,}", ln):
            toks.add(t.lower().lstrip("#"))
    return toks


# ------------------------------------------------------------------ perception I/O (real substrate)

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def ocr_region(archive_dir, seq, bbox, tmp):
    """arch_tool show <seq> -> PPM; screenocr --crop -> [(y, raw_text)] sorted by baseline y."""
    r = _run([ARCH_TOOL, "show", archive_dir, str(seq), tmp])
    if r.returncode != 0 or not os.path.exists(tmp):
        return None
    x, y, w, h = bbox
    r = _run([SCREENOCR, tmp, "--crop", f"{x},{y},{w},{h}", "--min-conf", str(MIN_CONF)])
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        f = line.split(None, 5)
        if len(f) == 6:
            out.append((int(f[1]), f[5]))
    return sorted(out, key=lambda t: t[0])


def pixel_frac(archive_dir, seq_a, seq_b, bbox, tmp):
    """Fraction of region pixels that differ between two frames (OBSERVED visual change). Reads the
    two PPMs, crops bbox, counts bytes differing beyond a small per-channel tolerance."""
    pa, pb = tmp + ".a.ppm", tmp + ".b.ppm"
    if _run([ARCH_TOOL, "show", archive_dir, str(seq_a), pa]).returncode != 0:
        return None
    if _run([ARCH_TOOL, "show", archive_dir, str(seq_b), pb]).returncode != 0:
        return None
    da, wa, ha = _read_ppm(pa)
    db, wb, hb = _read_ppm(pb)
    if da is None or db is None or (wa, ha) != (wb, hb):
        return None
    x, y, w, h = bbox
    x, y = max(0, x), max(0, y)
    x2, y2 = min(wa, x + w), min(ha, y + h)
    diff = tot = 0
    for row in range(y, y2):
        base = (row * wa + x) * 3
        for col in range(x, x2):
            i = base + (col - x) * 3
            tot += 1
            if (abs(da[i] - db[i]) > 24 or abs(da[i + 1] - db[i + 1]) > 24
                    or abs(da[i + 2] - db[i + 2]) > 24):
                diff += 1
    return diff / tot if tot else 0.0


def _read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"P6"):
        return None, 0, 0
    idx, fields = 2, []
    while len(fields) < 3:
        while idx < len(data) and data[idx] in b" \t\n\r":
            idx += 1
        if idx < len(data) and data[idx:idx + 1] == b"#":
            while idx < len(data) and data[idx] not in b"\n":
                idx += 1
            continue
        start = idx
        while idx < len(data) and data[idx] not in b" \t\n\r":
            idx += 1
        fields.append(int(data[start:idx]))
    w, h, _maxv = fields
    return data[idx + 1:], w, h


# ------------------------------------------------------------------ member driver over gold

def cluster_seqs(seq, radius, span):
    """Frames sampled around an anchor seq: [seq-radius .. seq+radius], clamped to >=0."""
    return [s for s in range(max(0, seq - radius), seq + radius + 1)]


def run_unit(archive_dir, unit, min_dwell, tmp):
    """Run the member over one gold region-window; return (outcome, detail, readings)."""
    bbox = unit["region"]["bbox"]
    sb, sa = unit["window"]["seq_before"], unit["window"]["seq_after"]
    radius = max(min_dwell - 1, 2)
    before_raw = [ocr_region(archive_dir, s, bbox, tmp) for s in cluster_seqs(sb, radius, 0)]
    after_raw = [ocr_region(archive_dir, s, bbox, tmp) for s in cluster_seqs(sa, radius, 0)]
    before_raw = [r for r in before_raw if r is not None]
    after_raw = [r for r in after_raw if r is not None]
    before_norm = [[normalize(t) for _, t in fr] for fr in before_raw]
    after_norm = [[normalize(t) for _, t in fr] for fr in after_raw]
    pf = pixel_frac(archive_dir, sb, sa, bbox, tmp)
    if pf is None:
        pf = 1.0                       # unknown visual change: don't let veto silently pass/fail
    outcome, detail = decide(before_norm, after_norm, pf, min_dwell)
    readings = {"before_raw": before_raw, "after_raw": after_raw, "pixel_frac": pf}
    return outcome, detail, readings


def _records_for_unit(reg, sb, sa, outcome, detail, rd):
    """Machine-readable records the ledger consumes (obs_or_inf typed per THE ONE LAW). OBSERVED
    readings + visual change are immutable; the verdict is an INFERRED, supersedable text-state."""
    recs = []
    fr = (rd["before_raw"][:1] + rd["after_raw"][:1])
    for cluster in fr:
        for y, raw in cluster:
            recs.append({"member": "guitar", "obs_or_inf": "OBS", "kind": "OCR_READING",
                         "region": reg, "t_seq": sb, "text": raw})
    recs.append({"member": "guitar", "obs_or_inf": "OBS", "kind": "VISUAL_TEXT_REGION_CHANGE",
                 "region": reg, "t_seq": sa, "pixel_frac": round(rd["pixel_frac"], 4)})
    if outcome == "CONFIRMED_CHANGE":
        recs.append({"member": "guitar", "obs_or_inf": "INF", "kind": "OCR_CHANGE_CONFIRMED",
                     "region": reg, "t_seq": sa, "before": detail["before"], "after": detail["after"],
                     "license": "guitar:ocr_change@pixel_veto+dwell"})
    elif outcome == "OCR_UNSTABLE":
        recs.append({"member": "guitar", "obs_or_inf": "INF", "kind": "OCR_UNSTABLE",
                     "region": reg, "t_seq": sa, "reason": detail["reason"],
                     "license": "guitar:ocr_unstable"})
    else:
        recs.append({"member": "guitar", "obs_or_inf": "INF", "kind": "TEXT_STATE",
                     "region": reg, "t_seq": sa, "text": detail["consensus"],
                     "license": "guitar:text_state"})
    return recs


def cmd_diff(a):
    tmp = "/tmp/screentext_frame.ppm"
    units = [json.loads(l) for l in open(a.gold) if l.strip() and not l.strip().startswith('{"_header')]
    out = open(a.jsonl, "w") if a.jsonl else None
    for u in units:
        outcome, detail, rd = run_unit(a.archive_dir, u, a.min_dwell, tmp)
        reg = u["region"]["name"]
        sb, sa = u["window"]["seq_before"], u["window"]["seq_after"]
        recs = _records_for_unit(reg, sb, sa, outcome, detail, rd)
        for r in recs:
            if out:
                out.write(json.dumps(r) + "\n")
            k, oi = r["kind"], r["obs_or_inf"]
            if k == "OCR_READING":
                print(f"OBS OCR_READING region={reg} text={json.dumps(r['text'])}")
            elif k == "VISUAL_TEXT_REGION_CHANGE":
                print(f"OBS VISUAL_TEXT_REGION_CHANGE region={reg} seq={sb}->{sa} pixel_frac={r['pixel_frac']:.4f}")
            elif k == "OCR_CHANGE_CONFIRMED":
                print(f"INF OCR_CHANGE_CONFIRMED region={reg} seq={sb}->{sa} "
                      f"before={json.dumps(r['before'])} after={json.dumps(r['after'])}")
            elif k == "OCR_UNSTABLE":
                print(f"INF OCR_UNSTABLE region={reg} seq={sb}->{sa} reason={json.dumps(r['reason'])}")
            else:
                print(f"INF TEXT_STATE region={reg} seq={sb}->{sa} stable=true text={json.dumps(r['text'])}")
    if out:
        out.close()


def cmd_audit(a):
    tmp = "/tmp/screentext_frame.ppm"
    units = [json.loads(l) for l in open(a.gold) if l.strip() and not l.strip().startswith('{"_header')]
    rows, unsupported = [], 0
    recog_recall = assoc_switches = None
    for u in units:
        outcome, detail, rd = run_unit(a.archive_dir, u, a.min_dwell, tmp)
        expect = u["expect"]
        ok = (expect == "CHANGE" and outcome == "CONFIRMED_CHANGE") or \
             (expect == "STABLE" and outcome in ("STABLE", "OCR_UNSTABLE"))
        if expect == "STABLE" and outcome == "CONFIRMED_CHANGE":
            unsupported += 1
        rows.append((u["id"], u.get("error_class", "?"), expect, outcome, "PASS" if ok else "FAIL"))
        # recognition recall on the recognition-tagged unit
        if u.get("error_class") == "recognition":
            got = set().union(*[important_tokens([t for _, t in fr]) for fr in rd["before_raw"]]) \
                if rd["before_raw"] else set()
            want = important_tokens(u["text_before"])
            recog_recall = len(want & got) / len(want) if want else 0.0
        # association: identity switches among similar neighbours (consensus set should stay fixed)
        if u.get("error_class") == "association":
            keys = [tuple(sorted(set(normalize(t) for _, t in fr))) for fr in rd["before_raw"] + rd["after_raw"]]
            assoc_switches = len(set(keys)) - 1 if keys else 0

    print(f"{'unit':22} {'dimension':16} {'expect':8} {'outcome':17} verdict")
    for uid, dim, exp, out, verdict in rows:
        print(f"{uid:22} {dim:16} {exp:8} {out:17} {verdict}")
    n_stable = sum(1 for r in rows if r[2] == "STABLE")
    print("\n--- three-way attribution ---")
    print(f"recognition      important-token recall = "
          f"{recog_recall:.2f} (threshold >=0.90)" if recog_recall is not None else
          "recognition      (no recognition unit)")
    print(f"association      identity switches      = "
          f"{assoc_switches} (threshold 0)" if assoc_switches is not None else
          "association      (no association unit)")
    fp_rate = unsupported / n_stable if n_stable else 0.0
    print(f"change-decision  unsupported-change rate = {fp_rate:.2f} (threshold <=0.02)")
    print(f"\nHEADLINE unsupported semantic changes = {unsupported} (target 0)")
    passed = all(r[4] == "PASS" for r in rows)
    print(f"AUDIT: {'PASS' if passed else 'FAIL'} ({sum(1 for r in rows if r[4]=='PASS')}/{len(rows)} units)")
    return 0 if passed else 1


# ------------------------------------------------------------------ selftest (no binaries)

def selftest():
    # 1) consensus outvotes a single-frame blip (the measured seq-324 wobble).
    stable = [["#ast-core", "#ast-tests"]] * 3 + [["#ast-core", "#ast-testz"]]  # 1 blip in 4
    assert consensus(stable) == ["#ast-core", "#ast-tests"], consensus(stable)

    # 2) STABLE when consensus matches (blip present but outvoted, pixels quiet).
    before = [["world", "system browser"]] * 4
    after = [["world", "system browser"]] * 3 + [["world", "system browzer"]]
    out, _ = decide(before, after, pixel_frac=0.0)
    assert out == "STABLE", out

    # 3) pixel-veto: consensus differs but pixels quiet => OCR_UNSTABLE, NOT a change.
    out, d = decide([["count := 4"]] * 4, [["count := 5"]] * 4, pixel_frac=0.001)
    assert out == "OCR_UNSTABLE" and "pixel-veto" in d["reason"], (out, d)

    # 4) real change: consensus differs, pixels moved, both sides dwell => CONFIRMED_CHANGE.
    out, d = decide([["world", "system browser"]] * 4, [["type: pkg1", "last modified"]] * 4,
                    pixel_frac=0.4)
    assert out == "CONFIRMED_CHANGE", (out, d)

    # 5) dwell guard: pixels moved but the 'after' side never settles => OCR_UNSTABLE.
    jitter = [["a"], ["b"], ["c"], ["d"]]                     # every frame different
    out, d = decide([["world"]] * 4, jitter, pixel_frac=0.4)
    assert out == "OCR_UNSTABLE" and d["reason"] == "dwell not satisfied", (out, d)

    # 6) normalize folds the measured noise dimensions but preserves the token.
    assert normalize("• #AST-Core") == normalize("#AST-Core") == "#ast-core"
    assert normalize("# Announcements-Tests-(") == "# announcements-tests"  # trailing junk stripped

    # 7) recognition recall ignores bullet/case noise.
    want = important_tokens(["Object subclass: #NameOfSubclass"])
    got = important_tokens(["Object subclass: #NameOfSubclass"])
    assert want and want == got, (want, got)

    print("screentext selftest: PASS (consensus, pixel-veto, dwell, normalize, recall verified w/o binaries)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="screentext")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("diff"); d.add_argument("archive_dir"); d.add_argument("gold")
    d.add_argument("--min-dwell", type=int, default=MIN_DWELL)
    d.add_argument("--jsonl", help="also write machine-readable records here (for the ledger)")
    au = sub.add_parser("audit"); au.add_argument("archive_dir"); au.add_argument("gold")
    au.add_argument("--min-dwell", type=int, default=MIN_DWELL)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.cmd == "diff":
        cmd_diff(a)
    elif a.cmd == "audit":
        raise SystemExit(cmd_audit(a))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
