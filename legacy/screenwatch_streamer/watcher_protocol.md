# Screen-watcher protocol v3.2 — structured visuospatial memory

The watcher (the orchestra's *singer*) is a cheap, short-lived LLM agent that reads the
delta-encoded batch stream and maintains an *external, structured* understanding of the screen.
The external memory is the point: a watcher session is ephemeral, but its understanding survives
in files, so cheap watchers can be rotated over an unbounded stream. System contracts (stream
format, ownership, invariants) live in SPEC.md; this file is the singer's playbook.

**What changed in v3.** v2 kept one flat `state.md`. v3 splits understanding into a **3-tier
memory hierarchy** with a **typed schema**, so the system can watch an unbounded stream while its
total memory stays compact — the central research question ("can an LLM consume a visual world
sequentially and maintain a compact, evolving understanding of it").

**What changed in v3.1.** The **cursor** (SPEC "The cursor") gains a neural verb, `describe`, backed
by the local video-VLM member `screenvlm` (the *violin*). It answers *what is happening* for
non-textual motion content the symbolic stream can only report as statistics. Two rules bound it: the
**singer never calls it** (deep attention is the boss's budget), and every `describe` answer is
**Inferred-only** and evidence-pinned (I8). The Phase-3 trap-test measured its confab profile (SPEC
"describe trust policy"): text/scene claims are usable Inferred hypotheses, spatial-motion claims are
untrusted and need symbolic corroboration.

**What changed in v3.2.** The cursor gains a **symbolic** motion verb, `track`, backed by the *viola*
(`tracker`, model-free, deterministic — SPEC "tracker CLI contract"). It measures object count,
direction, reversal (edge vs mid-field), vanish/discontinuity, and occlusion from the delta stream —
the exact predicates the trap-test proved the VLM prior-fills. **Routing rule:** any *motion*
question (direction, (dis)appearance, reversal location, continuity, moving-object count) goes to
`track` first; a `describe` motion claim may only *corroborate* the tracker, never stand alone, and
may never become a `story.md` EVENT on the VLM's word (I8/I9). `track`, like `describe`, is a
boss-budgeted verb the singer never calls. Its OBSERVED/INFERRED split is structural (I9); its trap-
measured trust policy (SPEC "tracker trust policy") tells the boss which of its claims are load-bearing
(count/direction/reversal/vanish/occlusion) vs the documented limits (identity through long occlusion
or merge — the tracker honestly reports UNKNOWN/new-id rather than a confident wrong association).

## The memory hierarchy (3 tiers)

Modelled on human memory: a small volatile working set, a linear episodic trace, and a compressed
semantic story built by replay. Each tier has a hard size discipline; consolidation moves
information *down* the hierarchy (working → episodic → semantic) and forgets the raw form.

| Tier | File | Discipline | Analogue | What it holds |
|------|------|-----------|----------|---------------|
| 1 Working | `state.md` | **rewritten each cycle, ≤ 2 KB** | prefrontal working memory | the *now*: OBSERVATIONS, OBJECTS, RELATIONS, HYPOTHESES, OPEN_QUESTIONS |
| 2 Episodic | `log.md` | **append-only**, one line/event | hippocampal event trace | EVENTS: what happened, when, to which object, with evidence |
| 3 Semantic | `story.md` | **rewritten on consolidation**, ≤ 6 KB | consolidated neocortical story | EPISODE_SUMMARY, DURABLE_OBJECTS, EVIDENCE pins |

**The forgetting is load-bearing, at every tier.** Rewriting `state.md` under its cap forgets stale
*perception*. Consolidation folds aged `log.md` EVENTS into `story.md`'s summary and truncates them,
forgetting *episodic detail*. `story.md` is itself re-summarised under its cap, forgetting *narrative
detail*. What is never forgotten is **evidence**: claims compress, but every claim keeps a pointer
back to the pixels (retained batch, or a Tier-A archive frame pinned before pruning — SPEC.md
Tier C). Forget representation, not evidence.

## Object identity — the visuospatial spine

OBJECTS are the reason this is *visuospatial* memory and not a chat log. An object is any tracked
entity on screen: a window, a panel, a dialog, the cursor, a progress bar, a counter widget, a
sprite. Each gets a **stable id** (`o1`, `o2`, …) that is reused across all three tiers and across
watcher rotations. Rules:

- An object that leaves the working set because it is **occluded or off-screen** is not deleted — its
  `state` becomes `occluded`/`offscreen` and it stays in OBJECTS (or is promoted to
  `story.md` DURABLE_OBJECTS if long-lived). This is what lets the system answer "what was behind
  that dialog?" and survive the *occlusion* and *brief-object* eval scenarios.
- Reuse an id only for the **same** entity. If unsure whether a reappearing thing is `o3` returning
  or a new `o7`, record it as new with a HYPOTHESIS linking them — never silently merge.
- A one-shot value that matters (a counter reading, a toast text) is an **OBSERVATION with an
  evidence pointer**, and if it changes an object it also updates that object's `state`.

## Files the singer touches

| Path | Discipline |
|------|-----------|
| `/tmp/screen_batches/latest.txt`, `batch_<n>.txt`, `batch_<n>.ocr.txt` | read |
| `/tmp/screen_watch/goal.md` | read — defines what matters |
| `/tmp/screen_watch/state.md` | **rewrite every cycle** (Tier 1), ≤ 2 KB |
| `/tmp/screen_watch/log.md` | **append-only** (Tier 2), event boundaries only |
| `/tmp/screen_watch/story.md` | read every cycle; **rewritten only by the consolidator** (Tier 3) |
| `/tmp/screen_watch/escalation.md` | overwrite when blocked on a question |

## The cycle

1. **Inherit.** Read `goal.md` (relevance), `story.md` (the consolidated story + DURABLE_OBJECTS —
   your long-term context), `state.md` (the prior watcher's working set), `log.md` tail. Cold start
   if absent — note it.
2. **Poll.** Read `latest.txt`; if unchanged, sleep ~2s and re-poll. Note skipped-batch gaps.
3. **Triage-gate.** Read only the batch's first 3 lines (BATCH/STATS/ACTIVITY) first.
   `ACTIVITY still` with high conf → bump the CURSOR line in state.md and stop — do NOT read the
   frames, do NOT rewrite the schema. Most cycles should cost near-zero tokens.
4. **Perceive (non-still batches).** Read `batch_<n>.ocr.txt` FIRST if present — real text beats
   inferred shapes; quote OCR strings as OBSERVATIONS. Then the keyframe for layout, then deltas as
   motion evidence (`SAME` runs = still; `ROWS` clusters = localized activity; `SHIFT` = scroll;
   mid-batch `KEYFRAME`/`rate>1` = heavy change).
5. **Update the working set (Tier 1).** Rewrite `state.md` (schema below):
   - refresh OBSERVATIONS from this batch only (they are the *now*; old observations are forgotten
     unless they became an object or an event);
   - update OBJECTS — move/restate tracked objects, add new ones with fresh ids, flip departed ones
     to `occluded`/`offscreen`/`gone`, never silently drop a still-relevant one;
   - update RELATIONS and HYPOTHESES in place; carry a one-line belief history so a rejected
     hypothesis can't return; list what you still can't tell in OPEN_QUESTIONS.
6. **Re-ground** every 10th batch OR when top-hypothesis confidence < 60%: rebuild OBSERVATIONS +
   OBJECTS from the current keyframe+OCR alone, ignoring inherited beliefs; diff against the
   inherited working set; log any correction as an EVENT.
7. **Encode an episode (Tier 2).** At an event boundary only (an object appeared/changed/left, an
   activity phase changed, a hypothesis was confirmed/refuted), append ONE line to `log.md` in the
   EVENT format — actor object id + one sentence + evidence pointer.
8. **Consolidate (Tier 2 → Tier 3).** If you are the consolidator this cycle (heartbeat, or `log.md`
   past its line cap — see the consolidator playbook), fold the oldest EVENTS into `story.md` and
   truncate them. Otherwise skip — a normal watcher never writes `story.md`.
9. **Escalate** when a goal-relevant question is beyond reach (needs zoom, needs pruned history, or
   needs *event semantics* the grid can't give): overwrite `escalation.md` with the question +
   evidence pointer + why it matters. Then continue — never stall the stream. A specific new trigger:
   **`ACTIVITY video`/`switching` batches that are goal-relevant but OCR-empty** (high churn, no text —
   the grid measures *that* something moves, not *what happens*) → escalate with a describe-shaped
   question naming the span, e.g. "what happens on screen during t=48000..54000?". Do **not** call
   `screenvlm` yourself; the boss owns that budget.

## Tier 1 — `state.md` (working memory, rewrite each cycle, ≤ 2 KB)

```markdown
# Working memory — batch <b> (t=<ms>)
Goal: <one-line echo of goal.md>

## OBSERVATIONS   (this batch only; Observed pixels/OCR, each with evidence)
- "<OCR string>" at rows <a>-<b>            [batch <b> f<i>]
- <motion signature, e.g. SHIFT dy=12 rows 4-40 = scroll>   [batch <b>]

## OBJECTS        (the tracked visuospatial set; ids stable & reused)
| id | type        | where (grid)   | state    | seen b→b | conf |
|----|-------------|----------------|----------|----------|------|
| o1 | editor pane | cols 0-120     | active   | 3→<b>    | 90   |
| o2 | dialog      | rows 20-34 ctr | occluding o1 | <b>→<b> | 75 |

## RELATIONS
- o2 occludes o1 (modal)         [batch <b> f<i>]

## HYPOTHESES     (best explanation + confidence; keep belief history)
- H1 (80%): user is saving a file (o2 is a save dialog).
  history: rejected "search box" @b<k> — had OK/Cancel buttons.

## OPEN_QUESTIONS
- What is o1's filename? (occluded by o2)

## CURSOR
last batch: <b> | watchers: <n> | re-ground in: <k> | consolidate in: <k>
```

## Tier 2 — `log.md` (episodic trace, append-only)

One line per event; the actor is an object id where there is one. Never rewritten by a watcher —
only the consolidator truncates its head.

```
t=<ms> [<obj-id|-> ] <one-sentence event>   [batch <b> f<i>]
```

Example: `t=48210 [o2] save dialog appeared over the editor   [batch 44 f3]`

## Tier 3 — `story.md` (semantic memory, consolidator-only, ≤ 6 KB)

```markdown
# Story so far — consolidated through batch <b> (t=<ms>)

## EPISODE_SUMMARY   (compressed narrative; oldest episodes fold to one line each)
- [t=0..40s] Opened project X in VS Code, edited src/main; ran the build twice.
- [t=40..92s] Save dialog → wrote src/main; terminal showed 1 test failure.

## DURABLE_OBJECTS   (entities that persist across episodes; ids shared with state.md)
| id | type | identity | last seen |
|----|------|----------|-----------|
| o1 | editor | VS Code, project X | b<b> |

## EVIDENCE   (pins: claims the summary depends on, anchored to exact pixels — SPEC.md Tier C)
- pin p1: "build failed with 1 error" → arch seq 512 / batch 44 f3
```

## Consolidator playbook (Tier 2 → Tier 3; a heartbeat role, not every cycle)

Runs on a slow heartbeat or when `log.md` exceeds its line cap (~200 lines). Its job is *replay*:
turn recent episodic detail into durable narrative and forget the detail.

1. Read `story.md`, the head (oldest unconsolidated span) of `log.md`, and `goal.md`.
2. **Fold**: rewrite EPISODE_SUMMARY so the oldest EVENTS become one summary line per episode; keep
   recent EVENTS verbatim. Promote any long-lived OBJECT to DURABLE_OBJECTS (same id).
3. **Pin**: for every summarized claim a future query might need to verify at pixel level, add an
   EVIDENCE pin and mark its exact frame to survive Tier-A pruning (Tier C). A claim with no
   retained evidence and no pin must be softened to "inferred" or dropped — the story never asserts
   what it can no longer show.
4. **Truncate**: remove the folded EVENTS from `log.md`'s head (move to `log.archive.md` if kept).
5. **Re-summarise** `story.md` under its ≤ 6 KB cap: merge adjacent episodes, drop superseded
   hypotheses. The cap is the semantic-forgetting pressure.

Hard rule: consolidation only ever *loses representation*, never *invents*. Every EPISODE_SUMMARY
line must trace to EVENTS or EVIDENCE that existed. "Do I understand fully?" is never the loop
condition — "are the aged events folded and their evidence pinned?" is.

## escalation.md format (mailbox to the boss)

```markdown
# Escalation — batch <b>
question: <one specific question>
evidence: [batch <b> f<i> rows <a>-<b>]
why: <one line tying it to the goal>
```

## Boss playbook (the orchestrator; a stronger model, woken by escalation.md changes)

1. Read goal.md, story.md, state.md, log.md tail, escalation.md.
2. If goal-relevant: formulate ≤ 2 SPECIFIC questions → write `queries/q_<id>.md` per SPEC.md
   → dispatch a cheap sub-watcher per query. Pick the query `type` by what the question needs:
   `zoom`/`ocr` for text, `compare` to confirm a claimed change at the pixel, **`track`** (the viola)
   for *what moved* in a span (count/direction/reversal/vanish/occlusion — the authority on motion),
   and **`describe`** (the violin) for *what it means* (semantics/scene/text). Motion questions go to
   `track` first; reach for `describe` for semantics, or to corroborate — never for a bare motion fact.
3. On answers: append a boss-attributed EVENT to log.md; if durable, reflect it in story.md via the
   consolidator; update goal.md sub-goals if warranted; clear the escalation. A `track` answer is a
   symbolic measurement (I9): its trusted predicates (count, direction, reversal zone, vanish,
   occlusion — SPEC "tracker trust policy") may become an EVENT directly; its `INF` events carry
   confidence and its documented limits (identity through long occlusion/merge) stay OPEN_QUESTIONs.
   A `describe` answer is INFERRED (I8): a **text/scene** claim may seed a HYPOTHESIS or answer the
   question (never overriding `ocr`/`compare`/`track`); a **spatial-motion** claim is **untrusted** —
   resolve it with `track`/`compare` before any EVENT. Neither VLM text nor an unresolved motion guess
   becomes a story.md EVENT on the neural member's word alone.
4. Hard stops: ≤ 3 queries per escalation; `evidence-exhausted`/`pruned` answers go to an
   OPEN_QUESTION in state.md and are never re-asked for the same event.

## Sub-watcher playbook (executes one query)

Read the query file; execute by type (SPEC.md): temporal = close-read the pointed batch range;
spatial = read only the pointed keyframe rows; zoom = map rows→pixels via SPEC.md's formula and run
`screenocr batch_<b>.jpg --crop x,y,w,h`; **track** = run `tracker run <batch_dir>` (or `batch
<file>`), read the OBS/INF records over the query span, and answer the motion predicate from them —
the trusted claims (count/direction/reversal/vanish/occlusion) go under **OBSERVED** for `OBS` records
and **INFERRED** for `INF` events (with their conf), and the evidence pointer is the seq/t of the
records; **describe** = run
`screenvlm <batch_dir> --span t0,t1 [--region x,y,w,h] --q "<question>"`, then copy its `ANSWER` as
the finding under **INFERRED** and its `EVIDENCE` line verbatim as the evidence pointer — never
promote VLM text to OBSERVED (I8). Map screenvlm's exit: partial answer (`partial=true`) → status
`answered` with the caveat noted; `evidence-exhausted` on stderr → status `pruned`/`evidence-exhausted`.
Write `q_<id>.answer.md` with status answered / evidence-exhausted / pruned, the finding
(OBSERVED/INFERRED separated), and fresh evidence pointers. Do nothing else — no state.md / story.md
writes.
