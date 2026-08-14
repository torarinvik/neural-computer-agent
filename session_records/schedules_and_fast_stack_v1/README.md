# F231: sequencing acquired (2026-08-13)

ms/   mechanism_schedule.py, 6 seeds: schedules from top-4 solo
      singles. Chosen 6/6 on resource1, +0.065 only.
fs/   fast_schedule.py v1, 6 seeds: full schedule race on the fast
      stack; clean schedule present but LOSES (completion
      unobservable for consumables).
fs2/  fast_schedule.py v2, 6 seeds: consumption-aware completion.
      resource1 +0.128 -> +0.586 (t=+5.73, 6/6); clean schedule wins
      in every seed.

Narrative: docs/MEMORY_BANK_DESIGN.md F231. Scope: DEV mechanism set.
