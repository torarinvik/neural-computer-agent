"""Comparable model-size slots for baseline runs.

Names are configuration labels, not claims that a particular checkpoint is
installed.  A run manifest must fill in the actual revision before scoring.
"""
MODEL_SLOTS = (
    {"name": "1m", "parameters": 1_000_000, "modality": "stream"},
    {"name": "5m", "parameters": 5_000_000, "modality": "stream"},
    {"name": "20m", "parameters": 20_000_000, "modality": "stream"},
    {"name": "100m", "parameters": 100_000_000, "modality": "stream"},
    {"name": "350m", "parameters": 350_000_000, "modality": "stream"},
    {"name": "1b", "parameters": 1_000_000_000, "modality": "stream"},
)
