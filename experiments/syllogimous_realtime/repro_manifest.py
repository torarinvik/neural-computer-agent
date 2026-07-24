"""Capture the provenance needed to reproduce a run."""
from __future__ import annotations
import json, platform, subprocess, hashlib, struct, os
from pathlib import Path

def build_manifest(path: str | Path, *, seeds: list[int], config: dict,
                   model_versions: dict, compiler: str = "~/.elisac/elisac",
                   artifacts: list[str] | None = None) -> None:
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        git_dirty = bool(status.strip())
        git_untracked = [line[3:] for line in status.splitlines()
                         if line.startswith("?? ")]
        diff = subprocess.check_output(["git", "diff", "--binary"], text=False)
        git_diff_sha256 = hashlib.sha256(diff).hexdigest()
    except Exception:
        git_dirty, git_untracked, git_diff_sha256 = None, [], "unknown"
    digest = hashlib.sha256(b"".join(struct.pack("<q", seed) for seed in seeds)).hexdigest()
    compiler_source = os.environ.get("ELISA_COMPILER_REPO", "/Users/torarinvikbjarko/Documents/Coding Projects/Go projects/Elisa-core/compiler")
    try:
        compiler_revision = subprocess.check_output(["git", "-C", compiler_source,
                                                     "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        compiler_revision = "unknown"
    try:
        compiler_path = Path(compiler).expanduser()
        compiler_hash = hashlib.sha256(compiler_path.read_bytes()).hexdigest()
    except Exception:
        compiler_hash = "unknown"
    record = {"schema": "syllogimous.reproducibility.v1", "git_commit": git,
              "git_dirty": git_dirty, "git_untracked": git_untracked,
              "git_diff_sha256": git_diff_sha256,
              "compiler": compiler, "platform": platform.platform(),
              "compiler_source_revision": compiler_revision,
              "compiler_binary_sha256": compiler_hash,
              "python": platform.python_version(),
              "seed_spec": {"count": len(seeds), "first": seeds[0] if seeds else None,
                            "last": seeds[-1] if seeds else None, "sha256_le_i64": digest},
              "config": config, "model_versions": model_versions}
    record["artifacts"] = artifacts or []
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
