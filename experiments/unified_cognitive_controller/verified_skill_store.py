"""Atomic, hash-verified disk storage for promoted latent skills."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VerifiedSkillStore:
    """Append-only skill files with atomic manifest updates and verification."""

    schema = "verified-latent-skill-store-v1"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "manifest.json"
        if not self.manifest_path.exists():
            self._write_manifest({"schema": self.schema, "entries": []})

    def _manifest(self) -> dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema") != self.schema:
            raise ValueError("unsupported skill-store schema")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=self.root, prefix=".manifest-",
                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)

    def commit(
            self, payload: dict[str, Any], *,
            context_key: torch.Tensor,
            lower_confidence_bound: float,
            verifier_bits: int,
            parent_id: str | None,
            provenance: dict[str, Any]) -> str:
        if context_key.ndim != 1:
            raise ValueError("context key must be one-dimensional")
        if lower_confidence_bound <= 0:
            raise ValueError("unverified skill cannot be committed")
        if verifier_bits < 0:
            raise ValueError("verifier bits cannot be negative")
        with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=".skill-", suffix=".tmp",
                delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save({
                "schema": "verified-latent-skill-v1",
                "payload": payload,
                "context_key": context_key.detach().cpu(),
            }, temporary)
            digest = _sha256(temporary)
            skill_id = digest[:20]
            destination = self.root / f"{skill_id}.pt"
            if not destination.exists():
                os.replace(temporary, destination)
            else:
                temporary.unlink()
            manifest = self._manifest()
            if not any(
                    row["skill_id"] == skill_id
                    for row in manifest["entries"]):
                manifest["entries"].append({
                    "skill_id": skill_id,
                    "file": destination.name,
                    "sha256": digest,
                    "parent_id": parent_id,
                    "lower_confidence_bound":
                        float(lower_confidence_bound),
                    "verifier_bits": int(verifier_bits),
                    "context_key": context_key.detach().cpu().tolist(),
                    "provenance": provenance,
                })
                self._write_manifest(manifest)
            return skill_id
        finally:
            if temporary.exists():
                temporary.unlink()

    def entries(self) -> list[dict[str, Any]]:
        return list(self._manifest()["entries"])

    def load(
            self, skill_id: str, *,
            device: torch.device | str = "cpu") -> dict[str, Any]:
        matches = [
            row for row in self._manifest()["entries"]
            if row["skill_id"] == skill_id]
        if len(matches) != 1:
            raise KeyError(skill_id)
        entry = matches[0]
        path = self.root / entry["file"]
        if _sha256(path) != entry["sha256"]:
            raise ValueError("skill file failed SHA-256 verification")
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != "verified-latent-skill-v1":
            raise ValueError("unsupported latent-skill schema")
        return payload
