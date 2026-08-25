from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(artifact_path: str | Path, hash_path: str | Path) -> str:
    expected = Path(hash_path).read_text(encoding="ascii").strip().split()[0]
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise RuntimeError(f"Frozen protocol hash mismatch: expected {expected}, got {actual}")
    return actual


def load_and_verify_protocol(
    artifact_path: str | Path,
    hash_path: str | Path,
    repository_root: str | Path,
) -> tuple[dict[str, object], str]:
    freeze_hash = verify_freeze(artifact_path, hash_path)
    protocol = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    root = Path(repository_root)
    for relative_path, expected in protocol.get("source_hashes", {}).items():
        actual = sha256_file(root / relative_path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen source hash mismatch for {relative_path}: expected {expected}, got {actual}"
            )
    return protocol, freeze_hash
