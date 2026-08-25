from __future__ import annotations

from pathlib import Path

import pytest

from src.freeze import load_and_verify_protocol, sha256_file, verify_freeze


def test_frozen_configuration_integrity(tmp_path: Path) -> None:
    artifact = tmp_path / "protocol.json"
    artifact.write_text('{"version": 1}\n', encoding="utf-8")
    hash_file = tmp_path / "protocol.sha256"
    hash_file.write_text(f"{sha256_file(artifact)}  protocol.json\n", encoding="ascii")
    assert verify_freeze(artifact, hash_file) == sha256_file(artifact)
    artifact.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_freeze(artifact, hash_file)


def test_frozen_source_hashes_are_verified(tmp_path: Path) -> None:
    source = tmp_path / "method.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "protocol.json"
    artifact.write_text(
        '{"source_hashes": {"method.py": "' + sha256_file(source) + '"}}\n',
        encoding="utf-8",
    )
    hash_file = tmp_path / "protocol.sha256"
    hash_file.write_text(f"{sha256_file(artifact)}  protocol.json\n", encoding="ascii")
    protocol, _ = load_and_verify_protocol(artifact, hash_file, tmp_path)
    assert protocol["source_hashes"]["method.py"] == sha256_file(source)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        load_and_verify_protocol(artifact, hash_file, tmp_path)
