from __future__ import annotations

from dataclasses import asdict

import pytest

from confirmation import _config_from_protocol
from src.config import ResearchConfig


def test_confirmation_config_requires_exact_frozen_fields() -> None:
    values = asdict(ResearchConfig())
    assert _config_from_protocol({"research_config": values}) == ResearchConfig()
    values["unexpected"] = True
    with pytest.raises(RuntimeError, match="do not match"):
        _config_from_protocol({"research_config": values})
