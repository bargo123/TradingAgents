from __future__ import annotations

import pytest

from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.importer import ExperienceRebuilder


def test_rebuild_failure_preserves_previous_generation(tmp_path):
    catalog = ExperienceCatalog(tmp_path / "artifact")
    rebuilder = ExperienceRebuilder(catalog)
    old = rebuilder.rebuild()

    with pytest.raises(RuntimeError):
        rebuilder.rebuild(fail_after_stage="features")

    assert catalog.active_generation()["generation_id"] == old.generation_id

