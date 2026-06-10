"""Tests for research_bot — with the path-isolation discipline this project requires.

CRITICAL: the literature brain's path constants live in BOTH ``state`` and ``discovery``. A test
that redirects only one leaks fixtures into the real ``literature/`` brain. The ``isolated_brain``
fixture patches every path constant in both modules.
"""
from __future__ import annotations

import pytest

from research_bot import discovery, state


@pytest.fixture
def isolated_brain(tmp_path, monkeypatch):
    lit = tmp_path / "literature"
    # state.py path constants
    monkeypatch.setattr(state, "LIT_DIR", lit)
    monkeypatch.setattr(state, "INDEX_PATH", lit / "index.yaml")
    monkeypatch.setattr(state, "AUTO_INDEX_PATH", lit / "auto_index.yaml")
    # discovery.py path constant (the one a careless test forgets)
    monkeypatch.setattr(discovery, "CACHE_DIR", lit / ".cache")
    return lit


def _fake_record(i: int) -> dict:
    return {"id": f"W{i}", "doi": None, "title": f"paper {i}", "year": 2020}


def test_add_discovered_dedups_and_separates(isolated_brain):
    first = state.add_discovered([_fake_record(1), _fake_record(2)])
    assert first["added"] == 2
    # re-adding the same ids adds nothing (dedup)
    again = state.add_discovered([_fake_record(1), _fake_record(2)])
    assert again["added"] == 0
    assert state.stats()["auto_discovered"] == 2


def test_curated_paper_not_re_added_to_auto(isolated_brain, monkeypatch):
    monkeypatch.setattr(state, "load_curated", lambda: [_fake_record(1)])
    summary = state.add_discovered([_fake_record(1), _fake_record(2)])
    assert summary["added"] == 1  # W1 is curated -> skipped; only W2 is new


def test_search_degrades_to_empty_on_network_error(isolated_brain, monkeypatch):
    def boom(*a, **k):
        raise discovery.requests.RequestException("no network")

    monkeypatch.setattr(discovery.requests, "get", boom)
    assert discovery.search("anything", use_cache=False) == []


def test_normalise_extracts_openalex_id():
    raw = {
        "id": "https://openalex.org/W123",
        "display_name": "t",
        "publication_year": 2021,
        "authorships": [{"author": {"display_name": "A. Smith"}}],
        "concepts": [{"display_name": "credit scoring"}],
        "cited_by_count": 5,
    }
    rec = discovery.normalise(raw, "q")
    assert rec["id"] == "W123"
    assert rec["authors"] == ["A. Smith"]
    assert rec["source_query"] == "q"
