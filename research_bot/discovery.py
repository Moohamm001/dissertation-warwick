"""OpenAlex discovery: turn seed queries into normalised paper records.

``CACHE_DIR`` is a module-level path constant *on purpose* — tests redirect caching by
monkeypatching ``discovery.CACHE_DIR`` in addition to the path constants in ``state.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
import yaml

from . import config as C

# --- Path constant (see module docstring re: test isolation) ---------------
CACHE_DIR: Path = C.PACKAGE_DIR.parent / "literature" / ".cache"


def _cache_path(query: str, per_page: int) -> Path:
    h = hashlib.sha1(f"{query}|{per_page}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def normalise(work: dict, source_query: str = "") -> dict:
    """Project a raw OpenAlex work down to the fields the brain stores."""
    oa_id = (work.get("id") or "").rsplit("/", 1)[-1]  # 'https://openalex.org/W..' -> 'W..'
    authorships = work.get("authorships") or []
    concepts = work.get("concepts") or []
    primary = work.get("primary_location") or {}
    source = (primary.get("source") or {}) if isinstance(primary, dict) else {}
    return {
        "id": oa_id,
        "doi": work.get("doi"),
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "venue": source.get("display_name"),
        "authors": [a.get("author", {}).get("display_name") for a in authorships[:8]],
        "concepts": [c.get("display_name") for c in concepts[:6]],
        "source_query": source_query,
    }


def search(query: str, per_page: int = C.DEFAULT_PER_PAGE, use_cache: bool = True) -> list[dict]:
    """Query OpenAlex for ``query``; return normalised records. Network failures degrade to []."""
    cache = _cache_path(query, per_page)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [normalise(w, query) for w in raw]

    params = {
        "search": query,
        "per-page": per_page,
        "mailto": C.MAILTO,
        "sort": "relevance_score:desc",
    }
    try:
        resp = requests.get(C.OPENALEX_BASE, params=params, timeout=C.REQUEST_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:  # network or JSON error
        print(f"[research_bot] WARN query {query!r} failed: {exc}")
        return []

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(results), encoding="utf-8")
    return [normalise(w, query) for w in results]


def load_seeds(path: Path | None = None) -> list[str]:
    path = path or C.SEEDS_PATH
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    queries: list[str] = []
    for group in data.get("queries", {}).values():
        queries.extend(group)
    return queries


def crawl(queries: list[str], per_page: int = C.DEFAULT_PER_PAGE, use_cache: bool = True) -> list[dict]:
    """Run every query, flatten, and de-duplicate within this crawl by OpenAlex id."""
    seen: set[str] = set()
    out: list[dict] = []
    for q in queries:
        for rec in search(q, per_page=per_page, use_cache=use_cache):
            key = (rec.get("id") or rec.get("doi") or rec.get("title") or "").lower()
            if key and key not in seen:
                seen.add(key)
                out.append(rec)
    return out
