"""The literature brain: load/save the paper index and keep curated vs auto-discovered separate.

Path constants below are module-level *on purpose* — tests redirect the brain by monkeypatching
``state.LIT_DIR`` / ``state.INDEX_PATH`` / ``state.AUTO_INDEX_PATH`` (and the matching constant in
``discovery.py``). Patch only one module and fixtures leak into the real brain.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from . import config as C

# --- Path constants (see module docstring re: test isolation) --------------
LIT_DIR: Path = C.PACKAGE_DIR.parent / "literature"
INDEX_PATH: Path = LIT_DIR / "index.yaml"          # curated, human-vetted
AUTO_INDEX_PATH: Path = LIT_DIR / "auto_index.yaml"  # auto-discovered


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or []


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(records, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def work_key(record: dict) -> str:
    """Stable dedup key: prefer the OpenAlex id, fall back to DOI, then title."""
    return str(record.get("id") or record.get("doi") or record.get("title", "")).lower()


def load_curated() -> list[dict]:
    return _read(INDEX_PATH)


def load_auto() -> list[dict]:
    return _read(AUTO_INDEX_PATH)


def add_discovered(new_records: Iterable[dict]) -> dict:
    """Add auto-discovered records to ``auto_index.yaml``, skipping anything already curated or
    already auto-discovered. Returns a small summary dict."""
    curated_keys = {work_key(r) for r in load_curated()}
    auto = load_auto()
    auto_keys = {work_key(r) for r in auto}

    added = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for rec in new_records:
        k = work_key(rec)
        if not k or k in curated_keys or k in auto_keys:
            continue
        rec = {**rec, "discovered_at": now, "origin": "auto"}
        auto.append(rec)
        auto_keys.add(k)
        added += 1

    if added:
        _write(AUTO_INDEX_PATH, auto)
    return {"added": added, "auto_total": len(auto), "curated_total": len(curated_keys)}


def stats() -> dict:
    return {
        "curated": len(load_curated()),
        "auto_discovered": len(load_auto()),
        "index_path": str(INDEX_PATH),
        "auto_index_path": str(AUTO_INDEX_PATH),
    }
