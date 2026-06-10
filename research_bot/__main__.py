"""CLI: ``python -m research_bot <command>`` (Windows-first).

Commands:
  crawl    Run seed queries against OpenAlex; add new hits to literature/auto_index.yaml
  status   Show curated vs auto-discovered counts
"""
from __future__ import annotations

import argparse
import sys

from . import discovery, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("crawl", help="crawl OpenAlex from seeds.yaml into the literature brain")
    pc.add_argument("--per-page", type=int, default=discovery.C.DEFAULT_PER_PAGE)
    pc.add_argument("--no-cache", action="store_true", help="bypass the on-disk response cache")
    pc.add_argument("--seeds", default=None, help="path to a seeds.yaml (defaults to packaged)")

    sub.add_parser("status", help="show literature-brain counts")

    args = parser.parse_args(argv)

    if args.command == "crawl":
        queries = discovery.load_seeds(args.seeds)
        print(f"[research_bot] crawling {len(queries)} seed queries...")
        records = discovery.crawl(queries, per_page=args.per_page, use_cache=not args.no_cache)
        summary = state.add_discovered(records)
        print(
            f"[research_bot] fetched {len(records)} unique papers; "
            f"added {summary['added']} new -> auto_index ({summary['auto_total']} total, "
            f"{summary['curated_total']} curated)"
        )
        return 0

    if args.command == "status":
        s = state.stats()
        print(f"[research_bot] curated={s['curated']} auto_discovered={s['auto_discovered']}")
        print(f"  index:      {s['index_path']}")
        print(f"  auto_index: {s['auto_index_path']}")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
