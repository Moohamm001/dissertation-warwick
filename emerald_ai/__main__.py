"""CLI entrypoint — Windows-first: ``python -m emerald_ai <command>``.

Commands:
  eda      Run the Phase 1 imbalance/censoring feasibility analysis -> reports/feasibility.md
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emerald_ai")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eda", help="Phase 1 imbalance/censoring feasibility report")
    sub.add_parser("audit", help="Phase 2 target-leakage feature catalogue")
    sub.add_parser("preprocess-check", help="Fit the leakage-safe pipeline and report output shape")
    sub.add_parser("bakeoff", help="Phase 3 model x imbalance-strategy bake-off (RQ1)")
    sub.add_parser("figures", help="Build the step-by-step visual story (reports/visual_story.md)")
    sub.add_parser("evidence", help="Rule 2 learning-evidence proofs (reports/learning_evidence.md)")
    sub.add_parser("clean-report", help="Data-quality report: impossible-value cleaning impact")
    sub.add_parser("sensitivity", help="Raw-vs-cleaned sensitivity analysis of the bake-off")

    args = parser.parse_args(argv)

    if args.command == "eda":
        from . import eda

        path = eda.build_report()
        print(f"[emerald_ai] feasibility report written -> {path}")
        return 0

    if args.command == "audit":
        from . import feature_audit

        path = feature_audit.write_catalogue()
        print(f"[emerald_ai] feature catalogue + summary written -> {path}")
        return 0

    if args.command == "preprocess-check":
        from . import data, preprocess

        df = data.build_target(data.load_raw(), "paidoff_only")
        pre, types = preprocess.build_preprocessor(df)
        X = pre.fit_transform(df, df["y"])
        print(f"[emerald_ai] pipeline OK. inputs={sum(len(v) for v in types.values())} "
              f"(num={len(types['numeric'])}, low={len(types['low_card'])}, high={len(types['high_card'])}) "
              f"-> output matrix {X.shape}")
        return 0

    if args.command == "bakeoff":
        from . import experiments

        path = experiments.build_report()
        print(f"[emerald_ai] bake-off report written -> {path}")
        return 0

    if args.command == "figures":
        from . import figures

        path = figures.build_story()
        print(f"[emerald_ai] visual story written -> {path}")
        return 0

    if args.command == "evidence":
        from . import validation

        path = validation.build_report()
        print(f"[emerald_ai] learning-evidence report written -> {path}")
        return 0

    if args.command == "clean-report":
        from . import clean

        path = clean.build_report()
        print(f"[emerald_ai] data-quality report written -> {path}")
        return 0

    if args.command == "sensitivity":
        from . import experiments

        path = experiments.build_sensitivity_report()
        print(f"[emerald_ai] sensitivity report written -> {path}")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
