"""CLI entrypoint — Windows-first: ``python -m emerald_ai <command>``.

Commands:
  eda      Run the Phase 1 imbalance/censoring feasibility analysis -> reports/feasibility.md
"""
from __future__ import annotations

import argparse
import os
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
    sub.add_parser("calibrate", help="Phase 4 calibration + conformal (RQ2)")
    sub.add_parser("explain", help="Phase 4 SHAP explainability (RQ3)")
    sub.add_parser("improve", help="RQ1 follow-up: sparsity/ratios vs the events ceiling")
    sub.add_parser("survival-check", help="Feasibility: can the censored loans support a survival model?")
    sub.add_parser("decide", help="Cost-sensitive decision policy: cost-optimal review threshold")
    sub.add_parser("followup-checks", help="Review follow-ups: paired RQ1 test, reason-code stability, population diagnostic")
    ba = sub.add_parser("build-artefact", help="Fit and write the deployable model artefact (no dataset needed to serve it)")
    ba.add_argument("--out", default=None, help="destination .joblib (default: artefacts/scorer.joblib)")
    serve_p = sub.add_parser("serve", help="Phase 5 decision-support demo (FastAPI + minimal UI)")
    # Hosted platforms (Render, Railway, Fly, Heroku, Cloud Run) inject the listening port and
    # expect the process to bind 0.0.0.0, so both default to the environment when it is set.
    serve_p.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    serve_p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    sf = sub.add_parser("score-file", help="Batch-score a CSV/XLSX of applicants -> *_scored.csv")
    sf.add_argument("input", help="path to a CSV/XLSX of applicants")
    sf.add_argument("-o", "--output", default=None, help="output CSV path (default: <input>_scored.csv)")
    ms = sub.add_parser("make-samples", help="Write data/example_cases.csv + sample_applicants.csv")
    ms.add_argument("-n", type=int, default=50, help="number of random sample applicants")

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

    if args.command == "calibrate":
        from . import calibrate

        path = calibrate.build_report()
        print(f"[emerald_ai] calibration report written -> {path}")
        return 0

    if args.command == "explain":
        from . import explain

        path = explain.build_report()
        print(f"[emerald_ai] explainability report written -> {path}")
        return 0

    if args.command == "improve":
        from . import improve

        path = improve.build_report()
        print(f"[emerald_ai] improvement experiment written -> {path}")
        return 0

    if args.command == "survival-check":
        from . import survival

        path = survival.build_report()
        print(f"[emerald_ai] survival feasibility report written -> {path}")
        return 0

    if args.command == "decide":
        from . import decision

        path = decision.build_report()
        print(f"[emerald_ai] decision policy report written -> {path}")
        return 0

    if args.command == "followup-checks":
        from . import followup

        path = followup.build_report()
        print(f"[emerald_ai] follow-up checks written -> {path}")
        return 0

    if args.command == "build-artefact":
        from . import serve

        info = serve.build_artefact(args.out)
        print(f"[emerald_ai] artefact written -> {info['path']} ({info['size_kb']} KB)")
        print(f"[emerald_ai] operating point P>={info['threshold']} "
              f"(catch-rate {100*info['catch_rate']:.0f}% of {info['training_events']} events, "
              f"{info['training_rows']} rows)")
        print("[emerald_ai] this file is all a server needs: no dataset, no row-level data")
        return 0

    if args.command == "serve":
        from . import serve

        serve.run(host=args.host, port=args.port)
        return 0

    if args.command == "score-file":
        from . import serve

        info = serve.score_file(args.input, args.output)
        print(f"[emerald_ai] ranked {info['n']} applicants — review queue: "
              f"{info['n_review_queue']} (top decile of batch); "
              f"{info['n_riskiest_decile']} clear the absolute historical threshold "
              f"-> {info['out_path']}")
        return 0

    if args.command == "make-samples":
        from . import serve

        info = serve.write_sample_files(n=args.n)
        print(f"[emerald_ai] wrote {info['example_cases']} and {info['sample_applicants']} "
              f"(n={info['n_random']})")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
