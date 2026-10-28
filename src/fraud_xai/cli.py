"""Command line: ``fraud-xai train | variants | report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    BAF_VARIANTS,
    baf_task,
    load_baf,
    load_ulb,
    make_synthetic_baf,
    ulb_task,
)
from .pipeline import Config, run_experiment, variant_sweep


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fraud-xai", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="tune, fit and evaluate on one dataset")
    t.add_argument("--dataset", choices=["baf", "ulb", "synthetic"], default="baf")
    t.add_argument("--variant", default="Base", choices=BAF_VARIANTS)
    t.add_argument("--data-dir", default="data")
    t.add_argument("--artifacts", default="artifacts")
    t.add_argument("--trials", type=int, nargs=3, metavar=("LGBM", "XGB", "CAT"))
    t.add_argument("--n-boot", type=int, default=500)
    t.add_argument(
        "--reuse-params", help="best_params.json; families listed in it are not tuned again"
    )

    v = sub.add_parser("variants", help="fixed LightGBM config across the six BAF variants")
    v.add_argument("--params", required=True, help="best_params.json from a BAF run")
    v.add_argument("--data-dir", default="data")
    v.add_argument("--out", default="results/variants.json")

    r = sub.add_parser("report", help="figures and the auditor case file")
    r.add_argument("--artifacts", default="artifacts")
    r.add_argument("--docs-dir", default="docs")
    r.add_argument("--reports-dir", default="reports")
    r.add_argument("--top-n", type=int, default=15)

    args = p.parse_args(argv)
    if args.cmd == "train":
        trials = dict(
            zip(("lightgbm", "xgboost", "catboost"), args.trials or (60, 50, 30), strict=True)
        )
        cfg = Config(
            trials=trials,
            n_boot=args.n_boot,
            reuse_params=args.reuse_params,
        )
        if args.dataset == "baf":
            task = baf_task(load_baf(args.data_dir, args.variant), f"BAF {args.variant}")
        elif args.dataset == "ulb":
            task = ulb_task(load_ulb(args.data_dir))
        else:
            task = baf_task(make_synthetic_baf(), "synthetic")
        m = run_experiment(task, args.artifacts, cfg)
        print(json.dumps({k: m[k] for k in ("task", "deployed", "best_on_cal", "seconds")}))
    elif args.cmd == "variants":
        params = json.loads(Path(args.params).read_text())["lightgbm"]["params"]
        res = variant_sweep(args.data_dir, params, BAF_VARIANTS, Config())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    else:
        from .report import build_report

        print(build_report(args.artifacts, args.docs_dir, args.reports_dir, args.top_n))


if __name__ == "__main__":
    main()
