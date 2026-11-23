"""Rainy-season onset stopping rule (woreda-level dry-spell detector)."""

from __future__ import annotations

import argparse
import json
import os
import pickle


def _n_jobs(default=-1):
    return int(os.environ.get("SLURM_CPUS_PER_TASK", default))


def cmd_aggregate(args):
    from . import chirps

    df = chirps.aggregate()
    print(f"CHIRPS woreda series: {df.shape} -> saved")


def cmd_check_leads(args):
    from . import aifs, chirps

    print(aifs.check_lead_alignment(chirps.load_woreda_series()).round(3))


def cmd_features(args):
    from . import features

    base = features.build_base()
    r = base["rows"]
    print(f"{len(r)} wet-spell candidates; {r.pre_onset.sum()} on/before true onset")
    print(f"base rate all={r.y.mean():.3f} pre-onset={r.y[r.pre_onset].mean():.3f}")


def cmd_train(args):
    from . import config as C
    from . import features, model

    base = features.load_base()
    years = C.MODEL_YEARS if not args.years else [int(y) for y in args.years.split(",")]
    oof, info, coefs = model.cross_validate(base, years=years, n_jobs=_n_jobs())
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    oof.to_parquet(C.OUTPUT_DIR / "oof_predictions.parquet")
    info.to_csv(C.OUTPUT_DIR / "cv_fold_info.csv", index=False)
    for g, df in coefs.items():
        df.to_csv(C.OUTPUT_DIR / f"coefficients_{g.replace('+', '_')}.csv")
    print(info.groupby("group")[["threshold"]].describe())
    if not args.years:
        m, method, thr, cols = model.fit_final(base)
        with open(C.OUTPUT_DIR / "final_model.pkl", "wb") as f:
            pickle.dump({"model": m, "calibration": method, "threshold": thr,
                         "features": cols, "pcodes": list(base["pcodes"])}, f)
        print(f"final model: calibration={method}, threshold={thr}")


def cmd_evaluate(args):
    import pandas as pd

    from . import config as C
    from . import evaluate as E
    from . import features as F

    oof = pd.read_parquet(C.OUTPUT_DIR / "oof_predictions.parquet")
    metrics = E.evaluate_all(oof)
    if not args.no_bootstrap:
        metrics["bootstrap_95ci"] = E.bootstrap(oof, list(F.GROUPS)).to_dict(orient="index")
    with open(C.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    for k in ["probabilistic_pre_onset", "probabilistic_all_wet_days", "stopping_rule"]:
        print(f"\n== {k}")
        print(pd.DataFrame(metrics[k]).round(3).to_string())
    if "bootstrap_95ci" in metrics:
        print("\n== bootstrap 95% CI")
        print(pd.DataFrame(metrics["bootstrap_95ci"]).T.round(3).to_string())
    if not args.no_plots:
        from . import plots

        plots.make_all(oof)


def cmd_report(args):
    from . import report

    print(f"report -> {report.make_report()}")


def main() -> None:
    p = argparse.ArgumentParser(prog="stopping-rule-2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("aggregate", help="aggregate CHIRPS to woredas")
    sub.add_parser("check-leads", help="verify AIFS lead/day alignment vs CHIRPS")
    sub.add_parser("features", help="build base feature arrays")
    t = sub.add_parser("train", help="leave-one-year-out CV + final model")
    t.add_argument("--years", help="comma-separated subset of test years (smoke test)")
    e = sub.add_parser("evaluate", help="metrics + figures")
    e.add_argument("--no-bootstrap", action="store_true")
    e.add_argument("--no-plots", action="store_true")
    sub.add_parser("report", help="HTML report from the evaluation outputs")
    a = sub.add_parser("all", help="aggregate, features, train, evaluate, report")
    a.set_defaults(years=None, no_bootstrap=False, no_plots=False)
    args = p.parse_args()
    if args.cmd == "all":
        for f in (cmd_aggregate, cmd_features, cmd_train, cmd_evaluate, cmd_report):
            f(args)
    else:
        {"aggregate": cmd_aggregate, "check-leads": cmd_check_leads, "features": cmd_features,
         "train": cmd_train, "evaluate": cmd_evaluate, "report": cmd_report}[args.cmd](args)
