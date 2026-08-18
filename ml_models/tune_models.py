"""
tune_models.py
==============
Optuna hyperparameter search for the credit and AML models. Fits a baseline with
the default parameters, searches a wide space, and promotes the tuned model only if
it beats that baseline on validation. Writes comparison figures and a report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ARTIFACT_DIR  # noqa: E402

FIGURE_DIR = _REPO_ROOT / "reports" / "figures"
REPORT_DIR = _REPO_ROOT / "reports" / "tuning"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def suggest_params(trial, imbalanced: bool) -> dict:
    """
    One wide XGBoost space, shared by both models.

    Ranges are intentionally broader than the training defaults: a search that
    only explores near the hand-picked values can confirm them but never
    overturn them, which makes it decoration rather than an experiment.

    `max_delta_step` is included only for the imbalanced case. It caps how far
    a single boosting step can move a leaf, which is the standard remedy when
    `scale_pos_weight` is large enough to make updates unstable -- exactly the
    AML situation at 200:1.
    """
    # The learning-rate floor is 0.02, not something smaller, because fit time
    # scales roughly inversely with it: measured on the credit split, lr=0.005
    # ran 77s and stopped at iteration 1099, while lr=0.05 ran 9.6s and stopped
    # at 111 -- for a validation F1 within a thousandth. With early stopping
    # choosing the iteration count anyway, a lower rate buys almost nothing and
    # costs most of the search budget.
    space = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    if imbalanced:
        space["max_delta_step"] = trial.suggest_int("max_delta_step", 0, 10)
    return space


def search_subset(X, y, max_rows: int, seed: int = 42):
    """
    Stratified subsample used ONLY inside the search.

    XGBoost evaluates the eval_set after every boosting round, so a 720k-row
    validation split is re-scored up to 1,500 times per trial -- that single
    cost dominates AML tuning. The search only needs to rank configurations
    against each other, and a stratified sample preserves that ranking. Every
    number that gets reported comes from a refit scored on the FULL validation
    split, so nothing published rests on the subsample.

    All positives are kept: at 0.06% prevalence, sampling them would leave too
    few to measure against.
    """
    if len(y) <= max_rows:
        return X, y
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_neg = max(max_rows - len(pos), 1)
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([pos, rng.choice(neg, size=min(keep_neg, len(neg)),
                                                   replace=False)]))
    return X.iloc[keep], y[keep]


def run_study(objective, n_trials: int, timeout: int | None, direction: str = "maximize"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # No pruner is configured. Optuna pruning requires the objective to report
    # intermediate values via trial.report(), which a single-fit objective cannot
    # do. Speed comes instead from bounding the search space and shrinking the
    # data used during the search.
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )
    t0 = time.time()
    reported = {"n": 0}

    def callback(study_, trial_):
        reported["n"] += 1
        n = reported["n"]
        if n % 5 == 0 or n == 1:
            elapsed = time.time() - t0
            eta = elapsed / n * (n_trials - n)
            print(f"    trial {n:>4}/{n_trials}  best {study_.best_value:.4f}  "
                  f"({elapsed/60:.1f} min elapsed, ~{eta/60:.0f} min left)", flush=True)

    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[callback], show_progress_bar=False)
    return study


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def write_tuning_figures(study, baseline_score: float, label: str,
                         objective_name: str) -> list[str]:
    """Optuna's own diagnostics plus a baseline-versus-tuned comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as ovm

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(name):
        path = FIGURE_DIR / name
        plt.gcf().set_size_inches(8, 5)
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close("all")
        written.append(name)

    values = [t.value for t in study.trials if t.value is not None]

    # --- history with the baseline drawn in --------------------------------
    # Optuna's own history plot has no notion of "what we had before", which is
    # the only line that answers whether the search was worth running.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(values)), values, s=20, alpha=0.6, label="trial")
    ax.plot(np.maximum.accumulate(values), color="C1", lw=2, label="best so far")
    ax.axhline(baseline_score, ls="--", color="grey",
               label=f"baseline {baseline_score:.4f}")
    ax.set_xlabel("trial"); ax.set_ylabel(objective_name)
    ax.set_title(f"{label} — search progress vs baseline"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR / f"tuning_{label}_history.png", dpi=200)
    plt.close(fig); written.append(f"tuning_{label}_history.png")

    for plot_fn, name in (
        (ovm.plot_param_importances, f"tuning_{label}_param_importance.png"),
        (ovm.plot_parallel_coordinate, f"tuning_{label}_parallel.png"),
        (ovm.plot_slice, f"tuning_{label}_slice.png"),
    ):
        try:
            plot_fn(study)
            save(name)
        except Exception as exc:
            print(f"    ({name} skipped: {exc})")
            plt.close("all")

    return written


def comparison_figure(y_val, base_proba, tuned_proba, label: str,
                      primary: str) -> str:
    """Baseline and tuned on the same PR and ROC axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, average_precision_score,
                                 precision_recall_curve, roc_curve)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, scores in (("baseline", base_proba), ("tuned", tuned_proba)):
        precision, recall, _ = precision_recall_curve(y_val, scores)
        axes[0].plot(recall, precision,
                     label=f"{name} — AP {average_precision_score(y_val, scores):.4f}")
        fpr, tpr, _ = roc_curve(y_val, scores)
        axes[1].plot(fpr, tpr, label=f"{name} — AUC {auc(fpr, tpr):.4f}")
    axes[0].axhline(y_val.mean(), ls="--", lw=1, color="grey", label="prevalence")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
    axes[0].set_title(f"{label} — precision/recall ({primary})"); axes[0].legend()
    axes[1].plot([0, 1], [0, 1], "--", lw=1, color="grey")
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
    axes[1].set_title(f"{label} — ROC"); axes[1].legend()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"tuning_{label}_baseline_vs_tuned.png"
    fig.tight_layout(); fig.savefig(FIGURE_DIR / name, dpi=200); plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# Credit
# ---------------------------------------------------------------------------
def tune_credit(trials: int, timeout: int | None, always_promote: bool,
                search_val_rows: int) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    from xgboost import XGBClassifier

    from ml_models.data_splits import get_split
    from ml_models.train_model import DEFAULT_PARAMS, encode, tune_threshold

    print("CREDIT MODEL", flush=True)
    X_tr, y_tr = encode(get_split("train"))
    X_val, y_val = encode(get_split("val"))
    X_val = X_val.reindex(columns=X_tr.columns, fill_value=0)
    y_tr, y_val = y_tr.to_numpy(), y_val.to_numpy()
    print(f"  train {X_tr.shape[0]:,} x {X_tr.shape[1]} | val {X_val.shape[0]:,}", flush=True)

    fixed = {"n_estimators": 1500, "early_stopping_rounds": 40,
             "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}

    print("  fitting baseline (current default parameters) ...", flush=True)
    baseline = XGBClassifier(**DEFAULT_PARAMS)
    baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    base_proba = baseline.predict_proba(X_val)[:, 1]
    base_threshold, base_f1 = tune_threshold(y_val, base_proba)
    base_auc = float(roc_auc_score(y_val, base_proba))
    print(f"  baseline val F1 {base_f1:.4f} | AUC {base_auc:.4f}", flush=True)

    Xs, ys = search_subset(X_val, y_val, search_val_rows)
    if len(ys) < len(y_val):
        print(f"  search scored on {len(ys):,} of {len(y_val):,} val rows "
              f"(final refit uses all)", flush=True)

    def objective(trial):
        params = {**fixed, **suggest_params(trial, imbalanced=False)}
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(Xs, ys)], verbose=False)
        # Scored at its own best threshold, not a fixed 0.5. Otherwise the
        # search optimises for a cutoff the production system does not use.
        _, f1 = tune_threshold(ys, model.predict_proba(Xs)[:, 1])
        return f1

    print(f"\n  searching {trials} trials (objective: val F1 at best threshold) ...",
          flush=True)
    study = run_study(objective, trials, timeout)

    best_params = {**fixed, **study.best_params}
    print(f"\n  refitting with best parameters ...", flush=True)
    tuned = XGBClassifier(**best_params)
    tuned.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    tuned_proba = tuned.predict_proba(X_val)[:, 1]
    tuned_threshold, tuned_f1 = tune_threshold(y_val, tuned_proba)
    tuned_auc = float(roc_auc_score(y_val, tuned_proba))

    improved = tuned_f1 > base_f1
    promote = improved or always_promote
    print(f"  tuned val F1 {tuned_f1:.4f} | AUC {tuned_auc:.4f} "
          f"({'+' if improved else ''}{tuned_f1 - base_f1:.4f} vs baseline)")
    print(f"  {'PROMOTING tuned model' if promote else 'KEEPING baseline — tuning did not improve'}\n",
          flush=True)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, ARTIFACT_DIR / "xgboost_credit_model_baseline.joblib")
    joblib.dump(base_threshold, ARTIFACT_DIR / "best_threshold_baseline.joblib")
    winner, win_threshold = (tuned, tuned_threshold) if promote else (baseline, base_threshold)
    joblib.dump(winner, ARTIFACT_DIR / "xgboost_credit_model.joblib")
    joblib.dump(list(X_tr.columns), ARTIFACT_DIR / "feature_columns.joblib")
    joblib.dump(win_threshold, ARTIFACT_DIR / "best_threshold.joblib")

    figures = write_tuning_figures(study, base_f1, "credit", "validation F1")
    figures.append(comparison_figure(y_val, base_proba, tuned_proba, "credit", "F1"))

    return {
        "model": "credit", "objective": "validation F1 at best threshold",
        "n_trials": len(study.trials), "promoted": promote, "improved": improved,
        "baseline": {"val_f1": base_f1, "val_roc_auc": base_auc,
                     "threshold": base_threshold,
                     "params": {k: v for k, v in DEFAULT_PARAMS.items() if k != "n_jobs"}},
        "tuned": {"val_f1": tuned_f1, "val_roc_auc": tuned_auc,
                  "threshold": tuned_threshold,
                  "params": {k: v for k, v in best_params.items() if k != "n_jobs"}},
        "history": [{"trial": t.number, "value": t.value, "params": t.params}
                    for t in study.trials if t.value is not None],
        "figures": figures,
        "metrics_path": ARTIFACT_DIR / "training_metrics.json",
    }


# ---------------------------------------------------------------------------
# AML
# ---------------------------------------------------------------------------
def tune_aml(trials: int, timeout: int | None, always_promote: bool,
             neg_per_pos: int, search_val_rows: int, search_neg_per_pos: int) -> dict:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    from ml_models.aml_features import load_cached
    from ml_models.train_aml_model import downsample_negatives, threshold_sweep

    print("AML MODEL", flush=True)
    X_tr_full, meta_tr = load_cached("train")
    X_val, meta_val = load_cached("val")
    y_tr_full = meta_tr["is_laundering"].to_numpy()
    y_val = meta_val["is_laundering"].to_numpy()
    X_tr, y_tr, n_pos, n_neg = downsample_negatives(X_tr_full, y_tr_full, neg_per_pos)
    del X_tr_full
    print(f"  train {len(y_tr):,} rows ({n_pos:,} positive, {neg_per_pos}:1) "
          f"| val {len(y_val):,} rows, {int(y_val.sum()):,} positives", flush=True)

    fixed = {"n_estimators": 1500, "early_stopping_rounds": 40,
             "eval_metric": "aucpr", "random_state": 42, "n_jobs": -1,
             "scale_pos_weight": n_neg / max(n_pos, 1)}

    from ml_models.train_aml_model import DEFAULT_PARAMS as AML_DEFAULTS
    base_params = dict(AML_DEFAULTS)
    base_params["scale_pos_weight"] = fixed["scale_pos_weight"]

    print("  fitting baseline (current default parameters) ...", flush=True)
    baseline = XGBClassifier(**base_params)
    baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    base_proba = baseline.predict_proba(X_val)[:, 1]
    base_pr = float(average_precision_score(y_val, base_proba))
    base_sweep = threshold_sweep(y_val, base_proba, meta_val["customer_id"].nunique())
    base_best = base_sweep.loc[base_sweep["f2"].idxmax()]
    print(f"  baseline val PR-AUC {base_pr:.4f} | F2 {base_best['f2']:.4f}", flush=True)

    # Two reductions for the search only: fewer training rows (a lighter
    # downsample ratio) and a subsampled eval set. The final model is refit at
    # the full ratio and scored on the whole validation split.
    Xts, yts, _, _ = downsample_negatives(X_tr, y_tr, search_neg_per_pos) \
        if search_neg_per_pos < neg_per_pos else (X_tr, y_tr, 0, 0)
    Xs, ys = search_subset(X_val, y_val, search_val_rows)
    print(f"  search fits on {len(yts):,} rows, scores on {len(ys):,} of "
          f"{len(y_val):,} val rows (final refit uses all)", flush=True)

    def objective(trial):
        params = {**fixed, **suggest_params(trial, imbalanced=True)}
        model = XGBClassifier(**params)
        model.fit(Xts, yts, eval_set=[(Xs, ys)], verbose=False)
        # PR-AUC, not F2 at a cutoff: it summarises every operating point, so
        # the search is not chasing one threshold's luck.
        return float(average_precision_score(ys, model.predict_proba(Xs)[:, 1]))

    print(f"\n  searching {trials} trials (objective: val PR-AUC) ...", flush=True)
    study = run_study(objective, trials, timeout)

    best_params = {**fixed, **study.best_params}
    print(f"\n  refitting with best parameters ...", flush=True)
    tuned = XGBClassifier(**best_params)
    tuned.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    tuned_proba = tuned.predict_proba(X_val)[:, 1]
    tuned_pr = float(average_precision_score(y_val, tuned_proba))
    tuned_sweep = threshold_sweep(y_val, tuned_proba, meta_val["customer_id"].nunique())
    tuned_best = tuned_sweep.loc[tuned_sweep["f2"].idxmax()]

    improved = tuned_pr > base_pr
    promote = improved or always_promote
    print(f"  tuned val PR-AUC {tuned_pr:.4f} | F2 {tuned_best['f2']:.4f} "
          f"({'+' if improved else ''}{tuned_pr - base_pr:.4f} vs baseline)")
    print(f"  {'PROMOTING tuned model' if promote else 'KEEPING baseline — tuning did not improve'}\n",
          flush=True)

    joblib.dump(baseline, ARTIFACT_DIR / "aml_xgboost_model_baseline.joblib")
    joblib.dump(float(base_best["threshold"]), ARTIFACT_DIR / "aml_threshold_baseline.joblib")
    winner = tuned if promote else baseline
    win_row = tuned_best if promote else base_best
    joblib.dump(winner, ARTIFACT_DIR / "aml_xgboost_model.joblib")
    joblib.dump(list(X_val.columns), ARTIFACT_DIR / "aml_feature_columns.joblib")
    joblib.dump(float(win_row["threshold"]), ARTIFACT_DIR / "aml_threshold.joblib")
    (tuned_sweep if promote else base_sweep).to_csv(
        ARTIFACT_DIR / "aml_threshold_sweep.csv", index=False)

    figures = write_tuning_figures(study, base_pr, "aml", "validation PR-AUC")
    figures.append(comparison_figure(y_val, base_proba, tuned_proba, "aml", "PR-AUC"))

    return {
        "model": "aml", "objective": "validation PR-AUC",
        "n_trials": len(study.trials), "promoted": promote, "improved": improved,
        "baseline": {"val_pr_auc": base_pr, "val_f2": float(base_best["f2"]),
                     "val_precision": float(base_best["precision"]),
                     "val_recall": float(base_best["recall"]),
                     "threshold": float(base_best["threshold"]),
                     "alerts_per_day": float(base_best["alerts_per_day_portfolio"]),
                     "params": {k: v for k, v in base_params.items() if k != "n_jobs"}},
        "tuned": {"val_pr_auc": tuned_pr, "val_f2": float(tuned_best["f2"]),
                  "val_precision": float(tuned_best["precision"]),
                  "val_recall": float(tuned_best["recall"]),
                  "threshold": float(tuned_best["threshold"]),
                  "alerts_per_day": float(tuned_best["alerts_per_day_portfolio"]),
                  "params": {k: v for k, v in best_params.items() if k != "n_jobs"}},
        "history": [{"trial": t.number, "value": t.value, "params": t.params}
                    for t in study.trials if t.value is not None],
        "figures": figures,
        "metrics_path": ARTIFACT_DIR / "aml_training_metrics.json",
    }


# ---------------------------------------------------------------------------
def write_report(results: list[dict], started: datetime, stamp: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = ["# Hyperparameter Tuning", "",
           f"Generated {started:%Y-%m-%d %H:%M} UTC.", "",
           "Search uses Optuna's TPE sampler over a deliberately wide space. An "
           "exhaustive grid is not attainable over continuous hyperparameters — "
           "ten points on each of four continuous dimensions, crossed with the "
           "discrete ones, is over a million fits — so the honest description is "
           "a wide guided search, not an exhaustive one. The discrete ranges "
           "(`max_depth` 3–12, `min_child_weight` 1–20) do cover every sensible "
           "value.", "",
           "The tuned model is promoted only if it beats the baseline on "
           "validation. Where it does not, the baseline remains the production "
           "artifact and the table below records that.", ""]

    for r in results:
        base, tuned = r["baseline"], r["tuned"]
        keys = [k for k in base if k != "params"]
        doc += [f"## {r['model'].upper()}", "",
                f"- Objective: **{r['objective']}**",
                f"- Trials completed: **{r['n_trials']}**",
                f"- Outcome: **{'tuned model promoted' if r['promoted'] else 'baseline retained'}**"
                + ("" if r["improved"] else " — the search did not beat the defaults"), "",
                "| Metric | Baseline | Tuned |", "|---|---|---|"]
        for k in keys:
            bv, tv = base[k], tuned[k]
            fmt = (lambda v: f"{v:.4f}") if isinstance(bv, float) else str
            doc.append(f"| {k} | {fmt(bv)} | {fmt(tv)} |")
        doc += ["", "### Best parameters", "", "```json",
                json.dumps(tuned["params"], indent=2), "```", "",
                "### Figures", ""]
        doc += [f"![{f}](../figures/{f})" for f in r["figures"]]
        doc.append("")

    path = REPORT_DIR / f"TUNING_{stamp}.md"
    path.write_text("\n".join(doc), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["credit", "aml", "both"], default="both")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None,
                        help="stop the search after N seconds regardless of trials")
    parser.add_argument("--neg-per-pos", type=int, default=200,
                        help="AML negative downsampling ratio")
    parser.add_argument("--search-val-rows", type=int, default=120_000,
                        help="validation rows used during the search; the final "
                             "refit is always scored on the full split")
    parser.add_argument("--search-neg-per-pos", type=int, default=50,
                        help="lighter AML downsampling during the search only")
    parser.add_argument("--fast", action="store_true",
                        help="30 trials on a smaller search set — minutes, not hours")
    parser.add_argument("--always-promote", action="store_true",
                        help="save the tuned model even if it loses to the baseline")
    args = parser.parse_args()
    if args.fast:
        args.trials = min(args.trials, 30)
        args.search_val_rows = min(args.search_val_rows, 50_000)
        args.search_neg_per_pos = min(args.search_neg_per_pos, 25)

    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S")
    print(f"FRAML — hyperparameter search ({args.trials} trials"
          + (f", {args.timeout}s cap" if args.timeout else "") + ")\n")

    results = []
    if args.model in ("credit", "both"):
        results.append(tune_credit(args.trials, args.timeout, args.always_promote,
                                   args.search_val_rows))
    if args.model in ("aml", "both"):
        results.append(tune_aml(args.trials, args.timeout, args.always_promote,
                                args.neg_per_pos, args.search_val_rows,
                                args.search_neg_per_pos))

    # Fold the search into the metrics files the evaluate scripts already read,
    # so the comparison figures pick it up with no further wiring.
    for r in results:
        path = r["metrics_path"]
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update({
            "tuned": r["promoted"],
            "tuning_objective": r["objective"],
            "tuning_trials": r["n_trials"],
            "tuning_improved": r["improved"],
            "baseline": r["baseline"],
            "tuning_history": r["history"],
            "tuned_at": started.isoformat(),
        })
        if r["promoted"]:
            existing["params"] = r["tuned"]["params"]
            existing["threshold"] = r["tuned"]["threshold"]
        path.write_text(json.dumps(existing, indent=2))

    report = write_report(results, started, stamp)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    print(f"Completed in {elapsed / 60:.1f} minutes")
    print(f"  report  : {report}")
    print(f"  figures : {FIGURE_DIR}")
    for r in results:
        verdict = "tuned promoted" if r["promoted"] else "baseline retained"
        print(f"  {r['model']:<7}: {verdict}")
    print("\nNext: python -m ml_models.evaluate_model")
    print("      python -m ml_models.evaluate_aml_model")


if __name__ == "__main__":
    main()
