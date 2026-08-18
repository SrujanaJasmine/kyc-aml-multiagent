"""
train_aml_model.py
==================
Trains the AML model on the `train` split using negative downsampling, then tunes
the decision threshold on the full validation split for F2 and reports the alert
volume each cutoff would generate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ARTIFACT_DIR  # noqa: E402
from ml_models.aml_features import load_cached  # noqa: E402

MODEL_PATH = ARTIFACT_DIR / "aml_xgboost_model.joblib"
COLUMNS_PATH = ARTIFACT_DIR / "aml_feature_columns.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "aml_threshold.joblib"
METRICS_PATH = ARTIFACT_DIR / "aml_training_metrics.json"

DEFAULT_NEG_PER_POS = 200
SIMULATION_DAYS = 182
TOTAL_CUSTOMERS = 12_500

DEFAULT_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    # aucpr, not auc: with 0.068% positives the ROC curve is dominated by the
    # negative class and barely moves, while precision-recall responds to what
    # we actually care about.
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def downsample_negatives(X: pd.DataFrame, y: np.ndarray, neg_per_pos: int, seed: int = 42):
    """Keep every positive, sample `neg_per_pos` negatives for each."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    keep_n = min(len(neg_idx), len(pos_idx) * neg_per_pos)
    sampled = rng.choice(neg_idx, size=keep_n, replace=False)
    keep = np.sort(np.concatenate([pos_idx, sampled]))
    return X.iloc[keep], y[keep], len(pos_idx), keep_n


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, customers_in_split: int) -> pd.DataFrame:
    """
    Precision, recall, F1, F2 and alert volume across candidate cutoffs.

    Alert volume is scaled to the full 12,500-customer book so the number means
    something operationally: the validation split holds only ~15% of customers,
    and 40 alerts a day there is 265 a day across the portfolio.
    """
    scale = TOTAL_CUSTOMERS / max(customers_in_split, 1)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": fbeta(precision, recall, 1), "f2": fbeta(precision, recall, 2),
            "alerts_per_day_portfolio": (tp + fp) / SIMULATION_DAYS * scale,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neg-per-pos", type=int, default=DEFAULT_NEG_PER_POS,
                        help="negatives kept per positive in training (default 200)")
    parser.add_argument("--tune", type=int, default=0, help="Optuna trials (needs optuna)")
    args = parser.parse_args()

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    print("loading cached features ...", flush=True)
    X_tr_full, meta_tr = load_cached("train")
    X_val, meta_val = load_cached("val")
    y_tr_full = meta_tr["is_laundering"].to_numpy()
    y_val = meta_val["is_laundering"].to_numpy()

    print(f"  train {X_tr_full.shape[0]:,} rows, {int(y_tr_full.sum()):,} positives "
          f"({100*y_tr_full.mean():.4f}%)")
    print(f"  val   {X_val.shape[0]:,} rows, {int(y_val.sum()):,} positives "
          f"({100*y_val.mean():.4f}%)")

    X_tr, y_tr, n_pos, n_neg = downsample_negatives(X_tr_full, y_tr_full, args.neg_per_pos)
    del X_tr_full
    print(f"\ndownsampled train: {len(y_tr):,} rows "
          f"({n_pos:,} positive + {n_neg:,} negative, {args.neg_per_pos}:1)")

    params = dict(DEFAULT_PARAMS)
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    history: list[dict] = []
    baseline_metrics = None

    if args.tune:
        import optuna
        from sklearn.metrics import average_precision_score as aps
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print(f"\nrunning {args.tune} Optuna trials (objective: val PR-AUC) ...", flush=True)

        def objective(trial):
            p = dict(params)
            p.update({
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            })
            m = XGBClassifier(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return aps(y_val, m.predict_proba(X_val)[:, 1])

        # Baseline first, and kept: the tuned model needs something on disk to
        # be compared against, or the "after" number stands alone.
        print("  fitting baseline (default parameters) for comparison ...", flush=True)
        baseline = XGBClassifier(**params)
        baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        base_proba = baseline.predict_proba(X_val)[:, 1]
        base_sweep = threshold_sweep(y_val, base_proba, meta_val["customer_id"].nunique())
        base_best = base_sweep.loc[base_sweep["f2"].idxmax()]
        baseline_metrics = {
            "threshold": float(base_best["threshold"]),
            "val_pr_auc": float(aps(y_val, base_proba)),
            "val_precision": float(base_best["precision"]),
            "val_recall": float(base_best["recall"]),
            "val_f1": float(base_best["f1"]), "val_f2": float(base_best["f2"]),
            "params": {k: v for k, v in params.items() if k != "n_jobs"},
        }
        joblib.dump(baseline, ARTIFACT_DIR / "aml_xgboost_model_baseline.joblib")
        joblib.dump(float(base_best["threshold"]), ARTIFACT_DIR / "aml_threshold_baseline.joblib")
        print(f"  baseline val PR-AUC {baseline_metrics['val_pr_auc']:.4f} "
              f"| F2 {baseline_metrics['val_f2']:.4f}")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.tune, show_progress_bar=False)
        params.update(study.best_params)
        history = [{"trial": t.number, "value": t.value, "params": t.params}
                   for t in study.trials if t.value is not None]
        print(f"  best val PR-AUC {study.best_value:.4f}")

    print("\nfitting ...", flush=True)
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    print(f"  stopped at iteration {best_iter}")

    val_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_proba)
    print(f"  val PR-AUC {pr_auc:.4f}")

    customers_val = meta_val["customer_id"].nunique()
    sweep = threshold_sweep(y_val, val_proba, customers_val)

    print(f"\nthreshold sweep on val ({customers_val:,} customers, {SIMULATION_DAYS} days):")
    print(f"  {'thr':>5}{'prec':>9}{'recall':>9}{'F1':>8}{'F2':>8}{'alerts/day':>13}")
    for _, r in sweep.iterrows():
        print(f"  {r['threshold']:>5.2f}{r['precision']:>9.3f}{r['recall']:>9.3f}"
              f"{r['f1']:>8.3f}{r['f2']:>8.3f}{r['alerts_per_day_portfolio']:>13.1f}")

    best = sweep.loc[sweep["f2"].idxmax()]
    threshold = float(best["threshold"])
    print(f"\nF2-optimal threshold: {threshold:.2f}")
    print(f"  precision {best['precision']:.3f} | recall {best['recall']:.3f} | "
          f"F2 {best['f2']:.3f}")
    print(f"  ~{best['alerts_per_day_portfolio']:.0f} alerts/day across "
          f"{TOTAL_CUSTOMERS:,} customers")
    if best["alerts_per_day_portfolio"] > 200:
        print("  NOTE: that alert volume is likely beyond a small compliance team. "
              "Consider a higher threshold from the sweep above -- the trade is "
              "explicit and yours to make.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X_val.columns), COLUMNS_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": "data/splits (train split of simulated transactions)",
        "n_train_rows_after_downsample": int(len(y_tr)),
        "n_train_positives": int(n_pos),
        "neg_per_pos": args.neg_per_pos,
        "n_val_rows": int(len(y_val)),
        "n_val_positives": int(y_val.sum()),
        "n_features": int(X_val.shape[1]),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "val_pr_auc": float(pr_auc),
        "threshold": threshold,
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
        "val_f1": float(best["f1"]),
        "val_f2": float(best["f2"]),
        "alerts_per_day_portfolio": float(best["alerts_per_day_portfolio"]),
        "params": {k: v for k, v in params.items() if k != "n_jobs"},
        "tuned": bool(args.tune),
        "baseline": baseline_metrics,
        "tuning_history": history,
        "caveat": ("Trained on simulated data with typologies injected by our own "
                   "generator. High scores demonstrate the pipeline works end to end; "
                   "they are not evidence of real-world detection performance."),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    sweep.to_csv(ARTIFACT_DIR / "aml_threshold_sweep.csv", index=False)

    print(f"\nartifacts written to {ARTIFACT_DIR}")
    print("  aml_xgboost_model.joblib / aml_feature_columns.joblib / "
          "aml_threshold.joblib / aml_training_metrics.json / aml_threshold_sweep.csv")
    print("\nNext: python -m ml_models.evaluate_aml_model")


if __name__ == "__main__":
    main()
