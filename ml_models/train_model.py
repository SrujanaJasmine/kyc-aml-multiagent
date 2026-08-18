"""
train_model.py
==============
Trains the credit model on the `train` split and tunes its decision threshold on
`validation`, then saves the model, the feature contract, the threshold and a
provenance record to ml_models/artifacts/.
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
from ml_models.credit_risk_model import sanitize_columns  # noqa: E402
from ml_models.data_splits import LABEL_COL, get_split  # noqa: E402

DROP_COLS = ["ID", "Customer_ID", "Name", "SSN"]
ENCODE_COLS = ["Month", "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

# Reasonable starting point in the region the notebook's Optuna study explored.
# Pass --tune to search properly.
DEFAULT_PARAMS = {
    "n_estimators": 2000,          # upper cap; early stopping finds the real number
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reproduce the notebook's encoding exactly.

    `drop_first=True` is kept here because at training time we have every
    category present and dropping a reference level avoids collinearity. The
    inference path in CreditScorer uses drop_first=False plus a reindex, which
    lands on the same columns from the other direction — a single row cannot
    safely drop its own only category.
    """
    frame = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    y = frame[LABEL_COL]
    X = frame.drop(columns=[LABEL_COL])
    encode_here = [c for c in ENCODE_COLS if c in X.columns]
    X = pd.get_dummies(X, columns=encode_here, drop_first=True)
    # Clean identifiers from here on: the saved feature_columns.joblib is the
    # contract the database schema now mirrors, so it must not contain names
    # that need quoting in SQL.
    X = sanitize_columns(X)
    return X, y


def tune_threshold(y_true, proba) -> tuple[float, float]:
    """
    Pick the cutoff maximising F1 on validation.

    Not 0.5: the classes are imbalanced, so the default cutoff systematically
    under-flags the minority class, which here is exactly the population the
    bank cares about finding.
    """
    from sklearn.metrics import f1_score
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def optuna_search(X_tr, y_tr, X_val, y_val, n_trials: int) -> tuple[dict, list[dict]]:
    """
    Search, and return the trial history alongside the winning parameters.

    The history is what makes a before/after figure possible: without the
    per-trial scores there is nothing to plot except two bars, which tells a
    reader that tuning happened but not whether it converged or got lucky.
    """
    import optuna
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 2.0, log=True),
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return f1_score(y_val, model.predict(X_val), zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  best val F1 during search: {study.best_value:.4f}")

    history = [{"trial": t.number, "value": t.value, "params": t.params}
               for t in study.trials if t.value is not None]

    params = dict(DEFAULT_PARAMS)
    params.update(study.best_params)
    return params, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit model on the train split")
    parser.add_argument("--tune", type=int, default=0,
                        help="run N Optuna trials before the final fit (needs optuna)")
    args = parser.parse_args()

    from sklearn.metrics import f1_score, roc_auc_score
    from xgboost import XGBClassifier

    train_df, val_df = get_split("train"), get_split("val")
    X_tr, y_tr = encode(train_df)
    X_val, y_val = encode(val_df)

    # Validation must be described in the training feature space, not its own —
    # a category absent from val would otherwise shift every column position.
    X_val = X_val.reindex(columns=X_tr.columns, fill_value=0)

    print(f"train: {X_tr.shape[0]:,} rows x {X_tr.shape[1]} features "
          f"(positive rate {y_tr.mean():.3f})")
    print(f"val  : {X_val.shape[0]:,} rows (positive rate {y_val.mean():.3f})")

    params = dict(DEFAULT_PARAMS)
    history: list[dict] = []
    baseline_metrics = None

    if args.tune:
        # Fit the default-parameter model FIRST and keep it. Without a saved
        # baseline there is nothing to compare the tuned model against later --
        # the evaluation would be reduced to quoting the tuned number and
        # asserting it improved on something no longer on disk.
        print("\nFitting baseline (default parameters) for comparison...")
        baseline = XGBClassifier(**DEFAULT_PARAMS)
        baseline.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        base_proba = baseline.predict_proba(X_val)[:, 1]
        base_threshold, base_f1 = tune_threshold(y_val, base_proba)
        baseline_metrics = {
            "threshold": base_threshold, "val_f1": base_f1,
            "val_roc_auc": float(roc_auc_score(y_val, base_proba)),
            "best_iteration": int(getattr(baseline, "best_iteration", 0) or 0),
            "params": {k: v for k, v in DEFAULT_PARAMS.items() if k != "n_jobs"},
        }
        joblib.dump(baseline, ARTIFACT_DIR / "xgboost_credit_model_baseline.joblib")
        joblib.dump(base_threshold, ARTIFACT_DIR / "best_threshold_baseline.joblib")
        print(f"  baseline val F1 {base_f1:.4f} | AUC {baseline_metrics['val_roc_auc']:.4f}")

        print(f"\nRunning {args.tune} Optuna trials...")
        params, history = optuna_search(X_tr, y_tr, X_val, y_val, args.tune)

    print("\nFitting...")
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = getattr(model, "best_iteration", None)
    print(f"  stopped at iteration {best_iter}" if best_iter is not None else "  fitted")

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = tune_threshold(y_val, val_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    train_f1 = f1_score(y_tr, (model.predict_proba(X_tr)[:, 1] >= threshold).astype(int),
                        zero_division=0)

    print(f"\n  threshold (F1-optimal on val): {threshold:.3f}")
    print(f"  val  F1 {val_f1:.4f} | val AUC {val_auc:.4f}")
    print(f"  train F1 {train_f1:.4f}")
    gap = train_f1 - val_f1
    print(f"  train-val F1 gap: {gap:+.4f}"
          + ("  <- large gap suggests overfitting" if gap > 0.05 else ""))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "xgboost_credit_model.joblib")
    joblib.dump(list(X_tr.columns), ARTIFACT_DIR / "feature_columns.joblib")
    joblib.dump(threshold, ARTIFACT_DIR / "best_threshold.joblib")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": "data/splits/train.csv",
        "validated_on": "data/splits/val.csv",
        "n_train_rows": int(X_tr.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "threshold": threshold,
        "val_f1": val_f1,
        "val_roc_auc": float(val_auc),
        "train_f1": train_f1,
        "params": {k: v for k, v in params.items() if k != "n_jobs"},
        "tuned": bool(args.tune),
        "baseline": baseline_metrics,
        "tuning_history": history,
    }
    (ARTIFACT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    print("  xgboost_credit_model.joblib / feature_columns.joblib / "
          "best_threshold.joblib / training_metrics.json")
    print("\nNext: python -m ml_models.evaluate_model")


if __name__ == "__main__":
    main()
