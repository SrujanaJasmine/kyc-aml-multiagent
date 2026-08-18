"""
evaluate_model.py
=================
Evaluates the credit model on the `test` split and writes ROC, precision-recall,
confusion, threshold-sweep, feature-importance and calibration figures to
reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ARTIFACT_DIR  # noqa: E402
from ml_models.credit_risk_model import CreditScorer  # noqa: E402
from ml_models.data_splits import LABEL_COL, get_split  # noqa: E402


def check_provenance() -> None:
    """Warn if the artifacts weren't produced by train_model.py, because then
    the split guarantee doesn't hold and the metrics are inflated."""
    meta_path = ARTIFACT_DIR / "training_metrics.json"
    if not meta_path.exists():
        print("!" * 72)
        print("WARNING: no training_metrics.json found next to the artifacts.")
        print("The current model was probably fit on ALL of cleaned_train.csv,")
        print("which INCLUDES the test rows scored below. Those metrics would be")
        print("memorisation, not generalisation.")
        print("Fix: python -m ml_models.train_model")
        print("!" * 72)
        print()
        return

    meta = json.loads(meta_path.read_text())
    print(f"Artifacts trained {meta.get('trained_at')} on {meta.get('trained_on')}")
    print(f"  {meta.get('n_train_rows'):,} train rows, {meta.get('n_features')} features, "
          f"threshold {meta.get('threshold')}")
    print()



FIGURE_DIR = _REPO_ROOT / "reports" / "figures"
METRICS_PATH = ARTIFACT_DIR / "training_metrics.json"


def write_figures(y, proba, pred, model, columns, threshold, meta_json) -> list[str]:
    """
    Publication figures for the credit model, written to reports/figures/.

    Filenames are stable rather than timestamped so a paper can reference them
    with a fixed path and re-running the evaluation refreshes the same images.

    Where a baseline model exists -- saved by `train_model.py --tune` before the
    Optuna search -- every comparable panel shows both curves. Without it the
    figures still render, just with a single series, because the common case is
    that nobody has tuned yet and a broken figure helps no one.
    """
    import matplotlib
    matplotlib.use("Agg")      # no display on a headless machine or in CI
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                                 f1_score, precision_recall_curve, precision_score,
                                 recall_score, roc_auc_score, roc_curve)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name):
        path = FIGURE_DIR / name
        fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
        written.append(name)

    # --- baseline, if one was saved --------------------------------------
    base_proba = None
    base_path = ARTIFACT_DIR / "xgboost_credit_model_baseline.joblib"
    if base_path.exists():
        try:
            base_model = joblib.load(base_path)
            base_proba = base_model.predict_proba(X_GLOBAL[columns])[:, 1]
        except Exception:
            base_proba = None

    series = [("tuned" if meta_json.get("tuned") else "model", proba)]
    if base_proba is not None:
        series.insert(0, ("baseline (default params)", base_proba))

    # --- ROC ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, scores in series:
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=f"{label} — AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], lw=1, color="grey", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Credit model: ROC"); ax.legend(loc="lower right")
    save(fig, "credit_roc.png")

    # --- precision / recall ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, scores in series:
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision,
                label=f"{label} (Average Precision = {average_precision_score(y, scores):.3f})")
    ax.axhline(y.mean(), lw=1, color="grey",
               label=f"prevalence {y.mean():.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Credit model: precision/recall"); ax.legend(loc="lower left")
    save(fig, "credit_precision_recall.png")

    # --- confusion matrix ---------------------------------------------------
    panels = [("tuned" if meta_json.get("tuned") else "model", pred)]
    if base_proba is not None:
        bt = float(joblib.load(ARTIFACT_DIR / "best_threshold_baseline.joblib")) \
            if (ARTIFACT_DIR / "best_threshold_baseline.joblib").exists() else threshold
        panels.insert(0, ("baseline", (base_proba >= bt).astype(int)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (label, predictions) in zip(np.atleast_1d(axes), panels):
        grid = confusion_matrix(y, predictions, labels=[0, 1])
        ax.imshow(grid, cmap="Blues")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > grid.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred. good", "pred. high risk"])
        ax.set_yticks([0, 1], ["actual good", "actual high risk"])
        ax.set_title(f"Credit: {label}")
    save(fig, "credit_confusion.png")

    # --- threshold sweep ----------------------------------------------------
    # The chosen cutoff is a decision, not a property of the model, so showing
    # the whole curve lets a reader see what was traded away to get it.
    grid = np.arange(0.05, 0.96, 0.01)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, fn in (("precision", precision_score), ("recall", recall_score),
                     ("F1", f1_score)):
        ax.plot(grid, [fn(y, (proba >= t).astype(int), zero_division=0) for t in grid],
                label=name)
    ax.axvline(threshold, ls="--", color="grey", label=f"chosen {threshold:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Credit model: threshold sweep"); ax.legend()
    save(fig, "credit_threshold_sweep.png")

    # --- Optuna history -----------------------------------------------------
    history = meta_json.get("tuning_history") or []
    if history:
        values = [h["value"] for h in history]
        running = np.maximum.accumulate(values)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(range(len(values)), values, s=18, alpha=0.6, label="trial")
        ax.plot(running, color="C1", label="best so far")
        if meta_json.get("baseline"):
            ax.axhline(meta_json["baseline"]["val_f1"], ls="--", color="grey",
                       label=f"baseline {meta_json['baseline']['val_f1']:.3f}")
        ax.set_xlabel("Optuna trial"); ax.set_ylabel("validation F1")
        ax.set_title("Credit model: hyperparameter search"); ax.legend()
        save(fig, "credit_tuning_history.png")

    # --- feature importance -------------------------------------------------
    importance = pd.Series(model.feature_importances_, index=columns).nlargest(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importance.index[::-1], importance.values[::-1] * 100)
    ax.set_xlabel("importance (%)"); ax.set_title("Credit model: top 20 features")
    save(fig, "credit_feature_importance.png")

    # --- calibration --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, scores in series:
        frac, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=label)
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey", label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Credit model — calibration"); ax.legend()
    save(fig, "credit_calibration.png")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the credit model")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                        help="which split to score (default: test)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip writing figures to reports/figures/")
    args = parser.parse_args()

    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    check_provenance()

    df = get_split(args.split)
    if LABEL_COL not in df.columns:
        raise SystemExit(f"'{LABEL_COL}' missing from the {args.split} split — cannot score.")

    scorer = CreditScorer.load()
    print(f"Scorer: {type(scorer.model).__name__}, {len(scorer.feature_columns)} features, "
          f"threshold {scorer.threshold:.3f}")
    print(f"Split : {args.split}  —  {len(df):,} rows, "
          f"{df['Customer_ID'].nunique():,} customers, "
          f"positive rate {df[LABEL_COL].mean():.3f}\n")

    y = df[LABEL_COL].to_numpy()
    # Built once and shared with the figure code so the baseline model is
    # scored on exactly the same matrix rather than a second, subtly different
    # encoding of the same rows.
    global X_GLOBAL
    X_GLOBAL = scorer.build_features(df)
    proba = scorer.model.predict_proba(X_GLOBAL)[:, 1]
    pred = (proba >= scorer.threshold).astype(int)

    # Degenerate spread is the signature of a broken feature contract: an
    # all-zero frame makes every row score the same. Worth catching before the
    # metrics below get quoted anywhere.
    if proba.std() < 1e-6:
        print("WARNING: near-zero variance in predictions — the feature mapping "
              "likely produced an empty frame.\n")

    print("--- Metrics ---")
    print(f"  accuracy : {accuracy_score(y, pred):.4f}")
    print(f"  precision: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"  recall   : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"  f1       : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  roc_auc  : {roc_auc_score(y, proba):.4f}")

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print("\n--- Confusion matrix ---")
    print(f"                 predicted low   predicted high")
    print(f"  actual low     {tn:>13,}   {fp:>14,}")
    print(f"  actual high    {fn:>13,}   {tp:>14,}")
    print(f"\n  false negatives (risky, approved): {fn:,} "
          f"({fn / max(tp + fn, 1):.1%} of truly high-risk)")
    print(f"  false positives (safe, flagged)  : {fp:,} "
          f"({fp / max(tn + fp, 1):.1%} of truly low-risk)")
    print("\n  These two errors are not equal in cost. A false negative is a loan "
          "that\n  defaults; a false positive is a customer sent to manual review. "
          "The\n  threshold is where you choose the trade — retune it if this mix "
          "is wrong.")

    print("\n--- Score distribution ---")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  p{int(q * 100):<3}: {np.quantile(proba, q):.4f}")

    if not args.no_figures:
        try:
            meta_json = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
            names = write_figures(y, proba, pred, scorer.model,
                                  scorer.feature_columns, scorer.threshold, meta_json)
            print(f"\nFigures written to {FIGURE_DIR}")
            for name in names:
                print(f"  {name}")
            if not meta_json.get("tuned"):
                print("\n  Note: no tuned model on disk, so the comparison panels show a")
                print("  single series. Run `python -m ml_models.train_model --tune 30`")
                print("  to produce a baseline-versus-tuned comparison.")
        except Exception as exc:
            print(f"\nFigures skipped ({exc})")

    print(f"\nThe `system` split remains untouched and is still available for "
          f"end-to-end runs.")


if __name__ == "__main__":
    main()
