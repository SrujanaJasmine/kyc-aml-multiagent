"""
evaluate_aml_model.py
=====================
Evaluates the AML model on the `test` split, reporting precision, recall, F1, F2,
PR-AUC, per-typology recall and alert volume, and writes figures to
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
from ml_models.aml_features import family_of, load_cached  # noqa: E402
from ml_models.train_aml_model import (  # noqa: E402
    COLUMNS_PATH, METRICS_PATH, MODEL_PATH, SIMULATION_DAYS, THRESHOLD_PATH,
    TOTAL_CUSTOMERS, fbeta,
)



FIGURE_DIR = _REPO_ROOT / "reports" / "figures"


def write_figures(y, proba, pred, model, columns, threshold, X, meta, meta_json) -> list[str]:
    """
    Publication figures for the AML model, written to reports/figures/.

    Precision-recall leads rather than ROC. At 0.06% prevalence the ROC curve is
    dominated by the negative class and hugs the top-left corner regardless of
    how useful the model is; PR responds to the thing being measured. ROC is
    still produced because reviewers expect it, but it is the weaker figure and
    should not be the one quoted.

    Where `train_aml_model.py --tune` saved a baseline, every comparable panel
    shows both series.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                                 precision_recall_curve, roc_curve)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    days = 182

    def save(fig, name):
        fig.tight_layout(); fig.savefig(FIGURE_DIR / name, dpi=200); plt.close(fig)
        written.append(name)

    base_proba = None
    base_path = ARTIFACT_DIR / "aml_xgboost_model_baseline.joblib"
    if base_path.exists():
        try:
            base_proba = joblib.load(base_path).predict_proba(X[columns])[:, 1]
        except Exception:
            base_proba = None

    series = [("tuned" if meta_json.get("tuned") else "model", proba)]
    if base_proba is not None:
        series.insert(0, ("baseline (default params)", base_proba))

    # --- precision / recall (the primary figure) ---------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, scores in series:
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision,
                label=f"{label} Avg.Precision = {average_precision_score(y, scores):.3f}")
    ax.axhline(y.mean(), ls="--", lw=1, color="grey",
               label=f"prevalence {y.mean():.5f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("AML model: precision/recall"); ax.legend()
    save(fig, "aml_precision_recall.png")

    # --- ROC ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, scores in series:
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], lw=1, color="grey", label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("AML model: ROC (flattering at this prevalence)"); ax.legend()
    save(fig, "aml_roc.png")

    # --- confusion ----------------------------------------------------------
    panels = [("tuned" if meta_json.get("tuned") else "model", pred)]
    bt_path = ARTIFACT_DIR / "aml_threshold_baseline.joblib"
    if base_proba is not None:
        bt = float(joblib.load(bt_path)) if bt_path.exists() else threshold
        panels.insert(0, ("baseline", (base_proba >= bt).astype(int)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (label, predictions) in zip(np.atleast_1d(axes), panels):
        grid = confusion_matrix(y, predictions, labels=[0, 1])
        ax.imshow(grid, cmap="Blues")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > grid.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred. clean", "pred. laundering"])
        ax.set_yticks([0, 1], ["actual clean", "actual laundering"])
        ax.set_title(f"AML — {label}")
    save(fig, "aml_confusion.png")

    # --- threshold sweep with alert volume ----------------------------------
    # Recall and alert volume are the two numbers a compliance team trades off,
    # and they belong on the same axes: a cutoff with excellent recall that
    # produces hundreds of alerts a day is not deployable, and no F-score shows
    # that.
    from sklearn.metrics import precision_score, recall_score, f1_score
    grid = np.arange(0.05, 0.96, 0.05)
    prec = [precision_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    rec = [recall_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    alerts = [(proba >= t).sum() / days for t in grid]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(grid, prec, label="precision"); ax.plot(grid, rec, label="recall")
    ax.plot(grid, f1s, label="F1")
    ax.axvline(threshold, ls="--", color="grey", label=f"chosen {threshold:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(grid, alerts, color="C3", ls=":", label="alerts/day")
    ax2.set_ylabel("alerts per day", color="C3")
    ax.set_title("AML model — threshold vs alert volume")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    save(fig, "aml_threshold_sweep.png")

    # --- Optuna history -----------------------------------------------------
    history = meta_json.get("tuning_history") or []
    if history:
        values = [h["value"] for h in history]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(range(len(values)), values, s=18, alpha=0.6, label="trial")
        ax.plot(np.maximum.accumulate(values), color="C1", label="best so far")
        if meta_json.get("baseline"):
            ax.axhline(meta_json["baseline"]["val_pr_auc"], ls="--", color="grey",
                       label=f"baseline {meta_json['baseline']['val_pr_auc']:.3f}")
        ax.set_xlabel("Optuna trial"); ax.set_ylabel("validation PR-AUC")
        ax.set_title("AML model — hyperparameter search"); ax.legend()
        save(fig, "aml_tuning_history.png")

    # --- feature importance, grouped and individual -------------------------
    imp = pd.Series(model.feature_importances_, index=columns)
    fam = imp.groupby([family_of(c) for c in columns]).sum().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(fam.index, fam.values * 100)
    axes[0].set_xlabel("importance (%)"); axes[0].set_title("By feature family")
    top = imp.nlargest(20)
    axes[1].barh(top.index[::-1], top.values[::-1] * 100)
    axes[1].set_xlabel("importance (%)"); axes[1].set_title("Top 20 individual")
    save(fig, "aml_feature_importance.png")

    # --- per-typology recall -------------------------------------------------
    if "typology" in meta.columns:
        pos = meta.loc[y == 1].copy()
        pos["detected"] = pred[y == 1]
        by_typ = pos.groupby("typology")["detected"].agg(["size", "mean"]).sort_values("mean")
        if len(by_typ):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            bars = ax.barh(by_typ.index, by_typ["mean"])
            for bar, (_, row) in zip(bars, by_typ.iterrows()):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"n={int(row['size'])}", va="center", fontsize=8)
            ax.set_xlim(0, 1.12); ax.set_xlabel("recall")
            ax.set_title("AML recall by injected typology")
            save(fig, "aml_typology_recall.png")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["test", "val", "system"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the trained threshold")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip writing figures to reports/figures/")
    args = parser.parse_args()

    from sklearn.metrics import average_precision_score, confusion_matrix

    if not MODEL_PATH.exists():
        raise SystemExit(f"No model at {MODEL_PATH}. Run: python -m ml_models.train_aml_model")

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    threshold = args.threshold if args.threshold is not None else float(joblib.load(THRESHOLD_PATH))

    if METRICS_PATH.exists():
        meta_json = json.loads(METRICS_PATH.read_text())
        print(f"Model trained {meta_json.get('trained_at')}")
        print(f"  {meta_json.get('n_train_positives'):,} positives, "
              f"{meta_json.get('neg_per_pos')}:1 downsampling, "
              f"{meta_json.get('n_features')} features\n")

    X, meta = load_cached(args.split)
    X = X[columns]
    y = meta["is_laundering"].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    customers = meta["customer_id"].nunique()
    print(f"Split  : {args.split} — {len(y):,} rows, {customers:,} customers, "
          f"{int(y.sum()):,} positives ({100*y.mean():.4f}%)")
    print(f"Cutoff : {threshold:.2f}\n")

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print("--- Detection metrics ---")
    print(f"  precision : {precision:.4f}")
    print(f"  recall    : {recall:.4f}")
    print(f"  F1        : {fbeta(precision, recall, 1):.4f}")
    print(f"  F2        : {fbeta(precision, recall, 2):.4f}   <- the tuned objective")
    print(f"  PR-AUC    : {average_precision_score(y, proba):.4f}")
    print("  (accuracy omitted on purpose: all-negative scores "
          f"{100*(1-y.mean()):.2f}% here and detects nothing)")

    print("\n--- Confusion matrix ---")
    print(f"                  predicted clean   predicted laundering")
    print(f"  actual clean    {tn:>15,}   {fp:>20,}")
    print(f"  actual launder  {fn:>15,}   {tp:>20,}")

    scale = TOTAL_CUSTOMERS / max(customers, 1)
    alerts_day = (tp + fp) / SIMULATION_DAYS * scale
    print(f"\n--- Operational load ---")
    print(f"  alerts in this split      : {tp+fp:,}")
    print(f"  scaled to {TOTAL_CUSTOMERS:,} customers : ~{alerts_day:.0f} per day")
    print(f"  of which genuine          : ~{alerts_day*precision:.1f} per day")
    print(f"  missed laundering         : {fn:,} of {int(y.sum()):,}")

    # --- per-typology recall -------------------------------------------------
    if "typology" in meta.columns:
        print("\n--- Recall by typology ---")
        print("  STRUCTURING and FAN_* are reachable from amount and velocity features.")
        print("  SCATTER_GATHER and CYCLE exist only in the graph -- weak recall there")
        print("  means the multi-hop features are not contributing.\n")
        pos = meta.loc[y == 1].copy()
        pos["detected"] = pred[y == 1]
        by_typ = pos.groupby("typology")["detected"].agg(["size", "sum", "mean"])
        for typ, r in by_typ.sort_values("mean", ascending=False).iterrows():
            flag = "" if r["mean"] >= 0.5 else "   <- weak"
            print(f"  {typ:<16} {int(r['sum']):>4}/{int(r['size']):<5} "
                  f"recall {r['mean']:.3f}{flag}")

    # --- grouped importance --------------------------------------------------
    print("\n--- Feature importance by family ---")
    imp = pd.Series(model.feature_importances_, index=columns)
    fam = imp.groupby([family_of(c) for c in columns]).sum().sort_values(ascending=False)
    for name, value in fam.items():
        print(f"  {name:<12} {100*value:>6.2f}%")

    print("\n  top 12 individual features:")
    for col, value in imp.sort_values(ascending=False).head(12).items():
        print(f"    {col:<30} {100*value:>6.2f}%")

    if not args.no_figures:
        try:
            meta_json = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
            names = write_figures(y, proba, pred, model, columns, threshold, X, meta, meta_json)
            print(f"\nFigures written to {FIGURE_DIR}")
            for name in names:
                print(f"  {name}")
            if not meta_json.get("tuned"):
                print("\n  Note: no tuned model on disk, so the comparison panels show a")
                print("  single series. Run `python -m ml_models.train_aml_model --tune 20`")
                print("  to produce a baseline-versus-tuned comparison.")
        except Exception as exc:
            print(f"\nFigures skipped ({exc})")

    print("\nNOTE: these figures come from simulated data whose laundering patterns "
          "we injected ourselves.\nThey show the pipeline works end to end. They are "
          "not evidence of real-world detection\nperformance, and the threshold would "
          "need recalibrating against true prevalence before use.")


if __name__ == "__main__":
    main()
