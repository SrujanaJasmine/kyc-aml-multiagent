"""
evaluate_system.py
==================
Evaluates the full decision layer credit, AML and KYC against the entire
held-out `system` split. Writes metric tables, an ablation comparing model-only
against rules-only against the combined agent logic, figures and per-item CSVs to
reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: F401,E402  loads .env
from agents.cda_agent import _make_decision as cda_decision  # noqa: E402
from database.customer_db import get_connection  # noqa: E402
from ml_models.data_splits import assign_split  # noqa: E402
from ml_models.feature_mapping import COLUMN_MAP, NON_FEATURE_COLUMNS  # noqa: E402
from policies.aml_rules import evaluate_aml_rules, recommended_action  # noqa: E402
from policies.credit_rules import evaluate_rules as evaluate_credit_rules  # noqa: E402

SPLIT = "system"
OUTPUT_ROOT = _REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_score: np.ndarray | None = None) -> dict:
    """
    Precision, recall, F1, F2 and the confusion matrix.

    Accuracy is included only for the credit task, where classes are within an
    order of magnitude of each other. For AML at 0.06% prevalence it is
    meaningless — predicting "never" scores 99.94% — and quoting it would
    mislead, so it is omitted there rather than reported with a caveat nobody
    reads.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, roc_auc_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0

    out = {
        "n": int(len(y_true)), "positives": int(y_true.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "f2": round(f2, 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "flagged": int(tp + fp),
        "flag_rate": round((tp + fp) / max(len(y_true), 1), 6),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
        out["pr_auc"] = round(average_precision_score(y_true, y_score), 4)
    return out


def md_table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> list[str]:
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return out + [""]


# ---------------------------------------------------------------------------
# Credit assessment
# ---------------------------------------------------------------------------
def evaluate_cda() -> dict:
    from ml_models.credit_risk_model import CreditScorer

    print("CREDIT ASSESSMENT", flush=True)
    scorer = CreditScorer.load()

    conn = get_connection()
    try:
        apps = pd.read_sql_query(
            """
            SELECT ca.*, c.name AS customer_name, c.age AS customer_age,
                   c.occupation AS customer_occupation,
                   c.annual_income AS customer_annual_income,
                   c.monthly_inhand_salary AS customer_monthly_inhand_salary,
                   c.num_bank_accounts AS customer_num_bank_accounts,
                   c.num_credit_card AS customer_num_credit_card
            FROM credit_applications ca JOIN customers c ON c.customer_id = ca.customer_id
            """, conn)
    finally:
        conn.close()

    apps = apps[apps["customer_id"].map(assign_split) == SPLIT].reset_index(drop=True)
    print(f"  {len(apps):,} applications from {apps['customer_id'].nunique():,} customers",
          flush=True)

    y = apps["Credit_Score"].to_numpy().astype(int)

    # Bulk rename rather than per-row mapping: identical result, and 9,392
    # dictionary round-trips is wasted work when the scorer takes a frame.
    frame = apps.rename(columns=COLUMN_MAP)
    frame = frame.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in frame.columns],
                       errors="ignore")

    t0 = time.time()
    scored = scorer.score(frame)
    proba = scored["high_risk_probability"].to_numpy()
    score_secs = time.time() - t0

    print(f"  scored in {score_secs:.1f}s; evaluating policy rules ...", flush=True)
    t0 = time.time()
    records = apps.to_dict("records")
    rule_results = [evaluate_credit_rules(r) for r in records]
    rule_secs = time.time() - t0

    breach_counts = np.array([r["breach_count"] for r in rule_results])
    high_sev = np.array([r["high_severity_count"] for r in rule_results])
    decisions = [cda_decision(p, scorer.threshold, r)[0] for p, r in zip(proba, rule_results)]

    model_flag = (proba >= scorer.threshold).astype(int)
    rules_flag = (breach_counts > 0).astype(int)
    combined_flag = np.array([0 if d == "Approve" else 1 for d in decisions])

    ablation = {
        "Model only": binary_metrics(y, model_flag, proba),
        "Rules only": binary_metrics(y, rules_flag),
        "Combined (agent)": binary_metrics(y, combined_flag, proba),
    }

    # Where the two disagree is where the agent routes to a human. Reporting how
    # much of the true-positive mass lives in that band is what justifies the
    # Review state existing at all.
    disagree = model_flag != rules_flag
    result = {
        "n_applications": int(len(apps)),
        "n_customers": int(apps["customer_id"].nunique()),
        "threshold": round(scorer.threshold, 4),
        "ablation": ablation,
        "decision_distribution": dict(Counter(decisions)),
        "disagreement": {
            "n": int(disagree.sum()),
            "share": round(float(disagree.mean()), 4),
            "true_positives_in_band": int(y[disagree].sum()),
            "share_of_all_true_positives": round(
                float(y[disagree].sum() / max(y.sum(), 1)), 4),
        },
        "rule_firing": dict(Counter(
            b["rule_id"] for r in rule_results for b in r["breached"])),
        "mean_breaches": round(float(breach_counts.mean()), 3),
        "mean_high_severity": round(float(high_sev.mean()), 3),
        "seconds_scoring": round(score_secs, 2),
        "seconds_rules": round(rule_secs, 2),
        "applications_per_second": round(len(apps) / max(score_secs + rule_secs, 1e-9), 1),
    }
    predictions = pd.DataFrame({
        "application_id": apps["application_id"], "customer_id": apps["customer_id"],
        "label": y, "probability": proba.round(6), "decision": decisions,
        "breaches": breach_counts, "high_severity": high_sev,
    })
    print(f"  combined F1 {ablation['Combined (agent)']['f1']:.4f} "
          f"(model only {ablation['Model only']['f1']:.4f}, "
          f"rules only {ablation['Rules only']['f1']:.4f})\n", flush=True)
    return {"metrics": result, "predictions": predictions, "proba": proba, "y": y}


# ---------------------------------------------------------------------------
# AML
# ---------------------------------------------------------------------------
def evaluate_aml(limit: int | None) -> dict | None:
    import joblib
    from ml_models.aml_features import load_cached
    from ml_models.train_aml_model import COLUMNS_PATH, MODEL_PATH, THRESHOLD_PATH

    print("AML SCREENING", flush=True)
    if not MODEL_PATH.exists():
        print("  SKIPPED — no AML model. Run: python -m ml_models.train_aml_model\n")
        return None

    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))

    X, meta = load_cached(SPLIT)
    if limit and len(meta) > limit:
        # Stratified: keep every positive, subsample negatives. Dropping
        # positives to save time would change the quantity being measured.
        pos = np.flatnonzero(meta["is_laundering"].to_numpy() == 1)
        neg = np.flatnonzero(meta["is_laundering"].to_numpy() == 0)
        keep = np.sort(np.concatenate(
            [pos, np.random.default_rng(42).choice(neg, size=limit - len(pos), replace=False)]))
        X, meta = X.iloc[keep].reset_index(drop=True), meta.iloc[keep].reset_index(drop=True)
        print(f"  sampled to {len(meta):,} rows (all {len(pos):,} positives retained)")

    y = meta["is_laundering"].to_numpy().astype(int)
    print(f"  {len(meta):,} transactions, {int(y.sum()):,} laundering "
          f"({100 * y.mean():.4f}%)", flush=True)

    t0 = time.time()
    proba = model.predict_proba(X[columns])[:, 1]
    score_secs = time.time() - t0

    print(f"  scored in {score_secs:.1f}s; evaluating rules over "
          f"{len(meta):,} rows ...", flush=True)
    t0 = time.time()
    feature_records = X.to_dict("records")
    txn_records = meta.to_dict("records")
    rule_results = []
    for i, (txn, feats) in enumerate(zip(txn_records, feature_records)):
        if i and i % 100_000 == 0:
            print(f"    {i:,}/{len(meta):,}", flush=True)
        rule_results.append(evaluate_aml_rules(txn, feats))
    rule_secs = time.time() - t0

    reg_breaches = np.array([r["regulatory_breach_count"] for r in rule_results])
    actions = [recommended_action(r, p, threshold)[0]
               for r, p in zip(rule_results, proba)]

    model_flag = (proba >= threshold).astype(int)
    rules_flag = (reg_breaches > 0).astype(int)
    combined_flag = np.array([0 if a == "No action" else 1 for a in actions])

    ablation = {
        "Model only": binary_metrics(y, model_flag, proba),
        "Rules only": binary_metrics(y, rules_flag),
        "Combined (agent)": binary_metrics(y, combined_flag, proba),
    }
    for arm in ablation.values():
        arm.pop("accuracy", None)   # meaningless at 0.06% prevalence

    days = 182
    disagree = model_flag != rules_flag
    per_typology = {}
    if "typology" in meta.columns:
        pos_meta = meta.loc[y == 1].copy()
        pos_meta["detected"] = combined_flag[y == 1]
        per_typology = {t: {"n": int(g.size), "recall": round(float(g.mean()), 4)}
                        for t, g in pos_meta.groupby("typology")["detected"]}

    result = {
        "n_transactions": int(len(meta)),
        "n_customers": int(meta["customer_id"].nunique()),
        "prevalence": round(float(y.mean()), 6),
        "threshold": round(threshold, 4),
        "ablation": ablation,
        "action_distribution": dict(Counter(actions)),
        "per_typology_recall": per_typology,
        "disagreement": {
            "n": int(disagree.sum()),
            "share": round(float(disagree.mean()), 4),
            "true_positives_in_band": int(y[disagree].sum()),
            "share_of_all_true_positives": round(
                float(y[disagree].sum() / max(y.sum(), 1)), 4),
        },
        "rule_firing": dict(Counter(
            b["rule_id"] for r in rule_results for b in r["breached"])),
        "alerts_per_day": round(float(combined_flag.sum()) / days, 1),
        "seconds_scoring": round(score_secs, 2),
        "seconds_rules": round(rule_secs, 2),
        "transactions_per_second": round(len(meta) / max(score_secs + rule_secs, 1e-9), 1),
    }
    predictions = pd.DataFrame({
        "transaction_id": meta["transaction_id"], "customer_id": meta["customer_id"],
        "label": y, "probability": proba.round(6), "action": actions,
        "regulatory_breaches": reg_breaches,
    })
    print(f"  combined F1 {ablation['Combined (agent)']['f1']:.4f} "
          f"(model only {ablation['Model only']['f1']:.4f}, "
          f"rules only {ablation['Rules only']['f1']:.4f})\n", flush=True)
    return {"metrics": result, "predictions": predictions, "proba": proba, "y": y}


# ---------------------------------------------------------------------------
# KYC — deterministic, so checked exhaustively rather than sampled
# ---------------------------------------------------------------------------
def evaluate_kyc() -> dict:
    from agents.kyc_agent import kyc_agent

    print("KYC", flush=True)
    conn = get_connection()
    try:
        customers = [c for (c,) in conn.execute("SELECT customer_id FROM customers")
                     if assign_split(c) == SPLIT]
    finally:
        conn.close()

    t0 = time.time()
    correct_existing = 0
    statuses = Counter()
    for cid in customers:
        res = kyc_agent({"type": "KYC", "input": {"customer_id": cid}})["completed_agents"][0]
        statuses[res.get("customer_status")] += 1
        if res.get("customer_status", "").startswith("Existing"):
            correct_existing += 1

    unknowns = [f"CUS_NOT_ON_FILE_{i:04d}" for i in range(200)]
    correct_new = sum(
        1 for cid in unknowns
        if kyc_agent({"type": "KYC", "input": {"customer_id": cid}})
        ["completed_agents"][0].get("customer_status") == "New")
    elapsed = time.time() - t0

    total = len(customers) + len(unknowns)
    print(f"  {correct_existing:,}/{len(customers):,} known customers classified Existing")
    print(f"  {correct_new}/{len(unknowns)} unknown ids classified New")
    print(f"  accuracy {100 * (correct_existing + correct_new) / total:.2f}%\n", flush=True)

    return {
        "n_known": len(customers), "n_unknown": len(unknowns),
        "correct_existing": correct_existing, "correct_new": correct_new,
        "accuracy": round((correct_existing + correct_new) / total, 4),
        "status_distribution": dict(statuses),
        "seconds": round(elapsed, 2),
        "lookups_per_second": round(total / max(elapsed, 1e-9), 1),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def write_figures(cda, aml, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")   # no display on a build machine or in CI
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    written = []

    def confusion(ax, m, title):
        grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(grid, cmap="Blues")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > grid.max() / 2 else "black")
        ax.set_xticks([0, 1], ["predicted\nnegative", "predicted\npositive"])
        ax.set_yticks([0, 1], ["actual\nnegative", "actual\npositive"])
        ax.set_title(title)

    panels = [(name, data) for name, data in (("Credit", cda), ("AML", aml)) if data]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    for ax, (name, data) in zip(np.atleast_1d(axes), panels):
        confusion(ax, data["metrics"]["ablation"]["Combined (agent)"], f"{name} — combined")
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=200); plt.close(fig); written.append(path.name)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    for ax, (name, data) in zip(np.atleast_1d(axes), panels):
        precision, recall, _ = precision_recall_curve(data["y"], data["proba"])
        ax.plot(recall, precision)
        ax.axhline(data["y"].mean(), ls="--", lw=1, color="grey",
                   label=f"prevalence {data['y'].mean():.4f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"{name} — precision/recall"); ax.legend()
    fig.tight_layout()
    path = out_dir / "precision_recall.png"
    fig.savefig(path, dpi=200); plt.close(fig); written.append(path.name)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    for ax, (name, data) in zip(np.atleast_1d(axes), panels):
        arms = list(data["metrics"]["ablation"])
        pos = np.arange(len(arms))
        for k, metric in enumerate(("precision", "recall", "f1")):
            ax.bar(pos + k * 0.26 - 0.26,
                   [data["metrics"]["ablation"][a][metric] for a in arms],
                   width=0.26, label=metric)
        ax.set_xticks(pos, [a.replace(" (agent)", "") for a in arms])
        ax.set_ylim(0, 1); ax.set_title(f"{name} — ablation"); ax.legend()
    fig.tight_layout()
    path = out_dir / "ablation.png"
    fig.savefig(path, dpi=200); plt.close(fig); written.append(path.name)

    return written


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aml-limit", type=int, default=None,
                        help="cap AML rows (positives always retained)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-kyc", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S")
    out_dir = OUTPUT_ROOT / f"evaluation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"FRAML — system evaluation on the `{SPLIT}` split")
    print(f"Both models were trained on `train`; every row below is unseen.\n")

    cda = evaluate_cda()
    aml = evaluate_aml(args.aml_limit)
    kyc = None if args.skip_kyc else evaluate_kyc()

    figures = []
    if not args.no_figures:
        try:
            figures = write_figures(cda, aml, out_dir)
            print(f"figures: {', '.join(figures)}\n")
        except Exception as exc:
            print(f"figures skipped ({exc})\n")

    cda["predictions"].to_csv(out_dir / "cda_predictions.csv", index=False)
    if aml:
        aml["predictions"].to_csv(out_dir / "aml_predictions.csv", index=False)

    metrics = {"generated": started.isoformat(), "split": SPLIT,
               "credit": cda["metrics"], "aml": aml["metrics"] if aml else None,
               "kyc": kyc}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- report -----------------------------------------------------------
    doc = [
        "# FRAML — System Evaluation", "",
        f"Generated {started:%Y-%m-%d %H:%M} UTC on the `{SPLIT}` split.", "",
        "Both the credit model and the AML model were fitted on the `train` "
        "partition. The `system` partition is split by customer, so no customer "
        "appearing below contributed a single row to training. Every figure here "
        "is out-of-sample.", "",
        "## 1. Credit assessment", "",
        f"{cda['metrics']['n_applications']:,} applications from "
        f"{cda['metrics']['n_customers']:,} customers, scored against the "
        f"dataset's `Credit_Score` label at a decision threshold of "
        f"{cda['metrics']['threshold']}.", "",
        "### Ablation", "",
    ]
    doc += md_table(
        [{"cfg": k, **v} for k, v in cda["metrics"]["ablation"].items()],
        ["cfg", "precision", "recall", "f1", "accuracy", "flagged", "tp", "fp", "fn"],
        ["Configuration", "Precision", "Recall", "F1", "Accuracy", "Flagged", "TP", "FP", "FN"])

    d = cda["metrics"]["disagreement"]
    doc += [
        f"Model and rules disagree on **{d['n']:,} applications ({d['share']:.1%})**, and "
        f"those contain **{d['true_positives_in_band']:,} of "
        f"{cda['metrics']['ablation']['Model only']['positives']:,} true positives "
        f"({d['share_of_all_true_positives']:.1%})**. Those are the cases the agent routes "
        f"to `Review` rather than deciding automatically.", "",
        "### Decision distribution", "",
    ]
    doc += md_table([{"d": k, "n": v} for k, v in cda["metrics"]["decision_distribution"].items()],
                    ["d", "n"], ["Decision", "Applications"])

    if aml:
        m = aml["metrics"]
        doc += ["## 2. AML screening", "",
                f"{m['n_transactions']:,} transactions from {m['n_customers']:,} customers, "
                f"prevalence {m['prevalence']:.5f}, threshold {m['threshold']}.", "",
                "Accuracy is deliberately omitted: at this prevalence, predicting "
                "\"never laundering\" scores over 99.9% and detects nothing.", "",
                "### Ablation", ""]
        doc += md_table([{"cfg": k, **v} for k, v in m["ablation"].items()],
                        ["cfg", "precision", "recall", "f1", "f2", "pr_auc", "flagged", "tp", "fp", "fn"],
                        ["Configuration", "Precision", "Recall", "F1", "F2", "PR-AUC",
                         "Flagged", "TP", "FP", "FN"])
        doc += [f"Alert volume at the combined decision: **{m['alerts_per_day']:.1f} per day** "
                f"across this population.", ""]
        if m["per_typology_recall"]:
            doc += ["### Recall by laundering typology", "",
                    "STRUCTURING and FAN_* are reachable from amount and velocity features. "
                    "SCATTER_GATHER and CYCLE exist only in the transaction graph, so weak "
                    "recall there would indicate the multi-hop features are not contributing.",
                    ""]
            doc += md_table([{"t": t, **v} for t, v in m["per_typology_recall"].items()],
                            ["t", "n", "recall"], ["Typology", "Positives", "Recall"])
        doc += ["### Action distribution", ""]
        doc += md_table([{"a": k, "n": v} for k, v in m["action_distribution"].items()],
                        ["a", "n"], ["Recommended action", "Transactions"])

    if kyc:
        doc += ["## 3. KYC", "",
                f"KYC is a deterministic database lookup, so it is verified exhaustively "
                f"rather than sampled: all {kyc['n_known']:,} `system` customers plus "
                f"{kyc['n_unknown']} identifiers known not to exist.", "",
                f"- Known customers classified Existing: **{kyc['correct_existing']:,} / "
                f"{kyc['n_known']:,}**",
                f"- Unknown identifiers classified New: **{kyc['correct_new']} / "
                f"{kyc['n_unknown']}**",
                f"- Accuracy: **{kyc['accuracy']:.4f}**", ""]

    doc += ["## 4. Policy rule coverage", "",
            "How often each deterministic rule fired. A rule that never fires across the "
            "whole split is either mis-specified or redundant.", ""]
    doc += md_table([{"r": k, "n": v} for k, v in
                     sorted(cda["metrics"]["rule_firing"].items(), key=lambda x: -x[1])],
                    ["r", "n"], ["Credit rule", "Applications"])
    if aml:
        doc += md_table([{"r": k, "n": v} for k, v in
                         sorted(aml["metrics"]["rule_firing"].items(), key=lambda x: -x[1])],
                        ["r", "n"], ["AML rule", "Transactions"])

    doc += ["## 5. Throughput", ""]
    thr = [{"c": "Credit", "n": cda["metrics"]["n_applications"],
            "s": cda["metrics"]["seconds_scoring"] + cda["metrics"]["seconds_rules"],
            "r": cda["metrics"]["applications_per_second"]}]
    if aml:
        thr.append({"c": "AML", "n": aml["metrics"]["n_transactions"],
                    "s": aml["metrics"]["seconds_scoring"] + aml["metrics"]["seconds_rules"],
                    "r": aml["metrics"]["transactions_per_second"]})
    if kyc:
        thr.append({"c": "KYC", "n": kyc["n_known"] + kyc["n_unknown"],
                    "s": kyc["seconds"], "r": kyc["lookups_per_second"]})
    doc += md_table(thr, ["c", "n", "s", "r"], ["Component", "Items", "Seconds", "Items/sec"])

    if figures:
        doc += ["## Figures", ""] + [f"![{f}]({f})" for f in figures] + [""]

    doc += ["## Caveats", "",
            "- The AML labels are typologies injected by our own generator "
            "(`EDA/transaction_generator.ipynb`). These figures demonstrate that the pipeline detects "
            "the patterns it was built to detect; they are not evidence of real-world "
            "detection performance, and the threshold would need recalibrating against "
            "true prevalence before any operational use.",
            "- Laundering prevalence was set at 3% of customers to make the problem "
            "trainable. That is far above any real portfolio, so precision in particular "
            "will not transfer.",
            "- The credit labels come from the source dataset and are used as provided.", ""]

    (out_dir / "SYSTEM_EVALUATION.md").write_text("\n".join(doc), encoding="utf-8")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"Completed in {elapsed / 60:.1f} minutes")
    print(f"  {out_dir}")
    print("  SYSTEM_EVALUATION.md · metrics.json · *_predictions.csv"
          + (" · figures" if figures else ""))


if __name__ == "__main__":
    main()
