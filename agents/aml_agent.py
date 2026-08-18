"""
aml_agent.py
============
AML worker. Scores a transaction for money laundering, tests it against the BSA/AML
obligations in policies/, and rolls the customer's recent activity up to the $5,000
SAR aggregation test. Returns both a transaction verdict and a customer-level one.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: F401,E402  loads .env
from agents.state import WorkerState  # noqa: E402
from database.customer_db import get_customer, get_transaction  # noqa: E402
from ml_models.aml_features import build_features, load_neighbourhood  # noqa: E402
from policies.aml_rules import (  # noqa: E402
    case_references, evaluate_aml_rules, recommended_action,
)

DEFAULT_WINDOW_DAYS = 30
SHAP_TOP_N = 5

_model = None
_columns = None
_threshold = None
_load_error: str | None = None
_explainer = None


def _load_model():
    """Load once, cache the failure too. A missing artifact degrades AML to a
    rules-only assessment rather than taking down the whole graph."""
    global _model, _columns, _threshold, _load_error
    if _model is None and _load_error is None:
        try:
            import joblib
            from ml_models.train_aml_model import COLUMNS_PATH, MODEL_PATH, THRESHOLD_PATH
            missing = [p.name for p in (MODEL_PATH, COLUMNS_PATH, THRESHOLD_PATH)
                       if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"missing AML artifacts: {', '.join(missing)}. "
                    "Run: python -m ml_models.train_aml_model")
            _model = joblib.load(MODEL_PATH)
            _columns = joblib.load(COLUMNS_PATH)
            _threshold = float(joblib.load(THRESHOLD_PATH))
        except Exception as exc:
            _load_error = str(exc)
    return _model, _columns, _threshold


def _explain(model, X_row: pd.DataFrame) -> list[dict]:
    global _explainer
    try:
        import shap
        if _explainer is None:
            _explainer = shap.TreeExplainer(model)
        values = _explainer.shap_values(X_row)
        values = values[1] if isinstance(values, list) else values
        values = values[0]

        frame = pd.DataFrame({"feature": X_row.columns, "shap_value": values})
        frame["abs"] = frame["shap_value"].abs()
        top = frame.sort_values("abs", ascending=False).head(SHAP_TOP_N)
        return [{"feature": r.feature,
                 "value": float(X_row.iloc[0][r.feature]),
                 "shap_value": round(float(r.shap_value), 4),
                 "direction": "raises suspicion" if r.shap_value > 0 else "lowers suspicion"}
                for r in top.itertuples()]
    except Exception as exc:
        return [{"error": f"SHAP unavailable: {exc}"}]


def _customer_rollup(scored: pd.DataFrame, window_days: int, threshold: float,
                     anchor: pd.Timestamp | None = None) -> dict:
    """
    Aggregate the customer's activity around the transaction under review.

    The window is anchored on the TARGET transaction's timestamp, not on the
    customer's most recent activity. An analyst reviewing a March payment needs
    March's context; anchoring on the latest row would roll a March transaction
    up against June and report that nothing was flagged, because the transaction
    being reviewed sits outside its own window.

    The $5,000 test is applied to the summed value of *flagged* transactions
    rather than to all activity, because 31 CFR 1020.320 concerns suspicious
    activity aggregating $5,000 -- not turnover.
    """
    if scored.empty:
        return {"window_days": window_days, "transactions": 0, "flagged": 0}

    end = anchor if anchor is not None else scored["timestamp"].max()
    cutoff = end - timedelta(days=window_days)
    window = scored[(scored["timestamp"] >= cutoff) & (scored["timestamp"] <= end)]
    flagged = window[window["probability"] >= threshold]

    flagged_value = float(flagged["amount"].sum())
    return {
        "window_days": window_days,
        "window_start": str(cutoff),
        "window_end": str(end),
        "transactions": int(len(window)),
        "flagged": int(len(flagged)),
        "flagged_rate": round(len(flagged) / max(len(window), 1), 4),
        "flagged_value": round(flagged_value, 2),
        "max_probability": round(float(window["probability"].max()), 4),
        "mean_probability": round(float(window["probability"].mean()), 4),
        "total_value": round(float(window["amount"].sum()), 2),
        # The SAR aggregation test, stated explicitly so the reasoning is visible.
        "meets_sar_aggregate": bool(flagged_value >= 5_000.0),
        "top_flagged": [
            {"transaction_id": r.transaction_id, "timestamp": str(r.timestamp),
             "amount": round(float(r.amount), 2), "payment_format": r.payment_format,
             "probability": round(float(r.probability), 4)}
            for r in flagged.nlargest(min(5, len(flagged)), "probability").itertuples()
        ],
    }


def aml_agent(state: WorkerState) -> dict:
    """Screen one transaction and roll its customer's recent activity up."""
    state = WorkerState.model_validate(state)
    payload = state.input or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    transaction_id = payload.get("transaction_id")
    customer_id = payload.get("customer_id")
    window_days = int(payload.get("window_days", DEFAULT_WINDOW_DAYS))
    supplied = payload.get("transaction")

    def _fail(event: str, message: str) -> dict:
        result = {"agent": "AML", "transaction_id": transaction_id,
                  "customer_id": customer_id, "error": message, "timestamp": timestamp}
        return {"completed_agents": [result],
                "audit_log": [{"agent": "AML", "event": event, "output": result}]}

    # --- resolve the transaction -------------------------------------------
    if supplied:
        transaction = dict(supplied)
        transaction.setdefault("transaction_id", transaction_id or "UNSAVED")
    elif transaction_id:
        transaction = get_transaction(transaction_id)
        if transaction is None:
            return _fail("aml_lookup_failed", f"transaction_id {transaction_id} not found")
    else:
        return _fail("aml_input_invalid",
                     "input must contain either 'transaction_id' or a 'transaction' payload")

    customer_id = customer_id or transaction.get("customer_id")
    if not customer_id:
        return _fail("aml_input_invalid", "no customer_id on the transaction or in the input")

    customer = get_customer(customer_id)
    # SSN is never forwarded: it is not a feature and the result is handed to an LLM.
    safe_customer = {k: v for k, v in (customer or {}).items() if k != "ssn"}

    # --- features over the account neighbourhood ---------------------------
    neighbourhood = load_neighbourhood(customer_id)
    if neighbourhood.empty:
        return _fail("aml_no_history", f"no transactions on file for customer {customer_id}")

    X, meta = build_features(neighbourhood)

    target_mask = (meta["transaction_id"] == transaction.get("transaction_id")).to_numpy()
    if not target_mask.any():
        # A payload transaction not yet persisted: score the customer's most
        # recent row as the closest available context rather than failing.
        target_mask = np.zeros(len(meta), dtype=bool)
        target_mask[-1] = True

    model, columns, threshold = _load_model()

    result: dict = {
        "agent": "AML",
        "transaction_id": transaction.get("transaction_id"),
        "customer_id": customer_id,
        "customer": safe_customer,
        "transaction": {k: transaction.get(k) for k in
                        ("timestamp", "amount_received", "payment_format",
                         "from_account", "to_account", "from_bank", "to_bank")},
        "timestamp": timestamp,
    }

    feature_row = X.loc[target_mask].iloc[[0]]
    features = feature_row.iloc[0].to_dict()

    # --- rules (independent of the model) ----------------------------------
    rules = evaluate_aml_rules(transaction, features)
    result["breached_rules"] = rules["breached"]
    result["rules_passed"] = len(rules["passed"])
    result["rules_not_evaluated"] = rules["not_evaluated"]
    result["regulatory_breach_count"] = rules["regulatory_breach_count"]
    result["case_references"] = case_references(rules["breached"])

    # --- model --------------------------------------------------------------
    if model is None:
        result["model_available"] = False
        result["model_error"] = _load_error
        probability, threshold = 0.0, 1.0
        result["verdict"] = "Rules-only assessment — model unavailable"
    else:
        result["model_available"] = True
        proba_all = model.predict_proba(X[columns])[:, 1]
        probability = float(proba_all[target_mask][0])

        result["probability"] = round(probability, 4)
        result["model_threshold"] = round(threshold, 4)
        result["is_laundering"] = bool(probability >= threshold)
        result["verdict"] = ("Suspicious — flagged" if probability >= threshold
                             else "Not flagged by the model")
        result["top_shap_features"] = _explain(model, feature_row[columns])

        scored = meta[["transaction_id", "customer_id", "timestamp", "amount",
                       "payment_format"]].copy()
        scored["probability"] = proba_all
        scored = scored[scored["customer_id"] == customer_id]
        anchor = meta.loc[target_mask, "timestamp"].iloc[0]
        result["customer_rollup"] = _customer_rollup(scored, window_days, threshold, anchor)

    action, reason = recommended_action(rules, probability, threshold)
    result["recommended_action"] = action
    result["action_reason"] = reason

    return {"completed_agents": [result],
            "audit_log": [{"agent": "AML", "event": "aml_assessment_complete",
                           "output": result}]}


if __name__ == "__main__":
    from database.customer_db import get_connection

    conn = get_connection()
    row = conn.execute(
        "SELECT transaction_id, customer_id FROM transactions "
        "WHERE is_laundering = 1 LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT transaction_id, customer_id FROM transactions LIMIT 1").fetchone()
    conn.close()

    out = aml_agent({"type": "AML",
                     "input": {"transaction_id": row[0], "customer_id": row[1]}})
    res = out["completed_agents"][0]

    if "error" in res:
        raise SystemExit(f"ERROR: {res['error']}")

    cust = res.get("customer", {})
    print(f"customer   : {cust.get('customer_id')}  {cust.get('name')}  "
          f"({cust.get('occupation')})")
    print(f"transaction: {res['transaction_id']}  "
          f"${res['transaction'].get('amount_received')}  "
          f"{res['transaction'].get('payment_format')}")
    print(f"verdict    : {res['verdict']}")
    if res.get("model_available"):
        print(f"probability: {res['probability']} (threshold {res['model_threshold']})")

    print(f"\nbreached rules ({len(res['breached_rules'])}):")
    for r in res["breached_rules"]:
        print(f"  [{r['severity']:<6}] [{r['authority']:<10}] {r['rule_id']:<18} "
              f"observed: {r['observed']}")
        print(f"           {r['source_name']}")

    print(f"\npublished cases / guidance:")
    for c in res["case_references"]:
        print(f"  - {c['publication']}\n    {c['url']}")

    roll = res.get("customer_rollup", {})
    if roll:
        print(f"\ncustomer roll-up ({roll.get('window_days')} days):")
        print(f"  {roll.get('flagged')} flagged of {roll.get('transactions')} "
              f"transactions, ${roll.get('flagged_value'):,.2f} flagged value")
        print(f"  meets SAR $5,000 aggregate: {roll.get('meets_sar_aggregate')}")

    print(f"\nrecommended action: {res['recommended_action']}")
    print(f"  {res['action_reason']}")
