"""
cda_agent.py
============
Credit assessment worker. Scores an application with the trained XGBoost model,
tests it against the US lending rules in policies/, and returns Approve, Review or
Decline together with Regulation B adverse-action reasons. Where the model and the
rules disagree the case is routed to Review rather than decided automatically.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: F401,E402  (loads .env before anything reads os.environ)
from agents.state import WorkerState  # noqa: E402
from database.customer_db import (  # noqa: E402
    get_credit_application,
    get_customer,
    get_customer_applications,
)
from ml_models.credit_risk_model import CreditScorer  # noqa: E402
from ml_models.feature_mapping import (  # noqa: E402
    application_row_to_features,
    normalize_application_payload,
)
from policies.credit_rules import adverse_action_reasons, evaluate_rules  # noqa: E402
from policies.retrieval import retrieve_policy_text  # noqa: E402

# Auto-approve below (threshold - margin), auto-decline above (threshold +
# margin). The model's tuned threshold is the point of maximum F1, i.e. the
# point of maximum ambiguity — deciding automatically right at it would mean
# auto-deciding the cases the model is least sure about. The margins carve out
# a band around it for human review.
APPROVE_MARGIN = 0.10
DECLINE_MARGIN = 0.30
SHAP_TOP_N = 5

# Customer table column -> the customer_-prefixed name the rest of the
# pipeline expects (matching get_credit_application's JOIN aliases).
_CUSTOMER_FIELD_MAP = {
    "age": "customer_age",
    "occupation": "customer_occupation",
    "annual_income": "customer_annual_income",
    "monthly_inhand_salary": "customer_monthly_inhand_salary",
    "num_bank_accounts": "customer_num_bank_accounts",
    "num_credit_card": "customer_num_credit_card",
    "name": "customer_name",
}

_scorer: CreditScorer | None = None
_scorer_error: str | None = None
_explainer = None


def _get_scorer() -> CreditScorer | None:
    """Load once, and cache the failure too — a missing artifact shouldn't
    retry a doomed disk read on every application."""
    global _scorer, _scorer_error
    if _scorer is None and _scorer_error is None:
        try:
            _scorer = CreditScorer.load()
        except Exception as exc:
            _scorer_error = str(exc)
    return _scorer


def _get_explainer(scorer: CreditScorer):
    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(scorer.model)
    return _explainer


def _enrich_from_customer(application: dict, customer_id: str | None) -> tuple[dict, dict | None]:
    """
    Fill stable customer attributes the payload didn't carry.

    The payload wins on conflict: if a caller explicitly sends an income, that
    is the figure being underwritten, and silently replacing it with a stale
    stored value would assess a different application than the one submitted.
    """
    if not customer_id:
        return application, None

    customer = get_customer(customer_id)
    if customer is None:
        return application, None

    enriched = dict(application)
    for db_field, prefixed in _CUSTOMER_FIELD_MAP.items():
        if enriched.get(prefixed) in (None, "") and customer.get(db_field) is not None:
            enriched[prefixed] = customer[db_field]
    enriched.setdefault("customer_id", customer_id)

    # SSN is deliberately not copied forward. It is never a model feature and
    # the result dict is handed to an LLM for report writing.
    safe_customer = {k: v for k, v in customer.items() if k != "ssn"}
    return enriched, safe_customer


def _summarize_history(customer_id: str | None, current: dict) -> dict:
    """
    Condense past applications into a few comparable numbers.

    The model sees one month in isolation — none of its 59 features encode
    history. Whether this month's utilization is an outlier or the customer's
    normal operating level is exactly the context a reviewer needs and the
    model cannot supply, so it is computed here and passed through to the
    report rather than folded into the score.
    """
    if not customer_id:
        return {"prior_applications": 0, "note": "no customer_id supplied"}

    history = get_customer_applications(customer_id, limit=20)
    if not history:
        return {"prior_applications": 0, "note": "no prior applications on file"}

    def _floats(key):
        return [float(h[key]) for h in history
                if h.get(key) is not None and str(h[key]).strip() != ""]

    utils = _floats("Credit_Utilization_Ratio")
    delays = _floats("Delay_from_due_date")
    late_counts = _floats("Num_of_Delayed_Payment")

    summary = {
        "prior_applications": len(history),
        "prior_statuses": sorted({h.get("status") for h in history if h.get("status")}),
        "mean_prior_utilization": round(sum(utils) / len(utils), 2) if utils else None,
        "max_prior_delay_days": max(delays) if delays else None,
        "max_prior_delayed_payments": max(late_counts) if late_counts else None,
        "months_on_file": sorted({h.get("Month") for h in history if h.get("Month")}),
    }

    # Trend: is this month worse than the customer's own baseline?
    current_util = current.get("Credit_Utilization_Ratio")
    if utils and current_util is not None:
        try:
            delta = float(current_util) - summary["mean_prior_utilization"]
            summary["utilization_vs_history"] = round(delta, 2)
            summary["utilization_trend"] = (
                "deteriorating" if delta > 10 else
                "improving" if delta < -10 else "stable"
            )
        except (TypeError, ValueError):
            pass

    return summary


def _explain_shap(scorer: CreditScorer, features: pd.DataFrame, top_n: int = SHAP_TOP_N) -> list[dict]:
    """
    Top-N drivers for this specific score.

    SHAP runs on `build_features` output — the same matrix the model consumed.
    Explaining a differently-shaped frame is how you get explanations that read
    well and describe a computation that never happened.
    """
    try:
        X = scorer.build_features(features)
        shap_values = _get_explainer(scorer).shap_values(X)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        sv = sv[0]

        frame = pd.DataFrame({"feature": X.columns, "shap_value": sv})
        frame["abs_shap"] = frame["shap_value"].abs()
        top = frame.sort_values("abs_shap", ascending=False).head(top_n)

        return [
            {
                "feature": row.feature,
                "value": float(X.iloc[0][row.feature]),
                "shap_value": round(float(row.shap_value), 4),
                "direction": "increases risk" if row.shap_value > 0 else "decreases risk",
            }
            for row in top.itertuples()
        ]
    except Exception as exc:
        return [{"error": f"SHAP unavailable: {exc}"}]


def _make_decision(probability: float, threshold: float, rules: dict) -> tuple[str, str]:
    """
    Combine the statistical and the policy view, and say why.

    They answer different questions — "how does this resemble past defaulters"
    versus "which written standards does it fail" — so when they disagree the
    application goes to a human rather than being auto-decided. Disagreement is
    the case a reviewer most needs to see.
    """
    approve_below = max(threshold - APPROVE_MARGIN, 0.0)
    decline_above = min(threshold + DECLINE_MARGIN, 1.0)
    high_sev = rules["high_severity_count"]
    regulatory = rules["regulatory_breach_count"]
    total = rules["breach_count"]

    if probability >= decline_above and high_sev > 0:
        return "Decline", (
            f"Model risk {probability:.1%} exceeds the auto-decline band "
            f"({decline_above:.0%}) and {high_sev} high-severity rule(s) were "
            f"breached, {regulatory} of them regulatory."
        )
    if probability < approve_below and total == 0:
        return "Approve", (
            f"Model risk {probability:.1%} is below the auto-approve band "
            f"({approve_below:.0%}) and no policy rules were breached."
        )
    if probability < approve_below and total > 0:
        return "Review", (
            f"Model risk is low ({probability:.1%}) but {total} rule(s) were "
            f"breached — the statistical and policy views disagree."
        )
    if probability >= decline_above:
        return "Review", (
            f"Model risk is high ({probability:.1%}) but no high-severity rule "
            f"was breached — the driver is not captured by written policy."
        )
    return "Review", (
        f"Model risk {probability:.1%} falls in the manual-review band with "
        f"{total} policy breach(es)."
    )


def cda_agent(state: WorkerState) -> dict:
    """Assess one credit application against the model and the policy rules."""
    state = WorkerState.model_validate(state)
    payload = state.input or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    application_id = payload.get("application_id")
    customer_id = payload.get("customer_id")
    raw_application = payload.get("application")

    def _fail(event: str, message: str) -> dict:
        result = {"agent": "CDA", "application_id": application_id,
                  "customer_id": customer_id, "error": message, "timestamp": timestamp}
        return {"completed_agents": [result],
                "audit_log": [{"agent": "CDA", "event": event, "output": result}]}

    # --- resolve the application -------------------------------------------
    if raw_application:
        application = normalize_application_payload(raw_application)
        source = "payload"
    elif application_id:
        stored = get_credit_application(application_id)
        if stored is None:
            return _fail("cda_lookup_failed",
                         f"no application payload supplied and application_id "
                         f"{application_id} not found in the database")
        application = normalize_application_payload(stored)
        source = "database"
    else:
        return _fail("cda_input_invalid",
                     "input must contain either an 'application' payload or an "
                     "'application_id' to load from the database")

    customer_id = customer_id or application.get("customer_id")
    application, customer = _enrich_from_customer(application, customer_id)

    scorer = _get_scorer()
    if scorer is None:
        return _fail("cda_model_unavailable", f"credit model unavailable: {_scorer_error}")

    # --- score --------------------------------------------------------------
    features = application_row_to_features(application)
    scored = scorer.score(features)
    probability = float(scored["high_risk_probability"].iloc[0])
    risk_label = str(scored["risk_label"].iloc[0])

    # --- policy -------------------------------------------------------------
    rules = evaluate_rules(application)
    policy_context = retrieve_policy_text(rules["breached"])

    # --- context, explanation, decision -------------------------------------
    history = _summarize_history(customer_id, application)
    top_shap_features = _explain_shap(scorer, features)
    decision, decision_reason = _make_decision(probability, scorer.threshold, rules)

    result = {
        "agent": "CDA",
        "application_id": application_id,
        "customer_id": customer_id,
        "application_source": source,
        "customer_on_file": customer is not None,
        "probability": round(probability, 4),
        "model_threshold": round(scorer.threshold, 4),
        "risk_label": risk_label,
        "decision": decision,
        "decision_reason": decision_reason,
        "adverse_action_reasons": (
            adverse_action_reasons(rules["breached"]) if decision != "Approve" else []
        ),
        "breached_rules": rules["breached"],
        "rules_passed": len(rules["passed"]),
        "rules_not_evaluated": rules["not_evaluated"],
        "regulatory_breach_count": rules["regulatory_breach_count"],
        "credit_policy_context": policy_context,
        "top_shap_features": top_shap_features,
        "customer_history": history,
        "timestamp": timestamp,
    }

    return {"completed_agents": [result],
            "audit_log": [{"agent": "CDA", "event": "cda_assessment_complete", "output": result}]}


if __name__ == "__main__":
    # Smoke test with a payload — no database row required for the application
    # itself, which is the point of the new contract.
    demo = {
        "type": "CDA",
        "input": {
            "application_id": "APP-DEMO-001",
            "customer_id": "CUST001",
            "application": {
                "Month": "January",
                "Interest_Rate": 21,
                "Num_of_Loan": 7,
                "Delay_from_due_date": 62,
                "Num_of_Delayed_Payment": 11,
                "Changed_Credit_Limit": 14.3,
                "Num_Credit_Inquiries": 9,
                "Credit_Mix": "Bad",
                "Outstanding_Debt": 3400.0,
                "Credit_Utilization_Ratio": 81.2,
                "Payment_of_Min_Amount": "Yes",
                "Total_EMI_per_month": 2600,
                "Amount_invested_monthly": 40,
                "Payment_Behaviour": "Low_spent_Small_value_payments",
                "Monthly_Balance": 180,
                "Credit_History_Age_Total": 16,
                # customer fields omitted on purpose — enriched from the DB if
                # CUST001 exists, otherwise scored without them
                "Age": 29,
                "Annual_Income": 41000,
                "Monthly_Inhand_Salary": 3200,
                "Num_Bank_Accounts": 6,
                "Num_Credit_Card": 7,
                "Occupation": "Mechanic",
                "Payday_Loan": 1, "Personal_Loan": 1, "Auto_Loan": 1,
            },
        },
    }

    res = cda_agent(demo)["completed_agents"][0]
    if "error" in res:
        raise SystemExit(f"ERROR: {res['error']}")

    print(f"source         : {res['application_source']}  "
          f"(customer on file: {res['customer_on_file']})")
    print(f"P(High Risk)   : {res['probability']}  (threshold {res['model_threshold']})")
    print(f"decision       : {res['decision']}")
    print(f"reason         : {res['decision_reason']}")
    print(f"\nadverse-action reasons (Reg B, max 4):")
    for r in res["adverse_action_reasons"]:
        print(f"  - {r}")
    print(f"\nbreached rules ({len(res['breached_rules'])}):")
    for r in res["breached_rules"]:
        print(f"  [{r['severity']:<6}] [{r['authority']:<12}] {r['rule_id']:<24} "
              f"observed={r['observed']} limit={r['threshold']}")
        print(f"           {r['source_name']}")
    print(f"\ncustomer history: {res['customer_history']}")
    print("\ntop SHAP drivers:")
    for f in res["top_shap_features"]:
        print("  ", f.get("error") or
              f"{f['feature']:<45} {f['shap_value']:>8} ({f['direction']})")
