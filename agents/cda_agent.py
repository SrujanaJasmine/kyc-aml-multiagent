"""
cda_agent.py
=============
Credit Assessment (CDA) worker for the multi-agent compliance graph.
Wired into `graph.py` as `builder.add_node("cda_agent", cda_agent)` and
reached only via `route_to_workers`'s `Send("cda_agent", {"type": "CDA",
"input": item})` in graph_state.py.

Unlike the earlier version of this file, CDA does NOT own its own SQLite
table or DB connection. `customer_db.py` is the single shared data layer
for every agent — CDA just calls `get_credit_application(application_id)`,
the same pattern `kyc_agent` uses for `get_customer` and `aml_agent` uses
for `get_transaction`. `assessment_input` items therefore only need an
`application_id`, not a full feature payload.

Responsibilities (per project spec):
    1. Calculate the credit risk probability using the trained XGBoost model.
    2. Make a credit decision: Approve / Review / Decline.
    3. Look up the credit application (+ joined customer fields) via
       `customer_db.get_credit_application` — this already distinguishes
       existing vs. new customers implicitly (a returned row means the
       application/customer already exists in `customer_data.db`).
    4. Summarize top-5 SHAP features + retrieve the applicable credit
       policy rule from the credit FAISS store, so the Explanation agent
       can cite both in its plain-English report.

TODO once the credit dataset CSV columns are confirmed: `NON_FEATURE_COLUMNS`
below and the fields kept in `feature_row` need to match whatever
`credit_applications` actually ends up storing (loan_amount, existing_debt,
credit_score, etc.) plus any joined customer fields the trained model was
fit on (annual_income, num_bank_accounts, ...).
"""

from datetime import datetime, timezone

import joblib
import pandas as pd
import shap

# Needed so joblib can unpickle the saved CreditRiskPredictor object
from credit_risk_model import CreditRiskPredictor

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from customer_db import get_credit_application
from graph_state import WorkerState

# ---------------------------------------------------------------------------
# Decision thresholds — tune these against your validation set / policy doc
# ---------------------------------------------------------------------------
APPROVE_THRESHOLD = 0.30   # probability of High Risk below this -> Approve
DECLINE_THRESHOLD = 0.60   # probability of High Risk above this -> Decline
                           # anything in between -> Review

MODEL_PATH = "ml_model/artifacts/credit_risk_model.joblib"
FAISS_INDEX_PATH = "rag_stores/credit_policy_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Columns from get_credit_application()'s joined row that aren't model
# features — strip these before building the DataFrame the predictor sees.
NON_FEATURE_COLUMNS = {"application_id", "customer_id", "customer_name", "status"}


# ---------------------------------------------------------------------------
# Model + RAG store — loaded once at import time (module-level singleton,
# same pattern as db_analyst_agent's cached ReAct agent). Avoids reloading
# the joblib model / embeddings / FAISS index on every Send.
# ---------------------------------------------------------------------------
_predictor: CreditRiskPredictor = joblib.load(MODEL_PATH)
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
_credit_policy_store = FAISS.load_local(
    FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True
)


def _calculate_score(application_df: pd.DataFrame) -> tuple[float, int]:
    proba = _predictor.predict_proba(application_df)[:, 1][0]  # P(High Risk)
    pred = int(_predictor.predict(application_df)[0])
    return float(proba), pred


def _make_decision(probability: float) -> str:
    if probability < APPROVE_THRESHOLD:
        return "Approve"
    elif probability < DECLINE_THRESHOLD:
        return "Review"
    return "Decline"


def _explain_shap(application_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    X = _predictor._preprocess(application_df)  # same encode/scale pipeline as training

    explainer = shap.TreeExplainer(_predictor.model)
    shap_values = explainer.shap_values(X)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    sv = sv[0]  # single application row

    shap_df = pd.DataFrame({"feature": X.columns, "shap_value": sv})
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    top = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

    return [
        {
            "feature": row.feature,
            "shap_value": round(float(row.shap_value), 4),
            "direction": "increases risk" if row.shap_value > 0 else "decreases risk",
        }
        for row in top.itertuples()
    ]


def _retrieve_policy(top_shap_features: list[dict], probability: float, k: int = 3) -> list[dict]:
    feature_names = ", ".join(f["feature"] for f in top_shap_features)
    query = (
        f"Credit policy guidance for an application with high-risk probability "
        f"{probability:.2f}, primarily driven by: {feature_names}."
    )
    docs = _credit_policy_store.similarity_search(query, k=k)
    return [
        {"policy_text": d.page_content, "source": d.metadata.get("source", "unknown")}
        for d in docs
    ]


def cda_agent(state: WorkerState) -> dict:
    """
    Looks up the credit application (joined with customer info) from the
    shared database, runs the XGBoost classifier + SHAP TreeExplainer,
    retrieves the applicable credit policy rule, and produces:
    probability, top_shap_features, decision.
    """
    state = WorkerState.model_validate(state)  # coerce dict from Send into WorkerState
    application_id = state.input.get("application_id")
    application = get_credit_application(application_id)

    if application is None:
        result = {"agent": "CDA", "error": f"application_id {application_id} not found"}
        log_entry = {"agent": "CDA", "event": "cda_lookup_failed", "output": result}
        return {"completed_agents": [result], "audit_log": [log_entry]}

    feature_row = {k: v for k, v in application.items() if k not in NON_FEATURE_COLUMNS}
    application_df = pd.DataFrame([feature_row])

    probability, pred_label = _calculate_score(application_df)
    decision = _make_decision(probability)
    risk_label = _predictor.class_names[pred_label]

    top_shap_features = _explain_shap(application_df, top_n=5)
    policy_matches = _retrieve_policy(top_shap_features, probability)

    timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "agent": "CDA",
        "application_id": application_id,
        "customer_id": application.get("customer_id"),
        "probability": round(probability, 4),
        "risk_label": risk_label,
        "top_shap_features": top_shap_features,
        "credit_policy_context": policy_matches,
        "decision": decision,          # Approve / Review / Decline
        "source_input": application,
        "timestamp": timestamp,
    }

    log_entry = {"agent": "CDA", "event": "cda_assessment_complete", "output": result}

    return {"completed_agents": [result], "audit_log": [log_entry]}