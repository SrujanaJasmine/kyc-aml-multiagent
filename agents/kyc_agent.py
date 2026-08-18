"""
kyc_agent.py
============
KYC worker. Determines whether a party is a new customer, an existing one, or on
file but dormant, and summarises the depth of the existing relationship. It does
not perform sanctions screening and reports `screening_performed: false` so no
screening result can be inferred from its output.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.state import WorkerState  # noqa: E402
from database.customer_db import get_customer, get_customer_relationship  # noqa: E402

# Fields never forwarded. The result dict is handed to an LLM to write a report,
# so identifiers that serve no analytical purpose should not travel with it.
PII_FIELDS = {"ssn"}


def _tenure_days(first_seen: str | None, last_seen: str | None) -> int | None:
    if not first_seen or not last_seen:
        return None
    try:
        start = datetime.fromisoformat(str(first_seen))
        end = datetime.fromisoformat(str(last_seen))
        return max((end - start).days, 0)
    except (TypeError, ValueError):
        return None


def kyc_agent(state: WorkerState) -> dict:
    """Determine customer status and summarise the existing relationship."""
    state = WorkerState.model_validate(state)
    customer_id = state.input.get("customer_id")
    timestamp = datetime.now(timezone.utc).isoformat()

    if not customer_id:
        result = {"agent": "KYC", "error": "no customer_id supplied", "timestamp": timestamp}
        return {"completed_agents": [result],
                "audit_log": [{"agent": "KYC", "event": "kyc_input_invalid", "output": result}]}

    customer = get_customer(customer_id)

    # --- new customer -------------------------------------------------------
    if customer is None:
        result = {
            "agent": "KYC",
            "customer_id": customer_id,
            "customer_status": "New",
            "customer_on_file": False,
            "onboarding_required": True,
            "relationship": {"applications_on_file": 0, "transactions_on_file": 0},
            "notes": ("No record for this customer. Full customer due diligence and "
                      "identity verification are required before any account activity."),
            "screening_performed": False,
            "screening_note": ("Sanctions and watchlist screening is not implemented. "
                               "No screening result should be inferred from this output."),
            "timestamp": timestamp,
        }
        return {"completed_agents": [result],
                "audit_log": [{"agent": "KYC", "event": "kyc_new_customer", "output": result}]}

    # --- existing customer --------------------------------------------------
    relationship = get_customer_relationship(customer_id)
    tenure = _tenure_days(relationship.get("first_seen"), relationship.get("last_seen"))
    relationship["tenure_days"] = tenure

    has_activity = (relationship["applications_on_file"] > 0
                    or relationship["transactions_on_file"] > 0)
    status = "Existing" if has_activity else "Existing-Dormant"

    safe_customer = {k: v for k, v in customer.items() if k not in PII_FIELDS}

    if has_activity:
        notes = (f"Existing customer with {relationship['applications_on_file']} credit "
                 f"application(s) and {relationship['transactions_on_file']:,} transaction(s) "
                 f"on file"
                 + (f" over {tenure} days." if tenure is not None else "."))
    else:
        notes = ("Customer record exists but carries no applications or transactions. "
                 "Treat as dormant: the record may be stale or the relationship never "
                 "became active.")

    result = {
        "agent": "KYC",
        "customer_id": customer_id,
        "customer_status": status,
        "customer_on_file": True,
        "onboarding_required": False,
        "customer": safe_customer,
        "relationship": relationship,
        "notes": notes,
        "screening_performed": False,
        "screening_note": ("Sanctions and watchlist screening is not implemented. "
                           "No screening result should be inferred from this output."),
        "timestamp": timestamp,
    }

    return {"completed_agents": [result],
            "audit_log": [{"agent": "KYC", "event": "kyc_assessment_complete",
                           "output": result}]}


if __name__ == "__main__":
    from database.customer_db import get_connection

    conn = get_connection()
    existing = conn.execute("SELECT customer_id FROM customers LIMIT 1").fetchone()
    conn.close()

    for label, cid in (("existing", existing[0] if existing else None),
                       ("new", "CUS_DOES_NOT_EXIST")):
        if cid is None:
            continue
        res = kyc_agent({"type": "KYC", "input": {"customer_id": cid}})["completed_agents"][0]
        print(f"--- {label} ---")
        print(f"  customer_id : {res['customer_id']}")
        print(f"  status      : {res.get('customer_status')}")
        print(f"  onboarding  : {res.get('onboarding_required')}")
        print(f"  relationship: {res.get('relationship')}")
        print(f"  notes       : {res.get('notes')}")
        print()
