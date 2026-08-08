"""AML screening agent — scores a transaction for anomalous/laundering behavior."""

from customer_db import get_transaction, get_customer_transactions
from state import WorkerState


def aml_agent(state: WorkerState):
    """
    Looks up the flagged transaction plus recent history for that customer
    from the shared database, runs the Isolation Forest + XGBoost ensemble
    anomaly score, retrieves the closest FATF typology match, produces:
    anomaly_score, matched_typology.
    """
    state = WorkerState.model_validate(state)  # coerce dict from Send into WorkerState
    transaction_id = state.input.get("transaction_id")
    transaction = get_transaction(transaction_id)

    if transaction is None:
        result = {"agent": "AML", "error": f"transaction_id {transaction_id} not found"}
        log_entry = {"agent": "AML", "event": "aml_lookup_failed", "output": result}
        return {"completed_agents": [result], "audit_log": [log_entry]}

    # Recent history for the same customer gives the model context beyond
    # this single transaction (e.g. structuring / rapid succession patterns).
    history = get_customer_transactions(transaction["customer_id"], limit=20)

    # TODO: replace with real model inference + RAG retrieval, using
    # `transaction` fields (amount_received, payment_format, is_laundering, etc.)
    # + `history` as features
    result = {
        "agent": "AML",
        "transaction_id": transaction_id,
        "anomaly_score": 0.0,   # 0-1
        "matched_typology": None,
        "source_input": transaction,
        "recent_history_count": len(history),
    }

    log_entry = {"agent": "AML", "event": "aml_assessment_complete", "output": result}

    return {"completed_agents": [result], "audit_log": [log_entry]}
