"""KYC onboarding agent — screens a customer against sanctions/FATF guidance."""

from customer_db import get_customer
from state import WorkerState


def kyc_agent(state: WorkerState):
    """
    Looks up the customer record from the shared customer database, screens
    it against the OFAC SDN list, retrieves FATF onboarding guidance from the
    KYC/FATF FAISS store, and produces: risk_level, matched_entities,
    due_diligence_level.
    """
    state = WorkerState.model_validate(state)  # coerce dict from Send into WorkerState
    customer_id = state.input.get("customer_id")
    customer = get_customer(customer_id)

    if customer is None:
        result = {"agent": "KYC", "error": f"customer_id {customer_id} not found"}
        log_entry = {"agent": "KYC", "event": "kyc_lookup_failed", "output": result}
        return {"completed_agents": [result], "audit_log": [log_entry]}

    # TODO: replace with real sanctions screening + RAG retrieval, using
    # `customer` fields (name, ssn, age, occupation) as screening inputs
    result = {
        "agent": "KYC",
        "customer_id": customer_id,
        "risk_level": "Medium",           # Low / Medium / High
        "matched_entities": [],
        "due_diligence_level": "Standard",  # Standard / Enhanced
        "source_input": customer,
    }

    log_entry = {"agent": "KYC", "event": "kyc_assessment_complete", "output": result}

    return {"completed_agents": [result], "audit_log": [log_entry]}
