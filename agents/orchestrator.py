"""Orchestrator (supervisor) node — fans out each assessment item via Send."""

from langgraph.types import Send

from state import GraphState


def route_to_workers(state: GraphState):
    """
    Supervisor/orchestrator node. Inspects each item in assessment_input and
    issues a Send to the correct worker node with only that item's payload.
    Returning a list of Send objects from a node (or conditional edge) tells
    LangGraph to run each one in parallel, each with its own WorkerState.
    """
    sends = []
    for item in state.assessment_input:
        item_type = item.get("type", state.assessment_type)

        if item_type == "KYC":
            sends.append(Send("kyc_agent", {"type": "KYC", "input": item}))
        elif item_type == "AML":
            sends.append(Send("aml_agent", {"type": "AML", "input": item}))
        elif item_type == "CDA":
            sends.append(Send("cda_agent", {"type": "CDA", "input": item}))
        elif item_type == "QUERY":
            sends.append(Send("db_analyst_agent", {"type": "QUERY", "input": item}))
        else:
            raise ValueError(f"Unknown assessment type: {item_type}")

    return sends
