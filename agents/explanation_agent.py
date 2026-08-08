"""Explanation agent — synthesizes all worker outputs into one final report."""

from state import GraphState


def explanation_agent(state: GraphState):
    """
    Reads GraphState.completed_agents (all worker outputs gathered so far)
    and synthesizes a structured, plain-English compliance report, citing
    top SHAP features and matched typologies where present.
    """
    # TODO: replace with an LLM call that synthesizes state.completed_agents
    sections = []
    for agent_output in state.completed_agents:
        sections.append(f"[{agent_output.get('agent')}] {agent_output}")

    report = "COMPLIANCE REPORT\n" + "\n".join(sections)

    log_entry = {"agent": "Explanation", "event": "report_generated"}

    return {"final_report": report, "audit_log": [log_entry]}
