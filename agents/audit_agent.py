"""Audit agent — persists every agent's log entries into system memory."""

import time

from langchain_core.runnables import RunnableConfig

from memory import conn
from state import GraphState


def audit_agent(state: GraphState, config: RunnableConfig):
    """
    Persists every accumulated audit_log entry into the `agent_history`
    SQLite table: agent name, a short input summary, the output verdict,
    token usage, and a timestamp. This is the queryable system-wide
    "usage history of every agent" (separate from the graph checkpoints,
    which store full state snapshots for resuming a run).
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    for entry in state.audit_log:
        output = entry.get("output", {})
        conn.execute(
            """
            INSERT INTO agent_history
                (thread_id, agent, input_summary, output_verdict, token_usage, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                entry.get("agent", "unknown"),
                str(output.get("source_input", ""))[:500],  # short input summary
                str(
                    output.get("decision")
                    or output.get("risk_level")
                    or output.get("event", "")
                ),
                entry.get("token_usage", 0),
                entry.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            ),
        )
    conn.commit()

    return {}
