"""
System memory (persistence) for the graph — separate from customer_db.py,
which holds the business data agents look up. This module backs:

  1. `checkpoints` / `writes` tables (managed automatically by SqliteSaver)
     -> full GraphState snapshots per thread_id, so a run can be resumed
        or inspected step by step (same pattern as the LangGraph Academy
        chatbot-external-memory notebook).
  2. `agent_history` table (managed by us, written by audit_agent.py)
     -> one queryable row per agent call: agent name, input summary,
        output verdict, timestamp, token usage. This is the system-wide
        "usage history of every agent" the project spec (M8) asks for.

Both live in the same SQLite file, system_memory.db.
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "system_memory.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)


def init_agent_history_table(connection: sqlite3.Connection = conn) -> None:
    """Create the audit agent's history table if it doesn't exist yet."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            agent TEXT NOT NULL,
            input_summary TEXT,
            output_verdict TEXT,
            token_usage INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        )
        """
    )
    connection.commit()


init_agent_history_table()


def get_agent_history(agent: str | None = None, thread_id: str | None = None):
    """Convenience query helper — e.g. get_agent_history(agent='AML')."""
    query = "SELECT * FROM agent_history WHERE 1=1"
    params: list = []
    if agent:
        query += " AND agent = ?"
        params.append(agent)
    if thread_id:
        query += " AND thread_id = ?"
        params.append(thread_id)
    query += " ORDER BY timestamp DESC"
    return conn.execute(query, params).fetchall()
