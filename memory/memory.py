"""
memory.py
=========
System memory for the graph: LangGraph checkpoints for resuming a run, plus the
`agent_history` audit table. Both live in memory/system_memory.db, separate from
the business database.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

# Anchored to this file's own folder (memory/), not the process's current
# working directory — so it's always memory/system_memory.db regardless of
# where the script that imports this was run from.
DB_PATH = str(Path(__file__).resolve().parent / "system_memory.db")

# TWO connections to the same file, deliberately.
#
# SqliteSaver issues its own BEGIN/COMMIT around every checkpoint write. Python's
# sqlite3 also opens an implicit transaction on any INSERT unless told otherwise.
# Sharing one connection between the checkpointer and the audit agent's inserts
# means the two transaction managers collide, and LangGraph fails mid-run with
# "cannot start a transaction within a transaction" -- after the assessment work
# is already done.
#
# The checkpointer gets its own autocommit connection (isolation_level=None) so
# it is the only thing managing transactions on it. `conn` stays for
# agent_history, where we do want ordinary transactional inserts.
checkpoint_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
checkpointer = SqliteSaver(checkpoint_conn)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)


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