"""
Builds and compiles the multi-agent orchestrator-worker graph.

File map (one file per agent, so each can be edited/tested independently):
    state.py              GraphState / WorkerState schemas
    memory.py              system_memory.db: checkpointer + agent_history table
    orchestrator.py        route_to_workers (the Send fan-out)
    kyc_agent.py            KYC specialist worker
    aml_agent.py            AML specialist worker
    cda_agent.py            CDA (credit) specialist worker
    db_analyst_agent.py     MCP-backed ad hoc query worker
    explanation_agent.py    synthesizes all worker outputs into final_report
    audit_agent.py          persists audit_log into agent_history
    graph.py (this file)    wires all of the above into one StateGraph
"""

from langgraph.graph import StateGraph, START, END

from state import GraphState
from memory import checkpointer, get_agent_history  # noqa: F401 (re-exported for convenience)
from orchestrator import route_to_workers
from kyc_agent import kyc_agent
from aml_agent import aml_agent
from cda_agent import cda_agent
from db_analyst_agent import db_analyst_agent
from explanation_agent import explanation_agent
from audit_agent import audit_agent

builder = StateGraph(GraphState)

builder.add_node("kyc_agent", kyc_agent)
builder.add_node("aml_agent", aml_agent)
builder.add_node("cda_agent", cda_agent)
builder.add_node("db_analyst_agent", db_analyst_agent)
builder.add_node("explanation_agent", explanation_agent)
builder.add_node("audit_agent", audit_agent)

# Orchestrator: START fans out to workers via Send
builder.add_conditional_edges(
    START, route_to_workers, ["kyc_agent", "aml_agent", "cda_agent", "db_analyst_agent"]
)

# All workers converge on the Explanation agent
builder.add_edge("kyc_agent", "explanation_agent")
builder.add_edge("aml_agent", "explanation_agent")
builder.add_edge("cda_agent", "explanation_agent")
builder.add_edge("db_analyst_agent", "explanation_agent")

# Explanation -> Audit -> End
builder.add_edge("explanation_agent", "audit_agent")
builder.add_edge("audit_agent", END)

# checkpointer=... is what gives the graph external, resumable memory,
# exactly like the notebook's workflow.compile(checkpointer=memory) call —
# but here it's persisting a multi-agent assessment run instead of a single
# chat thread.
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from customer_db import init_db, get_connection

    init_db()

    # Quick demo seed matching the real schema (customer_db.py's own CSV
    # loaders — load_customers_from_csv / load_credit_applications_from_csv /
    # load_transactions_from_csv — are what you'd use with the actual
    # credit_dataset.csv / customer_transactions_train.csv in production).
    conn = get_connection()
    conn.execute(
        """INSERT OR IGNORE INTO customers
           (customer_id, name, ssn, age, occupation, annual_income,
            monthly_inhand_salary, num_bank_accounts, num_credit_card)
           VALUES ('CUST001', 'Jane Doe', '123-45-6789', 28, 'Engineer',
                   72000, 5500, 2, 3)"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO credit_applications
           (application_id, customer_id, month, outstanding_debt, credit_score)
           VALUES ('APP456', 'CUST001', 'Jan', 1200.0, 710)"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO transactions
           (transaction_id, customer_id, timestamp, amount_received,
            receiving_currency, payment_format, is_laundering)
           VALUES ('TXN123', 'CUST001', '2026-07-28T14:32:00', 15000,
                   'US Dollar', 'Wire', 0)"""
    )
    conn.commit()
    conn.close()

    # The orchestrator now only needs IDs — each worker looks up the full
    # record it needs from customer_data.db itself.
    example_state = {
        "assessment_type": "BATCH",
        "assessment_input": [
            {"type": "KYC", "customer_id": "CUST001"},
            {"type": "AML", "transaction_id": "TXN123"},
            {"type": "CDA", "application_id": "APP456"},
        ],
    }

    # thread_id namespaces this run so it can be resumed/inspected later,
    # e.g. via graph.get_state(config) or graph.get_state_history(config).
    config = {"configurable": {"thread_id": "run-2026-07-31-001"}}

    final_state = graph.invoke(example_state, config=config)
    print(final_state["final_report"])

    print("\n--- agent_history (queryable system memory) ---")
    for row in get_agent_history():
        print(row)
