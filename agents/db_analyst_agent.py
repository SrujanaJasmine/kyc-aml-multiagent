"""
DB Analyst agent — ad hoc natural-language questions answered via MCP.

Unlike kyc_agent/aml_agent/cda_agent (deterministic ID lookups via
customer_db.py), this agent handles open-ended questions that need an LLM
to decide what to query — e.g. "how many wire transfers over $10,000 has
CUST002 made this month". It reasons and calls execute_sql/search_objects
through the DBHub MCP server (read-only) until it has an answer.
"""

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from mcp_db_tools import get_db_tools
from state import WorkerState

# Built once and cached, since spinning up the MCP subprocess + tool list
# on every call would be slow.
_db_agent = None


async def _get_db_agent():
    global _db_agent
    if _db_agent is None:
        tools = await get_db_tools()
        model = ChatAnthropic(model="claude-sonnet-4-6")
        _db_agent = create_react_agent(
            model,
            tools,
            prompt=(
                "You are a compliance database analyst. You have read-only "
                "SQL access to the customers, credit_applications, and "
                "transactions tables. Answer precisely, and cite the exact "
                "rows/values you found."
            ),
        )
    return _db_agent


async def db_analyst_agent(state: WorkerState):
    """
    Runs a natural-language question through the MCP-backed ReAct agent.
    This node is async — routes containing a 'QUERY' item must be run via
    `await graph.ainvoke(...)` rather than the sync `graph.invoke(...)`.
    """
    state = WorkerState.model_validate(state)
    question = state.input.get("question", "")

    agent = await _get_db_agent()
    agent_result = await agent.ainvoke({"messages": [("user", question)]})
    answer = agent_result["messages"][-1].content

    result = {"agent": "DB_ANALYST", "question": question, "answer": answer}
    log_entry = {"agent": "DB_ANALYST", "event": "query_complete", "output": result}

    return {"completed_agents": [result], "audit_log": [log_entry]}
