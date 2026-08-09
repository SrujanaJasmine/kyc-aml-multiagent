"""
MCP configuration for giving agents live access to customer_data.db through
DBHub, a general-purpose database MCP server (github.com/bytebase/dbhub).

DBHub is launched as a subprocess over stdio (no separate server process to
manage) and exposes two MCP tools:
    - execute_sql:    run a SQL query, with transaction support and
                       safety controls
    - search_objects: explore schemas/tables/columns without needing to
                       dump full table definitions into context

--readonly is important here: it makes DBHub reject any write operation at
the engine level, so an LLM-driven agent can explore/query the database but
can never accidentally UPDATE or DELETE a row. Writes should still go
through the explicit functions in customer_db.py (update_application_status,
flag_transaction), not through this MCP path.
"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

DB_PATH = os.path.abspath("customer_data.db").replace("\\", "/")

MCP_SERVERS = {
    "customer_db": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@bytebase/dbhub@latest",
            "--transport", "stdio",
            "--readonly",                 # hard block on INSERT/UPDATE/DELETE
            "--dsn", f"sqlite:///{DB_PATH}",
        ],
    }
}


async def get_db_tools():
    """
    Connects to the DBHub MCP server and returns its tools as LangChain
    BaseTool objects — ready to hand to bind_tools() or wrap in a ToolNode.
    """
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()