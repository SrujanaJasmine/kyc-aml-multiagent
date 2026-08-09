# FRAML — Multi-Agent Fraud, AML & Credit Risk Assessment

A LangGraph orchestrator-worker system for financial compliance. A supervisor node
fans a batch of assessment items out to specialist agents in parallel, each agent
looks up what it needs from a shared SQLite database, and the results converge into
a single plain-English compliance report plus a queryable audit trail.

## Architecture

```
                    START
                      │
              route_to_workers          (orchestrator.py — Send fan-out)
        ┌─────────┬────┴─────┬──────────────┐
        ▼         ▼          ▼              ▼
   kyc_agent  aml_agent  cda_agent   db_analyst_agent
        └─────────┴────┬─────┴──────────────┘
                       ▼
              explanation_agent          (synthesizes final_report)
                       ▼
                 audit_agent             (persists to agent_history)
                       ▼
                      END
```

| Agent | Role |
|---|---|
| `kyc_agent` | Screens a customer against sanctions/FATF guidance → risk level, matched entities, due-diligence level |
| `aml_agent` | Scores a transaction for laundering behaviour using the transaction plus recent customer history |
| `cda_agent` | Credit decision assessment from the XGBoost model in `ml_models/` |
| `db_analyst_agent` | Open-ended natural-language questions, answered via read-only SQL through the DBHub MCP server |
| `explanation_agent` | Merges all worker outputs into one compliance report |
| `audit_agent` | Writes one row per agent call into `agent_history` |

### Two databases, deliberately separate

- **`customer_data.db`** — *what* the system reasons about: `customers`, `credit_applications`, `transactions`. Built by `agents/customer_db.py`.
- **`system_memory.db`** — *how* the system ran: LangGraph checkpoints plus the `agent_history` audit table. Managed by `agents/memory.py`.

Neither is committed — both are rebuilt locally from the source CSVs.

## Repository layout

```
.
├── agents/
│   ├── state.py               GraphState / WorkerState schemas
│   ├── orchestrator.py        Send-based fan-out
│   ├── kyc_agent.py           KYC worker
│   ├── aml_agent.py           AML worker
│   ├── cda_agent.py           Credit worker
│   ├── db_analyst_agent.py    MCP-backed ad hoc query worker
│   ├── explanation_agent.py   Report synthesis
│   ├── audit_agent.py         Audit persistence
│   ├── memory.py              Checkpointer + agent_history
│   ├── customer_db.py         Schema + CSV loaders + accessors
│   ├── mcp_db_tools.py        DBHub MCP config (read-only)
│   └── graph.py               Wires everything into one StateGraph
├── ml_models/
│   ├── credit_risk_model.py   XGBoost credit risk training
│   ├── cda_model.ipynb
│   └── artifacts/             scaler, feature columns, threshold, model
├── EDA/                       Notebooks: credit EDA, transaction simulation
├── data/                      Datasets (gitignored — see data/README.md)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/<username>/FRAML_PROJECT.git
cd FRAML_PROJECT

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Node.js is also required — `mcp_db_tools.py` launches the DBHub MCP server via `npx`.

Then copy `.env.example` to `.env` and add your `GROQ_API_KEY`
(free key from [console.groq.com/keys](https://console.groq.com/keys)).

## Data

The datasets are not in this repo (~22 GB). See [`data/README.md`](data/README.md)
for sources and the expected directory layout, then build the database:

```bash
cd agents
python customer_db.py
```

## Running

```bash
cd agents
python graph.py
```

This seeds a demo customer/application/transaction, runs a mixed `BATCH`
assessment, prints the final report, and dumps `agent_history`.

Runs are namespaced by `thread_id`, so any run can be resumed or inspected:

```python
config = {"configurable": {"thread_id": "run-2026-07-31-001"}}
graph.get_state(config)
graph.get_state_history(config)
```

Batches containing a `QUERY` item hit the async `db_analyst_agent` — use
`await graph.ainvoke(...)` rather than `graph.invoke(...)`.

## Status

The graph, state, routing, persistence, and database layers are complete.
`kyc_agent`, `aml_agent`, and `explanation_agent` currently return stubbed
verdicts (marked `TODO`) pending real sanctions screening, model inference,
and RAG retrieval.
