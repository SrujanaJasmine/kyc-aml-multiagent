"""
graph.py
========
Builds the LangGraph orchestrator-worker graph that wires the KYC, credit and AML
workers to the explanation and audit agents. Run directly, it executes a
20-scenario end-to-end demonstration on the held-out `system` split and writes a
report per scenario to reports/.
"""

import config  # noqa: F401  loads .env before any agent reads os.environ
from langgraph.graph import StateGraph, START, END

from agents.state import GraphState
from memory.memory import checkpointer, get_agent_history  # noqa: F401 (re-exported for convenience)
from agents.orchestrator import route_to_workers
from agents.kyc_agent import kyc_agent
from agents.aml_agent import aml_agent
from agents.cda_agent import cda_agent
from agents.explanation_agent import explanation_agent
from agents.audit_agent import audit_agent

builder = StateGraph(GraphState)

builder.add_node("kyc_agent", kyc_agent)
builder.add_node("aml_agent", aml_agent)
builder.add_node("cda_agent", cda_agent)
builder.add_node("explanation_agent", explanation_agent)
builder.add_node("audit_agent", audit_agent)

# Orchestrator: START fans out to workers via Send
builder.add_conditional_edges(
    START, route_to_workers, ["kyc_agent", "aml_agent", "cda_agent"]
)

# All workers converge on the Explanation agent
builder.add_edge("kyc_agent", "explanation_agent")
builder.add_edge("aml_agent", "explanation_agent")
builder.add_edge("cda_agent", "explanation_agent")

# Explanation -> Audit -> End
builder.add_edge("explanation_agent", "audit_agent")
builder.add_edge("audit_agent", END)

# checkpointer=... is what gives the graph external, resumable memory,
# exactly like the notebook's workflow.compile(checkpointer=memory) call —
# but here it's persisting a multi-agent assessment run instead of a single
# chat thread.
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # =====================================================================
    # End-to-end demonstration.
    #
    # Runs a batch of complete assessments through the whole graph using the
    # `system` split -- the partition held back by ml_models/data_splits.py
    # and never seen by the credit model or the AML model during training.
    # Every verdict below is therefore produced on unseen data.
    #
    # Cases are selected by CRITERIA rather than hardcoded IDs, so the demo
    # survives regeneration of the underlying data, and they are chosen to
    # span outcomes rather than to flatter the system: customers with
    # injected laundering, customers with poor credit profiles, clean
    # customers, and parties with no record at all.
    #
    # Nothing is written to the business database. Reports go to
    # reports/<run>/ as Markdown and plain text, with an index summarising
    # every scenario. The audit trail goes to memory/. Neither is printed --
    # this console output is a run log, and the reports are the deliverable.
    # =====================================================================
    import argparse
    import traceback
    from collections import Counter
    from datetime import datetime, timezone

    from agents.audit_agent import TEXT_LOG_PATH
    from agents.explanation_agent import REPORT_DIR, write_report
    from database.customer_db import get_connection
    from memory.memory import DB_PATH as MEMORY_DB_PATH
    from ml_models.data_splits import assign_split

    SPLIT = "system"

    parser = argparse.ArgumentParser(description="FRAML end-to-end demonstration")
    parser.add_argument("--cases", type=int, default=20,
                        help="number of scenarios to run (default 20)")
    args = parser.parse_args()

    def check_indexes(conn) -> list[str]:
        """
        Report missing performance indexes WITHOUT creating them.

        Running a demonstration should never write to the business database --
        the same objection that applies to seeding a fictional customer applies
        to building an index, however useful. Creating one here is also a
        multi-hundred-megabyte rewrite of a 4.5M row table, and a failure
        partway through leaves a hot journal that blocks every subsequent read.
        Detect, advise, let the operator decide.
        """
        wanted = {"idx_txn_from": "transactions(from_account)",
                  "idx_txn_to": "transactions(to_account)",
                  "idx_txn_customer": "transactions(customer_id)"}
        try:
            existing = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}
        except Exception:
            return []
        return [n for n in wanted if n not in existing]

    def select_cases(conn, want: int) -> list[dict]:
        """
        Assemble a spread of scenarios across four deliberately different
        profiles. A demonstration made only of clean approvals would prove
        nothing; the point is to show the system separating outcomes and, where
        the model and the policy rules disagree, routing to a human.
        """
        quota = {"laundering": max(want * 3 // 10, 1),
                 "high_risk_credit": max(want * 3 // 10, 1),
                 "clean": max(want * 3 // 10, 1)}
        quota["new"] = max(want - sum(quota.values()), 1)

        used: set[str] = set()
        cases: list[dict] = []

        def add(kind: str, label: str, cid: str, items: list[dict]) -> None:
            n = len(cases) + 1
            cases.append({"kind": kind, "name": label, "customer_id": cid,
                          "stem": f"{n:02d}_{kind}", "items": items})
            used.add(cid)

        def app_for(cid: str, worst: bool):
            order = "DESC" if worst else "ASC"
            row = conn.execute(
                f"SELECT application_id FROM credit_applications WHERE customer_id = ? "
                f"ORDER BY Credit_Score {order} LIMIT 1", (cid,)).fetchone()
            return row[0] if row else None

        # --- 1. customers with injected laundering ---------------------------
        for cid, txn_id in conn.execute(
                "SELECT customer_id, transaction_id FROM transactions WHERE is_laundering = 1"):
            if len([c for c in cases if c["kind"] == "laundering"]) >= quota["laundering"]:
                break
            if cid in used or assign_split(cid) != SPLIT:
                continue
            app = app_for(cid, worst=True)
            if not app:
                continue
            add("laundering", "Laundering activity on file", cid,
                [{"type": "KYC", "customer_id": cid},
                 {"type": "CDA", "application_id": app},
                 {"type": "AML", "transaction_id": txn_id, "customer_id": cid}])

        flagged = {c["customer_id"] for c in cases}

        # --- 2. poor credit profiles, no laundering --------------------------
        for cid, app in conn.execute(
                "SELECT customer_id, application_id FROM credit_applications "
                "WHERE Credit_Score = 1"):
            if len([c for c in cases if c["kind"] == "high_risk_credit"]) >= quota["high_risk_credit"]:
                break
            if cid in used or cid in flagged or assign_split(cid) != SPLIT:
                continue
            txn = conn.execute("SELECT transaction_id FROM transactions "
                               "WHERE customer_id = ? AND is_laundering = 0 LIMIT 1",
                               (cid,)).fetchone()
            if not txn:
                continue
            add("high_risk_credit", "Elevated credit risk", cid,
                [{"type": "KYC", "customer_id": cid},
                 {"type": "CDA", "application_id": app},
                 {"type": "AML", "transaction_id": txn[0], "customer_id": cid}])

        # --- 3. clean profiles ------------------------------------------------
        for (cid,) in conn.execute("SELECT customer_id FROM customers"):
            if len([c for c in cases if c["kind"] == "clean"]) >= quota["clean"]:
                break
            if cid in used or assign_split(cid) != SPLIT:
                continue
            laundered = conn.execute("SELECT 1 FROM transactions WHERE customer_id = ? "
                                     "AND is_laundering = 1 LIMIT 1", (cid,)).fetchone()
            if laundered:
                continue
            app = app_for(cid, worst=False)
            txn = conn.execute("SELECT transaction_id FROM transactions "
                               "WHERE customer_id = ? AND is_laundering = 0 LIMIT 1",
                               (cid,)).fetchone()
            if not (app and txn):
                continue
            add("clean", "Established customer, clean profile", cid,
                [{"type": "KYC", "customer_id": cid},
                 {"type": "CDA", "application_id": app},
                 {"type": "AML", "transaction_id": txn[0], "customer_id": cid}])

        # --- 4. parties with no record ---------------------------------------
        for i in range(quota["new"]):
            cid = f"CUS_DEMO_NEW_{i + 1:02d}"
            add("new", "New customer, not on file", cid,
                [{"type": "KYC", "customer_id": cid}])

        return cases[:want]

    def verdicts_of(completed: list[dict]) -> dict[str, str]:
        out = {}
        for entry in completed:
            agent = entry.get("agent", "?")
            if "error" in entry:
                out[agent] = "error"
            elif agent == "KYC":
                out[agent] = str(entry.get("customer_status"))
            elif agent == "CDA":
                out[agent] = str(entry.get("decision"))
            elif agent == "AML":
                out[agent] = str(entry.get("recommended_action"))
        return out

    # ---------------------------------------------------------------------
    started = datetime.now(timezone.utc)
    # Seconds precision matters: the thread_id derives from this stamp, and
    # LangGraph RESUMES an existing thread rather than starting fresh. Two runs
    # inside the same minute would share a thread and the operator.add reducer
    # would append to the previous run's results.
    stamp = started.strftime("%Y%m%dT%H%M%S")
    run_dir = f"run_{stamp}"

    conn = get_connection()
    missing_indexes = check_indexes(conn)
    n_system = sum(1 for (c,) in conn.execute("SELECT customer_id FROM customers")
                   if assign_split(c) == SPLIT)

    print("FRAML — end-to-end demonstration")
    print(f"Data: {SPLIT} split ({n_system:,} customers, held out from all model training)")
    if missing_indexes:
        print(f"\nNote: {len(missing_indexes)} index(es) missing ({', '.join(missing_indexes)}).")
        print("      Each AML assessment will scan the full transaction table; with")
        print("      20 scenarios that is several minutes. See the README for the")
        print("      one-off command to create them.")
    print()

    try:
        cases = select_cases(conn, args.cases)
    finally:
        conn.close()

    print(f"Selected {len(cases)} scenarios: "
          + ", ".join(f"{n}x {k}" for k, n in Counter(c["kind"] for c in cases).items()))
    print()

    history_before = len(get_agent_history())
    rows, tally = [], {"CDA": Counter(), "AML": Counter(), "KYC": Counter()}

    for i, case in enumerate(cases, start=1):
        thread_id = f"demo-{stamp}-{case['stem']}"
        print(f"[{i:>2}/{len(cases)}] {case['name']:<36} {case['customer_id']:<16}", end="", flush=True)

        try:
            final_state = graph.invoke(
                {"assessment_type": "BATCH", "assessment_input": case["items"]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception:
            print("  FAILED")
            traceback.print_exc()
            continue

        verdicts = verdicts_of(final_state.get("completed_agents", []))
        for agent, verdict in verdicts.items():
            tally.setdefault(agent, Counter())[verdict] += 1
        print("  " + " · ".join(f"{a} {v}" for a, v in verdicts.items()))

        banner = (f"> **Demonstration run** — scenario {i} of {len(cases)}: {case['name']}  \n"
                  f"> Data source: `{SPLIT}` split, held out from all model training.  \n"
                  f"> Thread `{thread_id}` · generated {started:%Y-%m-%d %H:%M} UTC.")
        md_path, _ = write_report(final_state.get("final_report", ""),
                                  case["stem"], banner, subdir=run_dir)
        rows.append((i, case["name"], case["customer_id"], verdicts, md_path.name))

    # --- index ------------------------------------------------------------
    index = [
        "# FRAML — End-to-End Demonstration", "",
        f"Generated {started:%Y-%m-%d %H:%M} UTC · {len(rows)} scenarios completed", "",
        f"**Data:** `{SPLIT}` split — {n_system:,} customers held back by "
        f"`ml_models/data_splits.py` and never seen by the credit model or the AML "
        f"model during training. Every verdict below is produced on unseen data.", "",
        "## Scenarios", "",
        "| # | Scenario | Customer | KYC | Credit | AML | Report |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, name, cid, verdicts, fname in rows:
        index.append(f"| {n} | {name} | `{cid}` | {verdicts.get('KYC', '—')} "
                     f"| {verdicts.get('CDA', '—')} | {verdicts.get('AML', '—')} "
                     f"| [{fname}]({fname}) |")

    index += ["", "## Outcome distribution", ""]
    for agent in ("KYC", "CDA", "AML"):
        counts = tally.get(agent)
        if counts:
            index.append(f"- **{agent}** — "
                         + ", ".join(f"{v} {k}" for k, v in counts.most_common()))
    index += ["", "## How to read this", "",
              "Each scenario ran the full graph: the orchestrator fans the assessment "
              "items out to the KYC, credit and AML workers in parallel, their findings "
              "merge, the explanation agent writes the report, and the audit agent "
              "persists the trail. Where the model and the deterministic policy rules "
              "disagree, the case is routed to a human rather than auto-decided — that "
              "disagreement is visible in several scenarios below.", ""]

    write_report("\n".join(index), "00_INDEX", "", subdir=run_dir)

    entries = len(get_agent_history()) - history_before
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    print()
    print(f"Completed {len(rows)}/{len(cases)} scenarios in {elapsed/60:.1f} minutes")
    print(f"Reports saved — Markdown + plain text, with an index")
    print(f"  {REPORT_DIR / run_dir}")
    print(f"  start here: 00_INDEX.md")
    print()
    print(f"Audit history saved — {entries} entries this run")
    print(f"  {MEMORY_DB_PATH}  (agent_history table)")
    print(f"  {TEXT_LOG_PATH}")
