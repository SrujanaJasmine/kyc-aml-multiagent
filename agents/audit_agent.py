"""
audit_agent.py
==============
Persists the audit trail in two forms -- the queryable `agent_history` table in
memory/system_memory.db, and an append-only human-readable log at
memory/agent_history.txt.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.runnables import RunnableConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.state import GraphState  # noqa: E402
from memory.memory import conn  # noqa: E402

TEXT_LOG_PATH = _REPO_ROOT / "memory" / "agent_history.txt"
SEPARATOR = "=" * 80


def _verdict_of(agent: str, output: dict) -> str:
    """
    Pull the one line that says what this agent concluded.

    Per-agent rather than a chain of `or`s: each agent names its conclusion
    differently, so a generic fallback would silently log an empty verdict for
    whichever agent it failed to cover.
    """
    if "error" in output:
        return f"ERROR: {output['error']}"

    if agent == "KYC":
        return str(output.get("customer_status", ""))
    if agent == "CDA":
        decision = output.get("decision", "")
        prob = output.get("probability")
        return f"{decision} (p={prob})" if prob is not None else str(decision)
    if agent == "AML":
        action = output.get("recommended_action", "")
        prob = output.get("probability")
        return f"{action} (p={prob})" if prob is not None else str(action)
    if agent == "DB_ANALYST":
        return "query answered"
    if agent == "Explanation":
        return str(output.get("method", "report generated"))
    return str(output.get("event", ""))


def _input_summary_of(agent: str, output: dict) -> str:
    """Which record this assessment was about, from identifiers actually present."""
    for key in ("application_id", "transaction_id", "customer_id", "question"):
        if output.get(key):
            return f"{key}={output[key]}"
    return ""


def _detail_lines(agent: str, output: dict) -> list[str]:
    """A few agent-specific lines for the human-readable log."""
    if "error" in output:
        return [f"error      : {output['error']}"]

    lines: list[str] = []
    if agent == "KYC":
        rel = output.get("relationship", {})
        lines.append(f"customer   : {output.get('customer_id')} — {output.get('customer_status')}")
        lines.append(f"relationship: {rel.get('applications_on_file', 0)} application(s), "
                     f"{rel.get('transactions_on_file', 0):,} transaction(s)")
    elif agent == "CDA":
        lines.append(f"application: {output.get('application_id')}")
        lines.append(f"decision   : {output.get('decision')} — {output.get('decision_reason')}")
        breaches = output.get("breached_rules", [])
        lines.append(f"policy     : {len(breaches)} rule(s) breached"
                     + (f" [{', '.join(b['rule_id'] for b in breaches[:4])}]" if breaches else ""))
    elif agent == "AML":
        lines.append(f"transaction: {output.get('transaction_id')} "
                     f"(customer {output.get('customer_id')})")
        lines.append(f"verdict    : {output.get('verdict')}")
        lines.append(f"action     : {output.get('recommended_action')} — "
                     f"{output.get('action_reason')}")
        breaches = output.get("breached_rules", [])
        if breaches:
            lines.append(f"policy     : {', '.join(b['rule_id'] for b in breaches)}")
        roll = output.get("customer_rollup") or {}
        if roll:
            lines.append(f"roll-up    : {roll.get('flagged', 0)} flagged of "
                         f"{roll.get('transactions', 0)} in {roll.get('window_days')}d, "
                         f"SAR aggregate met: {roll.get('meets_sar_aggregate')}")
    return lines


def _write_text_log(thread_id: str, state: GraphState, run_time: str) -> None:
    """Append one readable block per run, ending with the final report."""
    TEXT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = [SEPARATOR,
             f"RUN  {thread_id:<40} {run_time}",
             SEPARATOR, ""]

    for entry in state.audit_log:
        agent = entry.get("agent", "unknown")
        output = entry.get("output", {}) or {}
        stamp = str(entry.get("timestamp", run_time))[11:19] or run_time
        parts.append(f"[{stamp}] {agent:<12} {entry.get('event', '')}")
        for line in _detail_lines(agent, output):
            parts.append(f"             {line}")
        parts.append("")

    report = getattr(state, "final_report", "") or ""
    if report:
        parts += ["--- FINAL REPORT " + "-" * 62, "", report, ""]

    parts.append("")
    with TEXT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def audit_agent(state: GraphState, config: RunnableConfig) -> dict:
    """
    Persist every accumulated audit_log entry to SQLite and to the text log.

    Returns a small status dict rather than `{}` so a logging failure is visible
    in the final state instead of disappearing.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "unknown")
    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written, errors = 0, []

    # --- structured trail ---------------------------------------------------
    try:
        for entry in state.audit_log:
            agent = entry.get("agent", "unknown")
            output = entry.get("output", {}) or {}
            conn.execute(
                """
                INSERT INTO agent_history
                    (thread_id, agent, input_summary, output_verdict, token_usage, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    agent,
                    _input_summary_of(agent, output)[:500],
                    _verdict_of(agent, output)[:500],
                    entry.get("token_usage", 0),
                    entry.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
                ),
            )
            written += 1
        conn.commit()
    except Exception as exc:
        errors.append(f"sqlite: {exc}")

    # --- human-readable trail ----------------------------------------------
    try:
        _write_text_log(thread_id, state, run_time)
    except Exception as exc:
        errors.append(f"text log: {exc}")

    return {"audit_log": [{
        "agent": "Audit",
        "event": "audit_persisted",
        "output": {"entries_written": written,
                   "text_log": str(TEXT_LOG_PATH),
                   "errors": errors},
    }]}


if __name__ == "__main__":
    # Exercise the writer without running the whole graph.
    from agents.state import GraphState as GS

    demo = GS(
        assessment_type="BATCH",
        assessment_input=[],
        completed_agents=[],
        audit_log=[
            {"agent": "KYC", "event": "kyc_assessment_complete",
             "output": {"agent": "KYC", "customer_id": "CUS_0x1000",
                        "customer_status": "Existing",
                        "relationship": {"applications_on_file": 8,
                                         "transactions_on_file": 371}}},
            {"agent": "AML", "event": "aml_assessment_complete",
             "output": {"agent": "AML", "transaction_id": "TRAIN_61669",
                        "customer_id": "CUS_0xa160", "probability": 0.4765,
                        "verdict": "Not flagged by the model",
                        "recommended_action": "Escalate for analyst review",
                        "action_reason": "1 regulatory threshold crossed",
                        "breached_rules": [{"rule_id": "SAR-5K"},
                                           {"rule_id": "LAYERING-CYCLE"}],
                        "customer_rollup": {"flagged": 0, "transactions": 3,
                                            "window_days": 30,
                                            "meets_sar_aggregate": False}}},
        ],
        final_report="# Compliance Assessment Report\n\n(demo run)",
    )

    out = audit_agent(demo, {"configurable": {"thread_id": "demo-run-001"}})
    status = out["audit_log"][0]["output"]
    print(f"entries written : {status['entries_written']}")
    print(f"text log        : {status['text_log']}")
    print(f"errors          : {status['errors'] or 'none'}")
    print("\n--- tail of the text log ---")
    print(TEXT_LOG_PATH.read_text(encoding="utf-8")[-900:])
