"""
explanation_agent.py
====================
Turns the workers' structured findings into a written compliance report: a factual
verdict table assembled in code, followed by an LLM-written narrative. Falls back
to a fully deterministic report when no LLM is reachable.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import config  # loads .env before anything reads os.environ
from agents.state import GraphState

REPORT_DIR = config.REPO_ROOT / "reports"

GROQ_MODEL = config.GROQ_MODEL

REPORT_INSTRUCTIONS = """You are a compliance analyst writing an assessment report for a bank's credit and financial-crime review team.

You will receive structured findings from specialist agents (KYC, AML, Credit Assessment, and ad hoc database queries) as JSON. Write the report they imply.

Ground rules:
- Use only the findings provided. Never introduce a risk factor, rule, statistic, or policy that is not in the JSON. If something important is missing, say it is not available.
- The decision and the rule breaches were determined by deterministic logic upstream. Report them as settled; do not re-argue or overturn them.
- Translate model output into plain language. "P(High Risk) 0.62 against a 0.30 threshold" should read as roughly twice the level at which the bank flags an application.
- SHAP values indicate which features drove this specific score. Name the features in business terms and say which direction each pushed the decision.
- For each breached rule, state the observed value, the threshold, and why the standard exists.
- Be direct about uncertainty. Rules listed as not evaluated mean missing data, which weakens the assessment — say so.

A factual header and a verdict table are prepended to your output automatically. Do NOT repeat
them — no title, no restating the scores in a table. Begin directly at "## Summary".

Structure the report exactly as:

## Summary
Two or three sentences: what was assessed, the outcome, the single most important reason.

## Credit Assessment
Model score, threshold, decision and its reason. Then the risk drivers from SHAP.

## Policy and Standards Breaches
One subsection per breached rule: observed value vs threshold, and its significance. If none were breached, say so plainly.

## KYC and AML Findings
Only if those agents ran. Otherwise omit this section entirely.

## Data Quality Caveats
Anything not evaluated, missing, or errored. Omit if there is nothing to report.

## Recommended Next Steps
Concrete actions for the reviewer, tied to specific findings.

Write in clear professional prose. No emoji. Do not pad."""


def _verdict_table(completed_agents: list[dict]) -> str:
    """
    A factual header written by code, not by the model.

    The facts a reviewer checks first — which customer, which decision, what
    score against what threshold — must be exactly right, and an LLM asked to
    restate numbers in prose will occasionally round, reorder or drop one. So
    the header is assembled deterministically and the model writes only the
    narrative beneath it. Belt and braces: the prompt also forbids inventing
    figures, but a table that cannot be paraphrased is a stronger guarantee
    than an instruction not to paraphrase.
    """
    subject = ""
    for output in completed_agents:
        if output.get("customer_id"):
            cust = output.get("customer") or {}
            name = cust.get("name")
            occupation = cust.get("occupation")
            subject = output["customer_id"]
            if name:
                subject += f" ({name}" + (f", {occupation}" if occupation else "") + ")"
            break

    lines = ["# Compliance Assessment Report", ""]
    lines += ["| | |", "|---|---|"]
    if subject:
        lines.append(f"| Customer | {subject} |")
    lines.append(f"| Assessed | {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC |")
    lines.append(f"| Agents run | {', '.join(o.get('agent', '?') for o in completed_agents)} |")
    lines += ["", "## Verdicts at a glance", "",
              "| Agent | Verdict | Detail |", "|---|---|---|"]

    for output in completed_agents:
        agent = output.get("agent", "?")
        if "error" in output:
            lines.append(f"| {agent} | Error | {output['error']} |")
            continue

        if agent == "KYC":
            rel = output.get("relationship", {})
            detail = (f"{rel.get('applications_on_file', 0)} application(s), "
                      f"{rel.get('transactions_on_file', 0):,} transaction(s)")
            if rel.get("tenure_days") is not None:
                detail += f", {rel['tenure_days']} days on file"
            lines.append(f"| KYC | {output.get('customer_status', '')} | {detail} |")

        elif agent == "CDA":
            breaches = output.get("breached_rules", [])
            detail = (f"risk {output.get('probability')} vs "
                      f"{output.get('model_threshold')} threshold; "
                      f"{len(breaches)} rule(s) breached")
            if breaches:
                detail += f" ({', '.join(b['rule_id'] for b in breaches[:4])})"
            lines.append(f"| CDA | {output.get('decision', '')} | {detail} |")

        elif agent == "AML":
            breaches = output.get("breached_rules", [])
            roll = output.get("customer_rollup") or {}
            detail = f"score {output.get('probability')} vs {output.get('model_threshold')}"
            if breaches:
                detail += f"; {', '.join(b['rule_id'] for b in breaches)}"
            if roll:
                detail += (f"; roll-up {roll.get('flagged', 0)}/{roll.get('transactions', 0)} "
                           f"flagged, SAR aggregate {roll.get('meets_sar_aggregate')}")
            lines.append(f"| AML | {output.get('recommended_action', '')} | {detail} |")

        else:
            lines.append(f"| {agent} | — | {output.get('answer', '')} |")

    lines.append("")
    return "\n".join(lines)


def _rule_table(breaches: list[dict]) -> list[str]:
    if not breaches:
        return ["No policy rules were breached.", ""]
    rows = ["| Rule | Severity | Authority | Observed | Limit |",
            "|---|---|---|---|---|"]
    rows += [f"| {b['rule_id']} — {b['title']} | {b['severity']} | {b.get('authority','')} "
             f"| {b['observed']} | {b['threshold']} |" for b in breaches]
    return rows + [""]


def _deterministic_report(state: GraphState) -> str:
    """
    Structured fallback that needs no API.

    This is not a degraded placeholder to be tolerated -- it is what gets
    produced whenever the network is down, the key is missing, or the rate limit
    is hit, which means it can reach exactly the same reader as the LLM version.
    Everything below is assembled from the same findings the LLM would receive
    and formatted for a person rather than a debugger.
    """
    by_agent = {o.get("agent"): o for o in state.completed_agents}
    lines = [_verdict_table(state.completed_agents), "## Summary", ""]

    kyc, cda, aml = by_agent.get("KYC"), by_agent.get("CDA"), by_agent.get("AML")
    summary = []
    if kyc and "error" not in kyc:
        summary.append(f"The customer is **{kyc.get('customer_status', 'unknown')}** on the "
                       f"bank's records")
    if cda and "error" not in cda:
        summary.append(f"the credit application was assessed **{cda.get('decision')}** with "
                       f"{len(cda.get('breached_rules', []))} policy rule(s) breached")
    if aml and "error" not in aml:
        summary.append(f"financial-crime screening recommends **{aml.get('recommended_action')}**")
    lines.append((", ".join(summary) + "." if summary else
                  "No agent produced an assessable result.").capitalize())
    lines.append("")

    # --- credit -------------------------------------------------------------
    if cda:
        lines += ["## Credit Assessment", ""]
        if "error" in cda:
            lines += [f"Assessment failed: {cda['error']}", ""]
        else:
            lines += [
                f"Application `{cda.get('application_id')}` scored "
                f"**{cda.get('probability')}** against a decision threshold of "
                f"{cda.get('model_threshold')}, giving a verdict of "
                f"**{cda.get('decision')}**.", "",
                f"*Reason:* {cda.get('decision_reason')}", "",
                "### Policy and standards breached", "",
            ]
            lines += _rule_table(cda.get("breached_rules", []))
            drivers = [f for f in cda.get("top_shap_features", []) if "error" not in f]
            if drivers:
                lines += ["### What drove the score", "",
                          "| Feature | Effect | Contribution |", "|---|---|---|"]
                lines += [f"| {d['feature']} | {d['direction']} | {d['shap_value']} |"
                          for d in drivers]
                lines.append("")
            reasons = cda.get("adverse_action_reasons") or []
            if reasons:
                lines += ["### Adverse-action reasons (Regulation B, maximum four)", ""]
                lines += [f"{i}. {r}" for i, r in enumerate(reasons, 1)]
                lines.append("")

    # --- financial crime ----------------------------------------------------
    if aml:
        lines += ["## Financial Crime Screening", ""]
        if "error" in aml:
            lines += [f"Screening failed: {aml['error']}", ""]
        else:
            txn = aml.get("transaction", {})
            lines += [
                f"Transaction `{aml.get('transaction_id')}` — "
                f"${txn.get('amount_received', 0):,.2f} by {txn.get('payment_format')} "
                f"on {txn.get('timestamp')}.", "",
                f"Model score **{aml.get('probability')}** against a "
                f"{aml.get('model_threshold')} cutoff: {aml.get('verdict')}.", "",
                f"*Recommended action:* **{aml.get('recommended_action')}** — "
                f"{aml.get('action_reason')}", "",
                "### Obligations and red flags triggered", "",
            ]
            lines += _rule_table(aml.get("breached_rules", []))

            roll = aml.get("customer_rollup") or {}
            if roll:
                lines += [
                    "### Customer activity roll-up", "",
                    f"Over the {roll.get('window_days')} days to the transaction under review, "
                    f"{roll.get('flagged', 0)} of {roll.get('transactions', 0)} transactions "
                    f"were flagged, totalling ${roll.get('flagged_value', 0):,.2f}.", "",
                    f"The SAR aggregation test (31 CFR 1020.320, $5,000) is "
                    f"**{'met' if roll.get('meets_sar_aggregate') else 'not met'}**.", "",
                ]
            refs = aml.get("case_references") or []
            if refs:
                lines += ["### Published guidance describing this pattern", ""]
                lines += [f"- [{r['publication']}]({r['url']})" for r in refs]
                lines.append("")

    # --- customer standing --------------------------------------------------
    if kyc and "error" not in kyc:
        rel = kyc.get("relationship", {})
        lines += ["## Customer Standing", "",
                  f"Status: **{kyc.get('customer_status')}**. {kyc.get('notes', '')}", ""]
        if rel.get("first_seen"):
            lines.append(f"On file since {rel['first_seen']}, most recent activity "
                         f"{rel.get('last_seen')}.")
            lines.append("")

    # --- caveats ------------------------------------------------------------
    caveats = []
    if kyc and not kyc.get("screening_performed", True):
        caveats.append(kyc.get("screening_note", "Sanctions screening not performed."))
    for source in (cda, aml):
        if source and source.get("rules_not_evaluated"):
            missing = ", ".join(r["rule_id"] for r in source["rules_not_evaluated"])
            caveats.append(f"{source['agent']}: rules not evaluated for missing data — {missing}.")
    if aml and aml.get("model_available") is False:
        caveats.append(f"AML model unavailable ({aml.get('model_error')}); "
                       f"the assessment above is rules-only.")
    if caveats:
        lines += ["## Data Quality Caveats", ""] + [f"- {c}" for c in caveats] + [""]

    # --- next steps ---------------------------------------------------------
    steps = []
    if kyc and kyc.get("onboarding_required"):
        steps.append("Complete customer due diligence and identity verification before "
                     "permitting account activity.")
    if cda and cda.get("decision") == "Review":
        steps.append("Route the credit application to a human underwriter — the model and "
                     "the policy rules disagree.")
    if cda and cda.get("decision") == "Decline":
        steps.append("Issue an adverse-action notice citing the principal reasons listed above "
                     "within 30 days (Regulation B, 12 CFR 1002.9).")
    if aml and str(aml.get("recommended_action", "")).startswith("File SAR"):
        steps.append("Prepare a SAR for review, citing the breached obligations and the "
                     "roll-up figures above.")
    elif aml and "Escalate" in str(aml.get("recommended_action", "")):
        steps.append("Escalate the transaction to an AML analyst for manual review.")
    if not steps:
        steps.append("No action required. Retain this assessment for the audit record.")
    lines += ["## Recommended Next Steps", ""] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]

    return "\n".join(lines)


def _to_plain_text(markdown: str) -> str:
    """
    Light Markdown -> text conversion for the .txt copy.

    Deliberately minimal. Pipe tables stay as they are because they remain
    perfectly readable as plain text, and rewriting them into aligned columns
    risks mangling content for no real gain. Only heading markers and inline
    emphasis are stripped, since those are pure noise without a renderer.
    """
    out = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            out += ["", text, "-" * len(text)]
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"\1", line))
    return "\n".join(out).strip() + "\n"


def write_report(report: str, stem: str, banner: str = "",
                 subdir: str | None = None) -> tuple[Path, Path]:
    """
    Persist one report as Markdown and plain text.

    Kept out of the agent node itself: a graph node that writes files has a
    side effect that fires on every run including replays from a checkpoint,
    which is not what you want from a resumable graph. The caller decides when
    a report is worth keeping.

    `subdir` groups a batch of reports under one folder. With twenty scenarios
    per run, a flat directory becomes unnavigable after two runs.
    """
    target = REPORT_DIR / subdir if subdir else REPORT_DIR
    target.mkdir(parents=True, exist_ok=True)
    body = f"{banner}\n\n{report}" if banner else report

    md_path = target / f"{stem}.md"
    txt_path = target / f"{stem}.txt"
    md_path.write_text(body, encoding="utf-8")
    txt_path.write_text(_to_plain_text(body), encoding="utf-8")
    return md_path, txt_path


def explanation_agent(state: GraphState):
    """
    Synthesize every worker output into one report written for a human
    reviewer, and record which path produced it.
    """
    if not state.completed_agents:
        report = "# Compliance Assessment Report\n\nNo agent produced a result for this run."
        return {"final_report": report,
                "audit_log": [{"agent": "Explanation", "event": "report_empty"}]}

    header = _verdict_table(state.completed_agents)
    findings = json.dumps(state.completed_agents, indent=2, default=str)
    method = "llm"

    try:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set")

        from langchain_groq import ChatGroq

        # temperature=0.2 — enough freedom for readable prose, little enough
        # that two runs over the same findings tell the same story. An audit
        # trail where the narrative shifts run to run is not much of an audit.
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
        response = llm.invoke([
            ("system", REPORT_INSTRUCTIONS),
            ("user", f"Findings from this assessment run:\n\n{findings}"),
        ])
        narrative = response.content.strip()
        if not narrative:
            raise RuntimeError("empty response from model")
        report = f"{header}\n{narrative}"

    except Exception as exc:
        report = _deterministic_report(state)
        report += f"\n\n---\n*Fallback report — LLM synthesis unavailable: {exc}*"
        method = f"fallback ({exc})"

    log_entry = {
        "agent": "Explanation",
        "event": "report_generated",
        "method": method,
        "agents_summarized": [a.get("agent") for a in state.completed_agents],
    }
    return {"final_report": report, "audit_log": [log_entry]}
