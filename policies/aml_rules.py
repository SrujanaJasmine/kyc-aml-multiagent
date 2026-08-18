"""
aml_rules.py
============
Deterministic BSA/AML rules, each tied to a published US source. Given a transaction
and its computed features, returns which reporting obligations and published red
flags it triggers. Full citations: policies/aml_references.md
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Verified sources. Every URL here was checked, not recalled.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, str]] = {
    "CTR_1010_311": {
        "name": "31 CFR § 1010.311 — Filing obligations for reports of transactions in currency",
        "issuer": "FinCEN",
        "url": "https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1010/subpart-C/section-1010.311",
        "note": ("Requires a CTR for currency transactions of more than $10,000 by, "
                 "through, or to the bank. Multiple currency transactions must be "
                 "treated as one where the institution knows they are by or on behalf "
                 "of the same person and total more than $10,000 in one business day."),
    },
    "SAR_1020_320": {
        "name": "31 CFR § 1020.320 — Reports by banks of suspicious transactions",
        "issuer": "FinCEN",
        "url": "https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1020/subpart-C/section-1020.320",
        "note": ("Banks must file a SAR for transactions aggregating $5,000 or more "
                 "where the bank knows, suspects, or has reason to suspect the "
                 "transaction involves illicit funds, is designed to evade BSA "
                 "requirements, or has no apparent lawful purpose."),
    },
    "STRUCTURING_5324": {
        "name": "31 U.S.C. § 5324 — Structuring transactions to evade reporting requirements",
        "issuer": "United States Code",
        "url": "https://uscode.house.gov/view.xhtml?req=granuleid%3AUSC-1999-title31-section5324&num=0&edition=1999",
        "note": ("Makes it unlawful to structure, assist in structuring, or attempt to "
                 "structure transactions to evade BSA reporting. Criminal even where "
                 "the funds are from a lawful source. Aggravated penalties apply where "
                 "the pattern exceeds $100,000 in twelve months."),
    },
    "FFIEC_APP_G": {
        "name": "FFIEC BSA/AML Examination Manual, Appendix G — Structuring",
        "issuer": "FFIEC",
        "url": "https://bsaaml.ffiec.gov/manual/Appendices/08",
        "note": "Examiner guidance on identifying and evidencing structuring patterns.",
    },
    "FFIEC_APP_F": {
        "name": "FFIEC BSA/AML Examination Manual, Appendix F — Money Laundering and Terrorist Financing Red Flags",
        "issuer": "FFIEC",
        "url": "https://bsaaml.ffiec.gov/manual/Appendices/07",
        "note": ("Published red-flag catalogue. Explicitly not exhaustive, and no single "
                 "indicator is conclusive on its own."),
    },
    "FIN_2014_A005": {
        "name": "FinCEN Advisory FIN-2014-A005 — Funnel Accounts and Trade-Based Money Laundering",
        "issuer": "FinCEN (28 May 2014)",
        "url": "https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a005",
        "note": ("Defines a funnel account as one receiving multiple cash deposits, often "
                 "below the reporting threshold, from which funds are withdrawn in a "
                 "different geographic area with little time elapsing between deposit "
                 "and withdrawal."),
    },
    "FIN_2020_A003": {
        "name": "FinCEN Advisory FIN-2020-A003 — Imposter Scams and Money Mule Schemes",
        "issuer": "FinCEN (7 July 2020)",
        "url": "https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2020-a003",
        "note": ("Red-flag indicators for money mule activity, where an account receives "
                 "funds from multiple unrelated parties and rapidly forwards them on."),
    },
    "FATF_40": {
        "name": "FATF 40 Recommendations",
        "issuer": "Financial Action Task Force",
        "url": "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html",
        "note": ("International AML/CFT standards. Influential on US supervisory "
                 "expectations but not directly enforceable in the United States."),
    },
}

CTR_THRESHOLD = 10_000.0
SAR_THRESHOLD = 5_000.0
CURRENCY_FORMATS = {"Cash"}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rules. Each `check` receives (transaction, features) and returns
# (breached, observed) or (None, None) when the inputs are unavailable.
# ---------------------------------------------------------------------------
def _ctr_check(txn: dict, f: dict):
    """CTR applies to CURRENCY transactions specifically, not to every large
    payment. Applying it to a $12,000 ACH transfer would be wrong -- an ACH is
    not a currency transaction and triggers no CTR obligation. Getting this
    right matters more than catching one extra row."""
    amount = _num(txn.get("amount_received") or txn.get("amount"))
    fmt = txn.get("payment_format")
    if amount is None or fmt is None:
        return (None, None)
    if fmt not in CURRENCY_FORMATS:
        return (False, f"{fmt} (not a currency transaction)")
    return (amount > CTR_THRESHOLD, amount)


def _structuring_check(txn: dict, f: dict):
    """The statutory pattern: several sub-threshold deposits into one account
    that together exceed what a single reportable transaction would have been."""
    count = _num(f.get("struct_band_cnt_7d"))
    total = _num(f.get("struct_band_sum_7d"))
    if count is None or total is None:
        return (None, None)
    breached = count >= 3 and total >= CTR_THRESHOLD
    return (breached, f"{int(count)} deposits totalling ${total:,.2f} in 7 days")


def _sar_aggregate_check(txn: dict, f: dict):
    total = _num(f.get("struct_band_sum_7d"))
    amount = _num(txn.get("amount_received") or txn.get("amount"))
    if total is None and amount is None:
        return (None, None)
    aggregate = max(total or 0.0, amount or 0.0)
    return (aggregate >= SAR_THRESHOLD, f"${aggregate:,.2f}")


def _funnel_check(txn: dict, f: dict):
    """FIN-2014-A005: deposits in, withdrawals out, little time between. A high
    pass-through ratio on the receiving account is that shape."""
    ratio = _num(f.get("hop_dst_pass_ratio"))
    band_count = _num(f.get("struct_band_cnt_7d"))
    if ratio is None:
        return (None, None)
    breached = ratio >= 0.8 and (band_count or 0) >= 2
    return (breached, f"pass-through ratio {ratio:.2f}")


def _mule_fan_in_check(txn: dict, f: dict):
    """FIN-2020-A003: an account collecting from many unrelated senders."""
    in_unique = _num(f.get("deg_in_unique"))
    in_cnt_7d = _num(f.get("vel_in_cnt_7d"))
    if in_unique is None:
        return (None, None)
    breached = in_unique >= 8 and (in_cnt_7d or 0) >= 6
    return (breached, f"{int(in_unique)} distinct senders, {int(in_cnt_7d or 0)} receipts in 7 days")


def _rapid_movement_check(txn: dict, f: dict):
    """FFIEC Appendix F: funds moving straight through with no economic purpose."""
    gap = _num(f.get("vel_out_secs_since_prev"))
    ratio = _num(f.get("hop_src_pass_ratio"))
    if gap is None or ratio is None:
        return (None, None)
    breached = 0 <= gap <= 86_400 and ratio >= 0.7
    return (breached, f"{gap/3600:.1f}h since previous, pass-through {ratio:.2f}")


def _cycle_check(txn: dict, f: dict):
    """FATF layering: value returning to its origin serves no commercial purpose."""
    in_cycle = _num(f.get("hop_in_3cycle"))
    recip = _num(f.get("hop_reciprocal_edge"))
    if in_cycle is None and recip is None:
        return (None, None)
    breached = bool(in_cycle) or bool(recip)
    kind = "3-hop cycle" if in_cycle else ("reciprocal pair" if recip else "none")
    return (breached, kind)


def _velocity_spike_check(txn: dict, f: dict):
    """FFIEC Appendix F: activity inconsistent with the customer's own profile."""
    ratio = _num(f.get("vel_amt_vs_out_mean"))
    if ratio is None:
        return (None, None)
    return (ratio >= 10.0, f"{ratio:.1f}x the account's 30-day average")


RULES: list[dict[str, Any]] = [
    {"id": "CTR-10K", "title": "Currency transaction exceeding $10,000",
     "source": "CTR_1010_311", "authority": "regulation", "severity": "High",
     "threshold": "$10,000 in currency",
     "rationale": ("A currency transaction above $10,000 creates a CTR filing "
                   "obligation. This is a reporting duty, not by itself an "
                   "indication of laundering."),
     "check": _ctr_check},

    {"id": "STRUCTURING", "title": "Deposits structured below the reporting threshold",
     "source": "STRUCTURING_5324", "authority": "regulation", "severity": "High",
     "threshold": "3+ deposits under $10,000 aggregating over $10,000 in 7 days",
     "rationale": ("Breaking a reportable sum into sub-threshold deposits is a "
                   "criminal offence in itself under 31 USC 5324, regardless of "
                   "whether the underlying funds are lawful."),
     "check": _structuring_check},

    {"id": "SAR-5K", "title": "Activity aggregating $5,000 or more",
     "source": "SAR_1020_320", "authority": "regulation", "severity": "Medium",
     "threshold": "$5,000 aggregate",
     "rationale": ("Crossing $5,000 brings the activity within the SAR reporting "
                   "range. It is a necessary condition for filing, not a "
                   "sufficient one -- suspicion must also be present."),
     "check": _sar_aggregate_check},

    {"id": "FUNNEL-ACCOUNT", "title": "Funnel account behaviour",
     "source": "FIN_2014_A005", "authority": "guidance", "severity": "High",
     "threshold": "pass-through ratio >= 0.80 with 2+ sub-threshold deposits",
     "rationale": ("Deposits in and withdrawals out with little time between and "
                   "little value retained is the funnel-account pattern FinCEN "
                   "described in the context of trade-based laundering."),
     "check": _funnel_check},

    {"id": "MULE-FAN-IN", "title": "Collection from multiple unrelated senders",
     "source": "FIN_2020_A003", "authority": "guidance", "severity": "High",
     "threshold": "8+ distinct senders with 6+ receipts in 7 days",
     "rationale": ("An account aggregating funds from many unrelated parties in a "
                   "short window matches FinCEN's published money-mule indicators."),
     "check": _mule_fan_in_check},

    {"id": "RAPID-PASSTHROUGH", "title": "Funds forwarded within 24 hours",
     "source": "FFIEC_APP_F", "authority": "guidance", "severity": "Medium",
     "threshold": "outbound within 24h retaining under 30% of value",
     "rationale": ("Value arriving and leaving almost immediately, with little "
                   "retained, indicates the account is a conduit rather than a "
                   "destination."),
     "check": _rapid_movement_check},

    {"id": "LAYERING-CYCLE", "title": "Funds returning to their origin",
     "source": "FATF_40", "authority": "guidance", "severity": "Medium",
     "threshold": "membership in a 2- or 3-hop cycle",
     "rationale": ("Circular flows serve no commercial purpose and are a classic "
                   "layering technique intended to obscure origin."),
     "check": _cycle_check},

    {"id": "VELOCITY-SPIKE", "title": "Transaction far outside the account's own norm",
     "source": "FFIEC_APP_F", "authority": "guidance", "severity": "Low",
     "threshold": "10x the account's 30-day average",
     "rationale": ("Activity inconsistent with the customer's established profile "
                   "is the general red flag underlying most monitoring rules."),
     "check": _velocity_spike_check},
]

SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1}

POLICY_TEXT: dict[str, str] = {
    r["id"]: (f"{r['title']} — {SOURCES[r['source']]['name']} "
              f"({SOURCES[r['source']]['issuer']}). Threshold: {r['threshold']}. "
              f"{r['rationale']} Source note: {SOURCES[r['source']]['note']} "
              f"Reference: {SOURCES[r['source']]['url']}")
    for r in RULES
}


def evaluate_aml_rules(transaction: dict, features: dict) -> dict[str, Any]:
    """
    Run every rule against one transaction plus its computed features.

    `not_evaluated` is kept distinct from `passed` because a missing feature is
    not evidence of compliance, and an incomplete record should not read as clean.
    """
    breached, passed, not_evaluated = [], [], []

    for rule in RULES:
        try:
            is_breach, observed = rule["check"](transaction, features)
        except Exception as exc:
            not_evaluated.append({"rule_id": rule["id"], "reason": f"check error: {exc}"})
            continue

        if is_breach is None:
            not_evaluated.append({"rule_id": rule["id"], "reason": "required input missing"})
        elif is_breach:
            src = SOURCES[rule["source"]]
            breached.append({
                "rule_id": rule["id"], "title": rule["title"],
                "severity": rule["severity"], "authority": rule["authority"],
                "threshold": rule["threshold"], "observed": observed,
                "rationale": rule["rationale"],
                "source_name": src["name"], "source_issuer": src["issuer"],
                "source_url": src["url"], "source_note": src["note"],
            })
        else:
            passed.append({"rule_id": rule["id"], "title": rule["title"], "observed": observed})

    breached.sort(key=lambda r: SEVERITY_RANK.get(r["severity"], 0), reverse=True)

    return {
        "breached": breached, "passed": passed, "not_evaluated": not_evaluated,
        "breach_count": len(breached),
        "high_severity_count": sum(1 for r in breached if r["severity"] == "High"),
        "regulatory_breach_count": sum(1 for r in breached if r["authority"] == "regulation"),
    }


def case_references(breached: list[dict]) -> list[dict]:
    """
    Published advisories describing the schemes these breaches resemble.

    Scoped to rules already found breached, so a case reference can explain a
    finding but never manufacture one.
    """
    seen, refs = set(), []
    for rule in breached:
        key = rule["rule_id"]
        src = SOURCES[next(r["source"] for r in RULES if r["id"] == key)]
        if src["name"] in seen:
            continue
        seen.add(src["name"])
        refs.append({"triggered_by": key, "publication": src["name"],
                     "issuer": src["issuer"], "url": src["url"], "summary": src["note"]})
    return refs


def recommended_action(rule_result: dict, probability: float, threshold: float) -> tuple[str, str]:
    """
    Translate findings into what a compliance team would actually do.

    Deliberately conservative about the word "SAR": a filing decision is a human
    judgement with legal consequences, so the agent recommends consideration and
    supplies the evidence rather than asserting an obligation.
    """
    reg = rule_result["regulatory_breach_count"]
    high = rule_result["high_severity_count"]
    model_flag = probability >= threshold

    if reg and model_flag:
        return ("File SAR — recommend review",
                f"{reg} regulatory threshold(s) crossed and the model scores "
                f"{probability:.1%} against a {threshold:.0%} cutoff.")
    if reg:
        return ("Escalate for analyst review",
                f"{reg} regulatory threshold(s) crossed although the model score "
                f"({probability:.1%}) is below the cutoff.")
    if model_flag and high:
        return ("Escalate for analyst review",
                f"Model scores {probability:.1%} with {high} high-severity red flag(s).")
    if model_flag:
        return ("Enhanced monitoring",
                f"Model scores {probability:.1%} but no published red flag was triggered.")
    if high:
        return ("Enhanced monitoring",
                f"{high} high-severity red flag(s) despite a model score of {probability:.1%}.")
    return ("No action", "Neither the model nor any policy rule flagged this activity.")


if __name__ == "__main__":
    print(f"{len(RULES)} AML rules across {len(SOURCES)} sources\n")
    for r in RULES:
        src = SOURCES[r["source"]]
        print(f"  [{r['severity']:<6}] [{r['authority']:<10}] {r['id']:<18} {r['title']}")
        print(f"           limit: {r['threshold']}")
        print(f"           {src['name']}\n")
