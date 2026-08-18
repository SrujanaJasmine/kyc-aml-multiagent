"""
credit_rules.py
===============
Deterministic credit-policy rules, each tied to a published US source. Given an
application, returns which rules it breaches with the observed value against the
published threshold. Full citations: policies/references.md
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Sources. Keyed so each rule can point at one without repeating the URL.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, str]] = {
    "ECOA_1002_9": {
        "name": "Equal Credit Opportunity Act, Regulation B — 12 CFR § 1002.9",
        "issuer": "Consumer Financial Protection Bureau",
        "url": "https://www.consumerfinance.gov/rules-policy/regulations/1002/9/",
        "note": (
            "Requires notice of action taken within 30 days and a statement of "
            "the specific principal reasons for adverse action. Reasons must "
            "describe the factors actually considered; citing internal policy "
            "or a failed credit score alone is insufficient. Disclosing more "
            "than four reasons is discouraged as unhelpful."
        ),
    },
    "FFIEC_URCC": {
        "name": "Uniform Retail Credit Classification and Account Management Policy",
        "issuer": "FFIEC (FRB, FDIC, OCC, OTS)",
        "url": "https://www.federalregister.gov/documents/2000/06/12/00-14704/uniform-retail-credit-classification-and-account-management-policy",
        "note": (
            "Retail loans 90 cumulative days past due are classified "
            "substandard. Closed-end loans charge off at 120 days past due; "
            "open-end credit at 180 days."
        ),
    },
    "FNMA_B3_6_02": {
        "name": "Fannie Mae Selling Guide B3-6-02, Debt-to-Income Ratios",
        "issuer": "Fannie Mae",
        "url": "https://selling-guide.fanniemae.com/sel/b3-6-02/debt-income-ratios",
        "note": (
            "Maximum total DTI of 36% for manually underwritten loans, "
            "extendable to 45% where credit score and reserve requirements in "
            "the Eligibility Matrix are met; 50% for loan casefiles underwritten "
            "through Desktop Underwriter."
        ),
    },
    "CFPB_SCORE": {
        "name": "CFPB — How do I get and keep a good credit score?",
        "issuer": "Consumer Financial Protection Bureau",
        "url": "https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-and-keep-a-good-credit-score-en-318/",
        "note": (
            "Advises keeping credit use at no more than 30% of the total credit "
            "limit. This is consumer guidance, not a regulatory threshold — FICO "
            "scores utilization on a sliding scale with no cliff at 30%."
        ),
    },
    "FICO_AMOUNTS_OWED": {
        "name": "FICO — Amounts Owed score factor",
        "issuer": "Fair Isaac Corporation",
        "url": "https://www.myfico.com/credit-education/blog/credit-score-factor-amounts-owed",
        "note": (
            "Amounts owed is 30% of a FICO Score, the second-largest component. "
            "The highest-scoring consumers typically carry utilization below 10%."
        ),
    },
    "TILA_1026_7": {
        "name": "Truth in Lending Act, Regulation Z — 12 CFR § 1026.7(b)(12)",
        "issuer": "Consumer Financial Protection Bureau",
        "url": "https://www.consumerfinance.gov/rules-policy/regulations/1026/7/",
        "note": (
            "Requires the periodic-statement minimum-payment warning: the effect "
            "of making only minimum payments on repayment time and total cost."
        ),
    },
    "REG_Z_ATR": {
        "name": "Ability-to-Repay / Qualified Mortgage — 12 CFR § 1026.43",
        "issuer": "Consumer Financial Protection Bureau",
        "url": "https://www.federalregister.gov/documents/2020/12/29/2020-27567/qualified-mortgage-definition-under-the-truth-in-lending-act-regulation-z-general-qm-loan-definition",
        "note": (
            "Creditors must make a reasonable, good-faith determination of "
            "ability to repay. NOTE: the General QM 43% DTI limit was REMOVED by "
            "the December 2020 final rule and replaced with a price-based "
            "threshold (APR vs APOR). Do not cite 43% DTI as a current QM limit."
        ),
    },
}


def _num(value: Any) -> float | None:
    """Coerce to float; None for NULL/blank/garbage so a missing field reports as
    'not evaluated' rather than silently passing."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _dti_check(a: dict, limit_pct: float):
    """DTI needs two fields, so it is a named function. Guards a zero or missing
    salary instead of dividing by it."""
    emi = _num(a.get("Total_EMI_per_month"))
    salary = _num(a.get("customer_monthly_inhand_salary"))
    if emi is None or salary is None or salary <= 0:
        return (None, None)
    ratio = (emi / salary) * 100.0
    return (ratio > limit_pct, round(ratio, 2))


# ---------------------------------------------------------------------------
# Rules. `check` receives the normalised application dict (model-style keys
# for application fields, customer_-prefixed for the joined customer fields)
# and returns (breached, observed) or (None, None) when data is missing.
# ---------------------------------------------------------------------------
RULES: list[dict[str, Any]] = [
    {
        "id": "DELINQ-90-SUBSTANDARD",
        "title": "Account 90+ days past due — classified substandard",
        "source": "FFIEC_URCC",
        "authority": "regulation",
        "severity": "High",
        "threshold": "90 days",
        "rationale": (
            "At 90 cumulative days past due the FFIEC policy requires the "
            "exposure to be classified substandard, and charge-off follows at "
            "120 days closed-end or 180 days open-end. This is a supervisory "
            "classification trigger, not a matter of lender discretion."
        ),
        "check": lambda a: (
            (v >= 90.0, v) if (v := _num(a.get("Delay_from_due_date"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "DELINQ-30-REPORTABLE",
        "title": "Payment more than 30 days past due",
        "source": "FFIEC_URCC",
        "authority": "regulation",
        "severity": "High",
        "threshold": "30 days",
        "rationale": (
            "30 days is the first delinquency bucket in standard bureau "
            "reporting — the point at which a lapse becomes reportable "
            "delinquency rather than an administrative delay."
        ),
        "check": lambda a: (
            (v > 30.0, v) if (v := _num(a.get("Delay_from_due_date"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "DTI-36-MANUAL",
        "title": "Debt service above 36% of monthly income",
        "source": "FNMA_B3_6_02",
        "authority": "underwriting",
        "severity": "High",
        "threshold": "36%",
        "rationale": (
            "Fannie Mae's base maximum total DTI for manually underwritten "
            "loans. Exceeding it does not automatically disqualify — the Guide "
            "permits up to 45% where credit score and reserves justify it — but "
            "it moves the file out of standard eligibility."
        ),
        "check": lambda a: _dti_check(a, 36.0),
    },
    {
        "id": "DTI-45-CEILING",
        "title": "Debt service above 45% of monthly income",
        "source": "FNMA_B3_6_02",
        "authority": "underwriting",
        "severity": "High",
        "threshold": "45%",
        "rationale": (
            "The upper bound for manually underwritten loans even with "
            "compensating factors. Above this, the file is outside standard "
            "agency eligibility regardless of credit strength."
        ),
        "check": lambda a: _dti_check(a, 45.0),
    },
    {
        "id": "UTIL-30-GUIDANCE",
        "title": "Revolving utilization above 30%",
        "source": "CFPB_SCORE",
        "authority": "guidance",
        "severity": "Medium",
        "threshold": "30%",
        "rationale": (
            "CFPB advises keeping usage at or below 30% of the total limit, and "
            "amounts owed is 30% of a FICO Score. Treat this as a soft signal: "
            "utilization is scored on a sliding scale, so 31% is not "
            "meaningfully different from 29% — the flag matters when combined "
            "with delinquency, not on its own."
        ),
        "check": lambda a: (
            (v > 30.0, v) if (v := _num(a.get("Credit_Utilization_Ratio"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "UTIL-70-SEVERE",
        "title": "Revolving utilization above 70%",
        "source": "FICO_AMOUNTS_OWED",
        "authority": "guidance",
        "severity": "High",
        "threshold": "70%",
        "rationale": (
            "Sustained utilization this high indicates revolving credit is "
            "funding ongoing consumption rather than being repaid. Unlike the "
            "30% advisory figure, this level is far outside the range typical of "
            "prime borrowers and is a substantive repayment-capacity signal."
        ),
        "check": lambda a: (
            (v > 70.0, v) if (v := _num(a.get("Credit_Utilization_Ratio"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "MINPAY-ONLY",
        "title": "Paying only the minimum amount due",
        "source": "TILA_1026_7",
        "authority": "regulation",
        "severity": "Medium",
        "threshold": "Payment_of_Min_Amount = Yes",
        "rationale": (
            "Congress considered minimum-payment behaviour consequential enough "
            "to mandate a warning on every periodic statement. Servicing only "
            "the minimum barely amortises principal and is a recognised marker "
            "of revolving-debt dependence."
        ),
        "check": lambda a: (
            (v.lower() == "yes", v) if (v := _text(a.get("Payment_of_Min_Amount"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "DELINQ-COUNT",
        "title": "More than 3 delayed payments on file",
        "source": "FFIEC_URCC",
        "authority": "underwriting",
        "severity": "High",
        "threshold": "3 occurrences",
        "rationale": (
            "A single late payment can be an oversight; a repeated pattern is "
            "the strongest behavioural predictor of default and normally routes "
            "to manual review regardless of model score."
        ),
        "check": lambda a: (
            (v > 3.0, v) if (v := _num(a.get("Num_of_Delayed_Payment"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "MIX-BAD",
        "title": "Poor credit mix",
        "source": "FICO_AMOUNTS_OWED",
        "authority": "guidance",
        "severity": "Medium",
        "threshold": "Credit_Mix = Bad",
        "rationale": (
            "A narrow or poorly performing mix of account types limits evidence "
            "that the borrower can manage different credit forms, and lowers "
            "confidence in the score itself."
        ),
        "check": lambda a: (
            (v.lower() == "bad", v) if (v := _text(a.get("Credit_Mix"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "INQUIRY-6",
        "title": "More than 6 hard credit inquiries",
        "source": "FICO_AMOUNTS_OWED",
        "authority": "guidance",
        "severity": "Medium",
        "threshold": "6 inquiries",
        "rationale": (
            "A cluster of hard inquiries suggests credit-seeking from several "
            "lenders at once, which historically precedes both over-extension "
            "and application fraud."
        ),
        "check": lambda a: (
            (v > 6.0, v) if (v := _num(a.get("Num_Credit_Inquiries"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "THIN-FILE",
        "title": "Credit history shorter than 24 months",
        "source": "REG_Z_ATR",
        "authority": "underwriting",
        "severity": "Low",
        "threshold": "24 months",
        "rationale": (
            "Below roughly two years there is insufficient repayment history for "
            "a score to be statistically stable. Under the ability-to-repay "
            "standard this is a confidence caveat requiring verification of "
            "income and obligations, not evidence of poor credit."
        ),
        "check": lambda a: (
            (v < 24.0, v) if (v := _num(a.get("Credit_History_Age_Total"))) is not None
            else (None, None)
        ),
    },
    {
        "id": "LOAN-COUNT",
        "title": "More than 5 active loans",
        "source": "REG_Z_ATR",
        "authority": "underwriting",
        "severity": "Medium",
        "threshold": "5 loans",
        "rationale": (
            "Numerous simultaneous obligations fragment cash flow and make total "
            "exposure harder to verify, which bears directly on the reasonable "
            "good-faith ability-to-repay determination."
        ),
        "check": lambda a: (
            (v > 5.0, v) if (v := _num(a.get("Num_of_Loan"))) is not None
            else (None, None)
        ),
    },
]

SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1}

POLICY_TEXT: dict[str, str] = {
    rule["id"]: (
        f"{rule['title']} — {SOURCES[rule['source']]['name']} "
        f"({SOURCES[rule['source']]['issuer']}). "
        f"Threshold: {rule['threshold']}. {rule['rationale']} "
        f"Source note: {SOURCES[rule['source']]['note']} "
        f"Reference: {SOURCES[rule['source']]['url']}"
    )
    for rule in RULES
}


def evaluate_rules(application: dict) -> dict[str, Any]:
    """
    Run every rule against one normalised application.

    breached / passed / not_evaluated are kept separate on purpose: a NULL
    income is not a compliant income, and collapsing the two would let an
    incomplete record look clean.
    """
    breached, passed, not_evaluated = [], [], []

    for rule in RULES:
        try:
            is_breach, observed = rule["check"](application)
        except Exception as exc:
            not_evaluated.append({"rule_id": rule["id"], "reason": f"check error: {exc}"})
            continue

        if is_breach is None:
            not_evaluated.append({"rule_id": rule["id"], "reason": "required field missing"})
        elif is_breach:
            source = SOURCES[rule["source"]]
            breached.append({
                "rule_id": rule["id"],
                "title": rule["title"],
                "severity": rule["severity"],
                "authority": rule["authority"],
                "threshold": rule["threshold"],
                "observed": observed,
                "rationale": rule["rationale"],
                "source_name": source["name"],
                "source_issuer": source["issuer"],
                "source_url": source["url"],
            })
        else:
            passed.append({"rule_id": rule["id"], "title": rule["title"], "observed": observed})

    breached.sort(key=lambda r: SEVERITY_RANK.get(r["severity"], 0), reverse=True)

    return {
        "breached": breached,
        "passed": passed,
        "not_evaluated": not_evaluated,
        "breach_count": len(breached),
        "high_severity_count": sum(1 for r in breached if r["severity"] == "High"),
        "regulatory_breach_count": sum(1 for r in breached if r["authority"] == "regulation"),
    }


def adverse_action_reasons(breached: list[dict], limit: int = 4) -> list[str]:
    """
    The top reasons for an adverse action notice.

    Capped at four because Regulation B's official interpretation states that
    disclosing more than four reasons is unlikely to be helpful to the
    applicant. Ordered by severity so the principal reasons come first.
    """
    return [f"{r['title']} (observed: {r['observed']}, limit: {r['threshold']})"
            for r in breached[:limit]]


if __name__ == "__main__":
    print(f"{len(RULES)} rules across {len(SOURCES)} sources\n")
    for r in RULES:
        src = SOURCES[r["source"]]
        print(f"  [{r['severity']:<6}] [{r['authority']:<12}] {r['id']:<24} {r['title']}")
        print(f"           limit {r['threshold']:<12} {src['name']}")
