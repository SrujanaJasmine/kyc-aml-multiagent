"""
feature_mapping.py
==================
Maps the six customer-table fields the credit model needs onto its feature names.
Columns from `credit_applications` already use the model's own names and pass
straight through untranslated.
"""

from __future__ import annotations

import pandas as pd

# `credit_applications` now uses the model's own feature names, so those columns
# need no translation at all -- they pass straight through. What remains is the
# six CUSTOMER-level features the model wants, which live in the `customers`
# table and arrive from get_credit_application's JOIN with a `customer_` prefix.
#
# `customers` deliberately keeps snake_case: it is read by the KYC and AML
# agents, which have no feature contract to satisfy, and renaming it would push
# churn into two agents to save six dictionary entries here.
COLUMN_MAP: dict[str, str] = {
    "customer_age": "Age",
    "customer_annual_income": "Annual_Income",
    "customer_monthly_inhand_salary": "Monthly_Inhand_Salary",
    "customer_num_bank_accounts": "Num_Bank_Accounts",
    "customer_num_credit_card": "Num_Credit_Card",
    "customer_occupation": "Occupation",
}

# Never forwarded. Credit_Score is the training label -- the reindex in
# build_features would drop it anyway, but dropping it here too means the leak
# cannot survive a future refactor of the scorer.
NON_FEATURE_COLUMNS = {
    "application_id",
    "customer_id",
    "customer_name",
    "status",
    "Credit_Score",
}

# Names a caller might plausibly use that match neither convention.
ALIASES: dict[str, str] = {
    "age": "Age",
    "annual_income": "Annual_Income",
    "monthly_inhand_salary": "Monthly_Inhand_Salary",
    "num_bank_accounts": "Num_Bank_Accounts",
    "num_credit_card": "Num_Credit_Card",
    "occupation": "Occupation",
    "credit_score": "Credit_Score",
}

REVERSE_COLUMN_MAP: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}


def normalize_application_payload(payload: dict) -> dict:
    """
    Accept an application in either naming convention and return the model's.

    Since the database now speaks the model's vocabulary, most keys pass through
    untouched; only the customer-level fields and a few plausible variants get
    renamed. Unrecognised keys are preserved rather than dropped, so extra
    context a caller attaches survives into the audit log even though the model
    ignores it.
    """
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if key in COLUMN_MAP:
            normalized[COLUMN_MAP[key]] = value
        elif key in ALIASES:
            normalized[ALIASES[key]] = value
        else:
            normalized[key] = value
    return normalized


def application_row_to_features(application: dict) -> pd.DataFrame:
    """
    Convert one `get_credit_application` row into a single-row DataFrame using
    the model's column names.

    Unmapped keys are KEPT rather than dropped, because the database columns
    already are the model's names. Only the customer_-prefixed fields are renamed
    and the non-feature columns removed. Anything genuinely unexpected is
    discarded later by the scorer's reindex onto the saved feature contract.
    """
    mapped = {COLUMN_MAP.get(k, k): v
              for k, v in application.items()
              if k not in NON_FEATURE_COLUMNS}
    return pd.DataFrame([mapped])


def mapping_coverage(feature_columns: list[str]) -> dict[str, list[str]]:
    """
    Diagnostic: which of the model's expected features can be populated.

    Three sources now count as covered -- renamed customer fields, one-hot
    expansions produced by get_dummies, and application columns that pass
    through because the schema uses the model's own names.
    """
    from_customers = set(COLUMN_MAP.values())
    one_hot_prefixes = ("Month_", "Occupation_", "Credit_Mix_",
                        "Payment_of_Min_Amount_", "Payment_Behaviour_")
    application_columns = {
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Age_Total",
        "Auto_Loan", "Credit_Builder_Loan", "Debt_Consolidation_Loan",
        "Home_Equity_Loan", "Mortgage_Loan", "Not_Specified", "Payday_Loan",
        "Personal_Loan", "Student_Loan", "Unknown_Loan",
    }

    mapped, unmapped = [], []
    for col in feature_columns:
        if (col in from_customers or col in application_columns
                or col.startswith(one_hot_prefixes)):
            mapped.append(col)
        else:
            unmapped.append(col)
    return {"mapped": mapped, "unmapped": unmapped}
