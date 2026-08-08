"""
Shared customer database for the Multi-Agent RAG system.

This is deliberately a SEPARATE database from `system_memory.db`:
  - system_memory.db  -> HOW the system ran (checkpoints, agent_history / audit trail)
  - customer_data.db  -> WHAT the system is reasoning about (customers, applications,
                          transactions) — the actual business data every agent looks up.

Schema
------
customers
    One row per Customer_ID, deduped from the credit dataset's monthly
    panel (the CSV repeats these 9 fields every month for the same
    customer — we only want the first occurrence of each). Also receives
    bare stub rows (customer_id only, everything else NULL) for any
    Customer_ID that shows up in the AML transactions CSVs but wasn't in
    the credit dataset — see load_transactions_from_csv.

        customer_id (PK)  name  ssn  age  occupation
        annual_income  monthly_inhand_salary
        num_bank_accounts  num_credit_card

credit_applications
    One row PER ROW of the credit CSV (the CSV's `ID` column is already
    unique per row, so it's used directly as application_id). Each row is
    really a monthly bureau snapshot rather than a discrete loan
    application, so `month` holds the CSV's raw Month string instead of a
    calendar date.

        application_id (PK)  customer_id (FK -> customers)  month
        interest_rate  num_of_loan  delay_from_due_date
        num_of_delayed_payment  changed_credit_limit  num_credit_inquiries
        credit_mix  outstanding_debt  credit_utilization_ratio
        payment_of_min_amount  total_emi_per_month  amount_invested_monthly
        payment_behaviour  monthly_balance  credit_score
        credit_history_age_total
        auto_loan  credit_builder_loan  debt_consolidation_loan
        home_equity_loan  mortgage_loan  not_specified_loan  payday_loan
        personal_loan  student_loan  unknown_loan
        status

transactions
    One row per row of the AML customer_transactions_train.csv /
    customer_transactions_test.csv files, restricted to the 11 columns
    actually needed (Timestamp / From Bank / Account / To Bank / Account.1
    / Amount Received / Receiving Currency / Amount Paid / Payment
    Currency / Payment Format / Is Laundering / Customer_ID). The CSV has
    no transaction ID column, so one is generated as "<PREFIX>_<row index>"
    (PREFIX = TRAIN or TEST by default) — unique per source file, stable
    across re-runs, and INSERT OR IGNORE-safe.

        transaction_id (PK)  customer_id (FK -> customers)  timestamp
        from_bank  from_account  to_bank  to_account
        amount_received  receiving_currency  amount_paid  payment_currency
        payment_format  is_laundering

Each specialist agent (KYC / AML / CDA) queries this database directly instead
of receiving a full payload in `assessment_input` — the orchestrator only needs
to pass an ID (customer_id / application_id / transaction_id), and the worker
looks up whatever fields it needs.
"""

import os
import sqlite3

import pandas as pd

DB_PATH = "customer_data.db"


def get_connection() -> sqlite3.Connection:
    """
    Every caller gets its own connection with row_factory set, so query
    results come back as dict-like sqlite3.Row objects instead of plain
    tuples. Cheap to open in SQLite, so we don't share a single global
    connection across agents (avoids threading issues under LangGraph's
    parallel Send execution).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _clean_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    """
    pandas nullable/NaN values (pd.NA, np.nan) are NOT accepted by
    sqlite3.executemany — it raises "type 'NAType' is not supported".
    This converts every missing value to plain Python None, which sqlite3
    binds correctly as SQL NULL. Call this right before itertuples() on
    any DataFrame headed for an INSERT.
    """
    return df.astype(object).where(df.notna(), None)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all three tables (and indexes) if they don't exist yet."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id             TEXT PRIMARY KEY,
                name                    TEXT,
                ssn                     TEXT,
                age                     INTEGER,
                occupation              TEXT,
                annual_income           REAL,
                monthly_inhand_salary   REAL,
                num_bank_accounts       INTEGER,
                num_credit_card         INTEGER
            );

            CREATE TABLE IF NOT EXISTS credit_applications (
                application_id             TEXT PRIMARY KEY,
                customer_id                 TEXT NOT NULL REFERENCES customers(customer_id),
                month                        TEXT,
                interest_rate                REAL,
                num_of_loan                  REAL,
                delay_from_due_date          REAL,
                num_of_delayed_payment       REAL,
                changed_credit_limit         REAL,
                num_credit_inquiries         REAL,
                credit_mix                   TEXT,
                outstanding_debt             REAL,
                credit_utilization_ratio     REAL,
                payment_of_min_amount        TEXT,
                total_emi_per_month          REAL,
                amount_invested_monthly      REAL,
                payment_behaviour            TEXT,
                monthly_balance              REAL,
                credit_score                 INTEGER,   -- dataset's own label; NOT a model feature, see cda_agent.py
                credit_history_age_total     REAL,
                auto_loan                    REAL DEFAULT 0,
                credit_builder_loan          REAL DEFAULT 0,
                debt_consolidation_loan      REAL DEFAULT 0,
                home_equity_loan             REAL DEFAULT 0,
                mortgage_loan                REAL DEFAULT 0,
                not_specified_loan           REAL DEFAULT 0,
                payday_loan                  REAL DEFAULT 0,
                personal_loan                REAL DEFAULT 0,
                student_loan                 REAL DEFAULT 0,
                unknown_loan                 REAL DEFAULT 0,
                status                       TEXT DEFAULT 'Pending'  -- Pending / Approved / Review / Declined
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id      TEXT PRIMARY KEY,
                customer_id          TEXT NOT NULL REFERENCES customers(customer_id),
                timestamp             TEXT,
                from_bank             TEXT,
                from_account          TEXT,
                to_bank               TEXT,
                to_account            TEXT,
                amount_received       REAL,
                receiving_currency    TEXT,
                amount_paid           REAL,
                payment_currency      TEXT,
                payment_format        TEXT,
                is_laundering         INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_credit_customer ON credit_applications(customer_id);
            CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

# CSV column -> customers table column
_CUSTOMER_COLUMNS = {
    "Customer_ID": "customer_id",
    "Name": "name",
    "SSN": "ssn",
    "Age": "age",
    "Occupation": "occupation",
    "Annual_Income": "annual_income",
    "Monthly_Inhand_Salary": "monthly_inhand_salary",
    "Num_Bank_Accounts": "num_bank_accounts",
    "Num_Credit_Card": "num_credit_card",
}

# CSV column -> credit_applications table column
_APPLICATION_COLUMNS = {
    "ID": "application_id",
    "Customer_ID": "customer_id",
    "Month": "month",
    "Interest_Rate": "interest_rate",
    "Num_of_Loan": "num_of_loan",
    "Delay_from_due_date": "delay_from_due_date",
    "Num_of_Delayed_Payment": "num_of_delayed_payment",
    "Changed_Credit_Limit": "changed_credit_limit",
    "Num_Credit_Inquiries": "num_credit_inquiries",
    "Credit_Mix": "credit_mix",
    "Outstanding_Debt": "outstanding_debt",
    "Credit_Utilization_Ratio": "credit_utilization_ratio",
    "Payment_of_Min_Amount": "payment_of_min_amount",
    "Total_EMI_per_month": "total_emi_per_month",
    "Amount_invested_monthly": "amount_invested_monthly",
    "Payment_Behaviour": "payment_behaviour",
    "Monthly_Balance": "monthly_balance",
    "Credit_Score": "credit_score",
    "Credit_History_Age_Total": "credit_history_age_total",
    "Auto Loan": "auto_loan",
    "Credit-Builder Loan": "credit_builder_loan",
    "Debt Consolidation Loan": "debt_consolidation_loan",
    "Home Equity Loan": "home_equity_loan",
    "Mortgage Loan": "mortgage_loan",
    "Not Specified": "not_specified_loan",
    "Payday Loan": "payday_loan",
    "Personal Loan": "personal_loan",
    "Student Loan": "student_loan",
    "Unknown Loan": "unknown_loan",
}

# The 10 one-hot loan-amount columns — NaN means "no loan of that type
# that month", so it's filled with 0 rather than left null.
_LOAN_TYPE_CSV_COLUMNS = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Not Specified", "Payday Loan",
    "Personal Loan", "Student Loan", "Unknown Loan",
]

# CSV column -> transactions table column. NOTE: the raw AML CSV has TWO
# columns literally named "Account" (sender account, receiver account) —
# pandas auto-renames the second one to "Account.1" on read, which is why
# that's the key used here rather than "Account" twice.
_TRANSACTION_COLUMNS = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
    "Customer_ID": "customer_id",
}


def load_customers_from_csv(csv_path: str) -> None:
    """
    Loads the 9 customer-level columns from the credit dataset. The CSV is
    a monthly panel (one row per customer per month) so these fields
    repeat — dedupe by Customer_ID, keeping the first row seen.
    `INSERT OR IGNORE` on the customer_id PK also protects re-runs.
    """
    df = pd.read_csv(csv_path, usecols=list(_CUSTOMER_COLUMNS.keys()))
    df = df.rename(columns=_CUSTOMER_COLUMNS)
    df = df.drop_duplicates(subset="customer_id", keep="first")

    # Age / Num_Bank_Accounts / Num_Credit_Card come in as float64 in the
    # CSV (e.g. 28.0) — round then cast to nullable Int64 so a missing
    # value doesn't force the whole column to stay float.
    for col in ("age", "num_bank_accounts", "num_credit_card"):
        df[col] = df[col].round().astype("Int64")

    df = _clean_for_sqlite(df)  # pd.NA/np.nan -> None, or sqlite3 raises on insert

    cols = list(_CUSTOMER_COLUMNS.values())
    conn = get_connection()
    try:
        conn.executemany(
            f"""INSERT OR IGNORE INTO customers ({", ".join(cols)})
                VALUES ({", ".join("?" for _ in cols)})""",
            df[cols].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


def load_credit_applications_from_csv(csv_path: str) -> None:
    """
    Loads every row of the credit dataset into credit_applications. Unlike
    customers, no dedup is needed — the CSV's `ID` column is already
    unique per row and becomes application_id directly.

    Run `load_customers_from_csv` on the same file first — customer_id is
    a foreign key here and PRAGMA foreign_keys is ON.

    Not every credit CSV has every column in _APPLICATION_COLUMNS — the
    Kaggle test split, in particular, omits `Credit_Score` since that's
    the label being predicted. Missing columns are filled with NULL
    instead of failing the whole load, which also happens to be the
    semantically right outcome: a test-split row becomes a "pending"
    application with no score yet, i.e. exactly what CDA is meant to
    assess.
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    available_csv_cols = [c for c in _APPLICATION_COLUMNS if c in header]
    missing_csv_cols = [c for c in _APPLICATION_COLUMNS if c not in header]
    if missing_csv_cols:
        print(f"{csv_path}: missing columns {missing_csv_cols} — inserting as NULL")

    df = pd.read_csv(csv_path, usecols=available_csv_cols)
    df = df.rename(columns=_APPLICATION_COLUMNS)
    for csv_col in missing_csv_cols:
        df[_APPLICATION_COLUMNS[csv_col]] = None

    loan_cols = [_APPLICATION_COLUMNS[c] for c in _LOAN_TYPE_CSV_COLUMNS]
    df[loan_cols] = df[loan_cols].fillna(0)

    df = _clean_for_sqlite(df)  # pd.NA/np.nan -> None, or sqlite3 raises on insert

    cols = list(_APPLICATION_COLUMNS.values())
    conn = get_connection()
    try:
        conn.executemany(
            f"""INSERT OR IGNORE INTO credit_applications ({", ".join(cols)})
                VALUES ({", ".join("?" for _ in cols)})""",
            df[cols].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


def load_transactions_from_csv(csv_path: str, id_prefix: str | None = None) -> None:
    """
    Loads the AML transactions CSV (train or test) into `transactions`,
    keeping only the 11 columns actually needed. The CSV has no built-in
    transaction ID, so one is generated as "<PREFIX>_<row index>" —
    PREFIX defaults to the filename stem uppercased, or pass
    id_prefix="TRAIN" / "TEST" for something shorter. Unique per file,
    stable across re-runs.

    Any Customer_ID in this CSV that isn't already in `customers` (e.g.
    the AML dataset's population doesn't fully overlap with the credit
    dataset's) gets a bare stub row inserted first — customer_id only,
    every other field NULL — so the transactions FK never rejects a row.
    Distinguish stubs later with: SELECT * FROM customers WHERE name IS NULL.
    """
    # Not using usecols here: "Account" appears twice in the raw header,
    # and usecols name-matching against a duplicated column is unreliable.
    # Read the full file, let pandas mangle the duplicate to "Account.1",
    # then select by name afterward.
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_TRANSACTION_COLUMNS)[list(_TRANSACTION_COLUMNS.values())]

    prefix = id_prefix or os.path.splitext(os.path.basename(csv_path))[0].upper()
    df.insert(0, "transaction_id", [f"{prefix}_{i}" for i in range(len(df))])

    df["is_laundering"] = df["is_laundering"].round().astype("Int64")

    df = _clean_for_sqlite(df)  # pd.NA/np.nan -> None, or sqlite3 raises on insert

    conn = get_connection()
    try:
        # Stub in any customer_id referenced here but missing from
        # `customers`, so the FK on transactions.customer_id never fails.
        unique_customers = df["customer_id"].dropna().unique().tolist()
        conn.executemany(
            "INSERT OR IGNORE INTO customers (customer_id) VALUES (?)",
            [(cid,) for cid in unique_customers],
        )

        cols = ["transaction_id"] + list(_TRANSACTION_COLUMNS.values())
        conn.executemany(
            f"""INSERT OR IGNORE INTO transactions ({", ".join(cols)})
                VALUES ({", ".join("?" for _ in cols)})""",
            df[cols].itertuples(index=False, name=None),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lookup helpers — what the agents actually call
# ---------------------------------------------------------------------------
def get_customer(customer_id: str) -> dict | None:
    """Used by the KYC agent."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_credit_application(application_id: str) -> dict | None:
    """
    Used by the CDA agent. Joins in the customer's stable fields (income,
    age, occupation, etc.) so the agent doesn't need a second lookup and
    the model gets both application-level and customer-level features in
    one row.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT ca.*,
                   c.name AS customer_name,
                   c.age AS customer_age,
                   c.occupation AS customer_occupation,
                   c.annual_income AS customer_annual_income,
                   c.monthly_inhand_salary AS customer_monthly_inhand_salary,
                   c.num_bank_accounts AS customer_num_bank_accounts,
                   c.num_credit_card AS customer_num_credit_card
            FROM credit_applications ca
            JOIN customers c ON c.customer_id = ca.customer_id
            WHERE ca.application_id = ?
            """,
            (application_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_customer_transactions(customer_id: str, limit: int = 50) -> list[dict]:
    """
    Used by the AML agent — pulls recent transaction history for a customer
    so anomaly detection has more than a single transaction to reason over.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM transactions
            WHERE customer_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (customer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_transaction(transaction_id: str) -> dict | None:
    """Used by the AML agent when routed a single transaction to check."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_application_status(application_id: str, status: str) -> None:
    """Lets the CDA agent (or a human reviewer) write a decision back."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE credit_applications SET status = ? WHERE application_id = ?",
            (status, application_id),
        )
        conn.commit()
    finally:
        conn.close()


def flag_transaction(transaction_id: str, flagged: bool = True) -> None:
    """Lets the AML agent mark a transaction as flagged after review."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE transactions SET is_laundering = ? WHERE transaction_id = ?",
            (1 if flagged else 0, transaction_id),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    load_customers_from_csv("../data/cleaned_credit/cleaned_train.csv")
    load_customers_from_csv("../data/cleaned_credit/cleaned_test.csv")
    load_credit_applications_from_csv("../data/cleaned_credit/cleaned_train.csv")
    load_credit_applications_from_csv("../data/cleaned_credit/cleaned_test.csv")
    load_transactions_from_csv("../data/simulated/customer_transactions_train.csv", id_prefix="TRAIN")
    load_transactions_from_csv("../data/simulated/customer_transactions_test.csv", id_prefix="TEST")

    conn = get_connection()
    sample_customer_id = conn.execute("SELECT customer_id FROM customers LIMIT 1").fetchone()[0]
    sample_application_id = conn.execute("SELECT application_id FROM credit_applications LIMIT 1").fetchone()[0]
    sample_transaction_id = conn.execute("SELECT transaction_id FROM transactions LIMIT 1").fetchone()[0]
    conn.close()

    print("customer:", get_customer(sample_customer_id))
    print("application:", get_credit_application(sample_application_id))
    print("transaction:", get_transaction(sample_transaction_id))