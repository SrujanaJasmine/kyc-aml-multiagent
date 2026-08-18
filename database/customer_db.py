"""
customer_db.py
==============
SQLite schema, CSV loaders and read accessors for the three business tables:
customers, credit_applications and transactions. `credit_applications` uses the
credit model's own feature names; the other two use snake_case.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

# Anchored to database/ so the DB is found no matter where python was
# launched from (repo root for graph.py, database/ for this script).
# NAMING CONVENTION, DELIBERATELY SPLIT
# ------------------------------------
# `credit_applications` uses the credit model's own feature names (Month,
# Outstanding_Debt, Auto_Loan). The model's feature contract and this table are
# therefore the same vocabulary, and rows can be scored without translation.
#
# `customers` and `transactions` stay snake_case: they are read by KYC and AML,
# which have no such contract to satisfy, and renaming them would spread churn
# for no gain. The six customer-level fields the credit model needs are still
# bridged by ml_models/feature_mapping.py.
DB_PATH = str(Path(__file__).resolve().parent / "customer_data.db")
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


def _load_csv_mapped(csv_path: str, column_map: dict[str, str]) -> pd.DataFrame:
    """
    Reads only the CSV columns in `column_map` that actually exist in this
    file, and fills any mapped db column whose source column is missing
    with None. This tolerates a cleaned/pre-processed CSV that dropped or
    never had some columns — e.g. a test split commonly drops the label
    column (Credit_Score) entirely, or a train/test pair may not have
    identical schemas after separate cleaning passes.

    Without this, pd.read_csv(..., usecols=[...]) raises a ValueError the
    moment ANY requested column is missing, even if the rest are fine.
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    available_csv_cols = [c for c in column_map if c in header]
    missing_csv_cols = [c for c in column_map if c not in header]

    if missing_csv_cols:
        print(
            f"[customer_db] {csv_path}: {len(missing_csv_cols)} expected "
            f"column(s) not found, storing as NULL: {missing_csv_cols}"
        )

    df = pd.read_csv(csv_path, usecols=available_csv_cols)
    df = df.rename(columns=column_map)

    for csv_col in missing_csv_cols:
        df[column_map[csv_col]] = None

    return df[list(column_map.values())]  # enforce consistent column order


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
                customer_id            TEXT PRIMARY KEY,
                name                   TEXT,
                ssn                    TEXT,
                age                    INTEGER,
                occupation              TEXT,
                annual_income          REAL,
                monthly_inhand_salary  REAL,
                num_bank_accounts      INTEGER,
                num_credit_card        INTEGER
            );

            CREATE TABLE IF NOT EXISTS credit_applications (
                application_id             TEXT PRIMARY KEY,
                customer_id                 TEXT NOT NULL REFERENCES customers(customer_id),
                Month                        TEXT,
                Interest_Rate                REAL,
                Num_of_Loan                  REAL,
                Delay_from_due_date          REAL,
                Num_of_Delayed_Payment       REAL,
                Changed_Credit_Limit         REAL,
                Num_Credit_Inquiries         REAL,
                Credit_Mix                   TEXT,
                Outstanding_Debt             REAL,
                Credit_Utilization_Ratio     REAL,
                Payment_of_Min_Amount        TEXT,
                Total_EMI_per_month          REAL,
                Amount_invested_monthly      REAL,
                Payment_Behaviour            TEXT,
                Monthly_Balance              REAL,
                Credit_Score                 INTEGER,   -- dataset's own label; NOT a model feature, see cda_agent.py
                Credit_History_Age_Total     REAL,
                Auto_Loan                    REAL DEFAULT 0,
                Credit_Builder_Loan          REAL DEFAULT 0,
                Debt_Consolidation_Loan      REAL DEFAULT 0,
                Home_Equity_Loan             REAL DEFAULT 0,
                Mortgage_Loan                REAL DEFAULT 0,
                Not_Specified           REAL DEFAULT 0,
                Payday_Loan                  REAL DEFAULT 0,
                Personal_Loan                REAL DEFAULT 0,
                Student_Loan                 REAL DEFAULT 0,
                Unknown_Loan                 REAL DEFAULT 0,
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
    "Month": "Month",
    "Interest_Rate": "Interest_Rate",
    "Num_of_Loan": "Num_of_Loan",
    "Delay_from_due_date": "Delay_from_due_date",
    "Num_of_Delayed_Payment": "Num_of_Delayed_Payment",
    "Changed_Credit_Limit": "Changed_Credit_Limit",
    "Num_Credit_Inquiries": "Num_Credit_Inquiries",
    "Credit_Mix": "Credit_Mix",
    "Outstanding_Debt": "Outstanding_Debt",
    "Credit_Utilization_Ratio": "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount": "Payment_of_Min_Amount",
    "Total_EMI_per_month": "Total_EMI_per_month",
    "Amount_invested_monthly": "Amount_invested_monthly",
    "Payment_Behaviour": "Payment_Behaviour",
    "Monthly_Balance": "Monthly_Balance",
    "Credit_Score": "Credit_Score",
    "Credit_History_Age_Total": "Credit_History_Age_Total",
    "Auto Loan": "Auto_Loan",
    "Credit-Builder Loan": "Credit_Builder_Loan",
    "Debt Consolidation Loan": "Debt_Consolidation_Loan",
    "Home Equity Loan": "Home_Equity_Loan",
    "Mortgage Loan": "Mortgage_Loan",
    "Not Specified": "Not_Specified",
    "Payday Loan": "Payday_Loan",
    "Personal Loan": "Personal_Loan",
    "Student Loan": "Student_Loan",
    "Unknown Loan": "Unknown_Loan",
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
    df = _load_csv_mapped(csv_path, _CUSTOMER_COLUMNS)
    df = df.drop_duplicates(subset="customer_id", keep="first")

    # Age / Num_Bank_Accounts / Num_Credit_Card come in as float64 in the
    # CSV (e.g. 28.0) — coerce first in case any stray non-numeric values
    # slipped through cleaning, then round and cast to nullable Int64 so a
    # missing value doesn't force the whole column to stay float.
    for col in ("age", "num_bank_accounts", "num_credit_card"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

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

    Doesn't require load_customers_from_csv to have already run on this
    exact file — any customer_id referenced here but not yet in
    `customers` gets a bare stub row first (customer_id only, everything
    else NULL), so a train/test pair with a different customer population
    never trips the customer_id foreign key.
    """
    df = _load_csv_mapped(csv_path, _APPLICATION_COLUMNS)

    loan_cols = [_APPLICATION_COLUMNS[c] for c in _LOAN_TYPE_CSV_COLUMNS]
    df[loan_cols] = df[loan_cols].fillna(0)

    df = _clean_for_sqlite(df)  # pd.NA/np.nan -> None, or sqlite3 raises on insert

    cols = list(_APPLICATION_COLUMNS.values())
    conn = get_connection()
    try:
        # Stub in any customer_id referenced here but missing from
        # `customers`, so the FK on credit_applications.customer_id never fails.
        unique_customers = df["customer_id"].dropna().unique().tolist()
        conn.executemany(
            "INSERT OR IGNORE INTO customers (customer_id) VALUES (?)",
            [(cid,) for cid in unique_customers],
        )

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
    # then select by name afterward — tolerating any column this
    # particular file doesn't have.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing_csv_cols = [c for c in _TRANSACTION_COLUMNS if c not in header]
    if missing_csv_cols:
        print(
            f"[customer_db] {csv_path}: {len(missing_csv_cols)} expected "
            f"column(s) not found, storing as NULL: {missing_csv_cols}"
        )

    df = pd.read_csv(csv_path)
    for csv_col in missing_csv_cols:
        df[csv_col] = None
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


def get_customer_relationship(customer_id: str) -> dict:
    """
    How deep is the bank's existing relationship with this customer.

    Used by the KYC agent to answer "existing or new" with evidence rather than
    a bare boolean. Counts and date bounds are computed in SQL rather than by
    pulling rows into Python, because a customer can have thousands of
    transactions and KYC only needs the summary.
    """
    conn = get_connection()
    try:
        app_count = conn.execute(
            "SELECT COUNT(*) FROM credit_applications WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]

        row = conn.execute(
            """
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM transactions WHERE customer_id = ?
            """,
            (customer_id,),
        ).fetchone()
        txn_count, first_seen, last_seen = row[0], row[1], row[2]

        return {
            "applications_on_file": int(app_count),
            "transactions_on_file": int(txn_count),
            "first_seen": first_seen,
            "last_seen": last_seen,
        }
    finally:
        conn.close()


def get_customer_applications(customer_id: str, limit: int = 20) -> list[dict]:
    """
    Prior credit applications for a customer, newest first.

    Added because the CDA agent now receives the application under assessment
    as a payload rather than reading it from the database. The database's role
    for CDA shifted from "supply the application" to "supply the history it
    should be judged against" — repeat delinquency across past applications is
    a different signal from one bad Month, and the model's 59 features contain
    no history at all.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT application_id, Month, Credit_Utilization_Ratio,
                   Delay_from_due_date, Num_of_Delayed_Payment,
                   Outstanding_Debt, Credit_Mix, Payment_of_Min_Amount,
                   Num_Credit_Inquiries, status
            FROM credit_applications
            WHERE customer_id = ?
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (customer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
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
    load_customers_from_csv(str(_DATA_DIR / "cleaned_credit" / "cleaned_train.csv"))
    load_credit_applications_from_csv(str(_DATA_DIR / "cleaned_credit" / "cleaned_train.csv"))
    load_transactions_from_csv(str(_DATA_DIR / "simulated" / "customer_transactions_train.csv"), id_prefix="TRAIN")


    conn = get_connection()
    sample_customer_id = conn.execute("SELECT customer_id FROM customers LIMIT 1").fetchone()[0]
    sample_application_id = conn.execute("SELECT application_id FROM credit_applications LIMIT 1").fetchone()[0]
    sample_transaction_id = conn.execute("SELECT transaction_id FROM transactions LIMIT 1").fetchone()[0]
    conn.close()

    print("customer:", get_customer(sample_customer_id))
    print("application:", get_credit_application(sample_application_id))
    print("transaction:", get_transaction(sample_transaction_id))