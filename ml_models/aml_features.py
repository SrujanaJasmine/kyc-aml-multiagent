"""
aml_features.py
===============
Builds the 54 transaction features the AML model uses -- amount, time, payment
format, account velocity, degree, structuring and multi-hop graph patterns -- and
caches them to data/features/ so training and evaluation share one matrix.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from config import CUSTOMER_DB_PATH, DATA_DIR  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))



TRANSACTIONS_CSV = DATA_DIR / "simulated" / "customer_transactions_train.csv"

# Never features. Checked at the end of build_features().
LEAK_COLUMNS = {"_stream", "Typology", "typology", "Is Laundering", "is_laundering",
                "Customer_ID", "customer_id", "Transaction_ID", "transaction_id",
                "Split", "split", "_role"}

# BSA thresholds that shape the amount features. See policies/aml_rules.py for
# the citations; they are repeated as plain numbers here because the feature
# layer must not depend on the policy layer.
CTR_THRESHOLD = 10_000.0
SAR_THRESHOLD = 5_000.0
STRUCTURING_BAND = (2_000.0, 10_000.0)

# Fixed payment-format vocabulary.
#
# The one-hot columns MUST NOT be derived from whatever rows happen to be in
# front of us. At training time all six formats are present; at scoring time the
# agent sees one customer's account neighbourhood, which may contain only two or
# three. Deriving categories from the data then produces a narrower matrix than
# the model was trained on and the prediction fails with a missing-column error
# -- or worse, silently misaligns if the columns were merely reordered.
#
# These are the six formats present in the US-Dollar, US-bank slice of the IBM
# reference data, which is what the simulator reproduces. Bitcoin appears a
# handful of times and is deliberately excluded.
PAYMENT_FORMATS = ["ACH", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

VELOCITY_WINDOWS = {"1d": "1D", "7d": "7D", "30d": "30D"}
PASS_THROUGH_WINDOW = "7D"
CYCLE_MAX_LEN = 3

# Feature family prefixes, used by the evaluation script to report grouped
# importance. If the graph families contribute nothing, we want to see it.
FAMILIES = {
    "amount": "amt_",
    "time": "time_",
    "format": "fmt_",
    "velocity": "vel_",
    "degree": "deg_",
    "structuring": "struct_",
    "multihop": "hop_",
}


# ---------------------------------------------------------------------------
# Loading and normalisation
# ---------------------------------------------------------------------------
_CSV_RENAME = {
    "Transaction_ID": "transaction_id", "Timestamp": "timestamp",
    "From Bank": "from_bank", "Account": "from_account",
    "To Bank": "to_bank", "Account.1": "to_account",
    "Amount Received": "amount", "Payment Format": "payment_format",
    "Is Laundering": "is_laundering", "Customer_ID": "customer_id",
    "Split": "split", "Typology": "typology",
}

_DB_RENAME = {"amount_received": "amount"}


def load_transactions(source: str = "csv", split: str | list[str] | None = None,
                      customer_id: str | None = None) -> pd.DataFrame:
    """
    Load transactions from the CSV (training) or the database (agent runtime)
    into one canonical schema, so downstream code never branches on origin.

    `split` filters to one or more partitions. The database has no split column
    — the loader does not carry it — so split filtering only applies to the CSV.
    """
    if source == "csv":
        usecols = list(_CSV_RENAME)
        # Account identifiers as `category`, not object. At 4.5M rows the string
        # columns are the bulk of the frame -- categorical codes cut the memory
        # several-fold and make every groupby dramatically faster, because
        # pandas groups on int32 codes instead of hashing strings.
        df = pd.read_csv(TRANSACTIONS_CSV, usecols=usecols,
                         dtype={"Amount Received": "float64", "Is Laundering": "int8",
                                "Account": "category", "Account.1": "category",
                                "Payment Format": "category", "Customer_ID": "category",
                                "Split": "category", "Typology": "category"})
        df = df.rename(columns=_CSV_RENAME)
    elif source == "db":
        conn = sqlite3.connect(CUSTOMER_DB_PATH)
        try:
            query = ("SELECT transaction_id, customer_id, timestamp, from_bank, from_account, "
                     "to_bank, to_account, amount_received, payment_format, is_laundering "
                     "FROM transactions")
            params: tuple = ()
            if customer_id:
                query += " WHERE customer_id = ?"
                params = (customer_id,)
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
        df = df.rename(columns=_DB_RENAME)
        df["split"] = "unknown"
        df["typology"] = "NONE"
    else:
        raise ValueError(f"source must be 'csv' or 'db', got {source!r}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if split is not None:
        wanted = [split] if isinstance(split, str) else list(split)
        df = df[df["split"].isin(wanted)]

    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def load_neighbourhood(customer_id: str, hops: int = 1) -> pd.DataFrame:
    """
    Every transaction touching any account the customer touches.

    WHY NOT JUST THE CUSTOMER'S OWN TRANSACTIONS
    --------------------------------------------
    Training computed degree, pass-through and cycle features over the whole
    payment graph. If the agent scored a transaction using features derived from
    one customer's rows alone, in-degree would count only that customer's
    counterparties, pass-through ratios would be computed from a fraction of the
    flow, and no cycle could ever close. The model would receive inputs on a
    different scale from the ones it was trained on, and its probabilities would
    be quietly wrong -- the classic train/serve skew, and the failure mode is
    silent.

    Pulling the one-hop neighbourhood restores most of that context: every
    account the customer transacts with, and everything those accounts do. It is
    still narrower than the full graph, so a three-hop cycle passing entirely
    through strangers can be missed. The remedy, if that matters later, is to
    precompute account-level statistics once and have the agent join them; this
    function is the pragmatic version that needs no extra infrastructure.
    """
    conn = sqlite3.connect(CUSTOMER_DB_PATH)
    try:
        own = pd.read_sql_query(
            "SELECT from_account, to_account FROM transactions WHERE customer_id = ?",
            conn, params=(customer_id,))
        if own.empty:
            return pd.DataFrame()

        accounts = pd.unique(np.concatenate([own["from_account"].to_numpy(),
                                             own["to_account"].to_numpy()]))
        for _ in range(max(hops - 1, 0)):
            marks = ",".join("?" * len(accounts))
            more = pd.read_sql_query(
                f"SELECT from_account, to_account FROM transactions "
                f"WHERE from_account IN ({marks}) OR to_account IN ({marks})",
                conn, params=[*accounts, *accounts])
            accounts = pd.unique(np.concatenate([accounts,
                                                 more["from_account"].to_numpy(),
                                                 more["to_account"].to_numpy()]))

        marks = ",".join("?" * len(accounts))
        df = pd.read_sql_query(
            f"SELECT transaction_id, customer_id, timestamp, from_bank, from_account, "
            f"to_bank, to_account, amount_received, payment_format, is_laundering "
            f"FROM transactions WHERE from_account IN ({marks}) OR to_account IN ({marks})",
            conn, params=[*accounts, *accounts])
    finally:
        conn.close()

    df = df.rename(columns=_DB_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["split"] = "runtime"
    df["typology"] = "NONE"
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------
def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    amt = df["amount"].to_numpy(dtype="float64")
    out = pd.DataFrame(index=df.index)

    out["amt_log"] = np.log1p(amt)
    out["amt_raw"] = amt

    # Structuring is defined by evasion of a reporting threshold, so the
    # model needs the *distance* to that threshold, not just the amount.
    # An unsigned gap is what makes $9,900 and $500 different rather than
    # merely "both under $10,000".
    out["amt_below_ctr_gap"] = np.where(amt < CTR_THRESHOLD, CTR_THRESHOLD - amt, 0.0)
    out["amt_over_ctr"] = (amt >= CTR_THRESHOLD).astype("int8")
    out["amt_over_sar"] = (amt >= SAR_THRESHOLD).astype("int8")
    out["amt_in_struct_band"] = ((amt >= STRUCTURING_BAND[0]) & (amt < STRUCTURING_BAND[1])).astype("int8")
    out["amt_near_ctr"] = ((amt >= 9_000) & (amt < CTR_THRESHOLD)).astype("int8")

    # Round numbers indicate deliberate sizing rather than a real invoice.
    out["amt_round_100"] = (np.mod(amt, 100) == 0).astype("int8")
    out["amt_round_1000"] = (np.mod(amt, 1000) == 0).astype("int8")
    out["amt_cents"] = np.round(np.mod(amt, 1), 2)
    return out


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    out = pd.DataFrame(index=df.index)
    out["time_hour"] = ts.dt.hour.astype("int8")
    out["time_dow"] = ts.dt.dayofweek.astype("int8")
    out["time_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    out["time_is_offhours"] = ((ts.dt.hour < 7) | (ts.dt.hour >= 21)).astype("int8")
    out["time_day_index"] = (ts - ts.min()).dt.days.astype("int16")
    return out


def _format_features(df: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    # Defaults to the fixed vocabulary, never to what this particular frame
    # happens to contain -- see PAYMENT_FORMATS for why that matters.
    cats = categories or PAYMENT_FORMATS
    fmt = pd.Categorical(df["payment_format"], categories=cats)
    out = pd.get_dummies(fmt, prefix="fmt").astype("int8")
    out.index = df.index
    return out


def _trailing_stats(df: pd.DataFrame, key: str, prefix: str,
                    windows: dict[str, str]) -> pd.DataFrame:
    """
    Trailing count and sum per account over time windows.

    Uses groupby().rolling() on the timestamp rather than a fixed row count:
    "five transactions in the last day" is the signal, and a row-count window
    would mean something different for a busy account than a quiet one.
    """
    n = len(df)
    work = df[[key, "timestamp", "amount"]].copy()
    # Carry the original row position through the sort. groupby().rolling()
    # returns a (group, timestamp) index whose labels are not unique, so index
    # alignment cannot be used to get back; scattering by position can, and it
    # avoids building a large intermediate index besides.
    work["_pos"] = np.arange(n, dtype="int64")
    work = work.sort_values([key, "timestamp"], kind="mergesort")
    pos = work["_pos"].to_numpy()

    grouped = work.set_index("timestamp").groupby(key, observed=True, sort=False)["amount"]

    out = pd.DataFrame(index=df.index)
    for name, window in windows.items():
        rolled = grouped.rolling(window)
        cnt = np.empty(n, dtype="float32"); cnt[pos] = rolled.count().to_numpy()
        total = np.empty(n, dtype="float32"); total[pos] = rolled.sum().to_numpy()
        out[f"{prefix}_cnt_{name}"] = cnt
        out[f"{prefix}_sum_{name}"] = total

    # Gap since this account's previous transaction. Bursts are the temporal
    # signature of every injected typology. -1 marks an account's first
    # transaction, which is a real state rather than a missing value.
    ts = work["timestamp"].to_numpy()
    keys = work[key].to_numpy()
    same_account = np.zeros(len(work), dtype=bool)
    same_account[1:] = keys[1:] == keys[:-1]
    delta = np.zeros(len(work), dtype="float64")
    delta[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    gap = np.empty(n, dtype="float32")
    gap[pos] = np.where(same_account, delta, -1.0)
    out[f"{prefix}_secs_since_prev"] = gap

    return out


def _velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    src = _trailing_stats(df, "from_account", "vel_out", VELOCITY_WINDOWS)
    dst = _trailing_stats(df, "to_account", "vel_in", VELOCITY_WINDOWS)
    out = pd.concat([src, dst], axis=1)

    # How unusual is this amount for this account? A z-score against the
    # account's own 30-day history separates "large" from "large for them".
    mean_30 = out["vel_out_sum_30d"] / out["vel_out_cnt_30d"].clip(lower=1)
    out["vel_amt_vs_out_mean"] = (df["amount"].to_numpy() / mean_30.clip(lower=1e-6)).astype("float32")
    return out


def _degree_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # agg-then-map rather than transform(): transform("nunique") re-derives the
    # value for every row, which is the single slowest call in this module at
    # scale. Aggregating once per group and broadcasting is the same answer for
    # a fraction of the work.
    out_deg = df["from_account"].map(
        df.groupby("from_account", observed=True)["to_account"].nunique())
    in_deg = df["to_account"].map(
        df.groupby("to_account", observed=True)["from_account"].nunique())
    out_cnt = df["from_account"].map(
        df.groupby("from_account", observed=True)["amount"].size())
    in_cnt = df["to_account"].map(
        df.groupby("to_account", observed=True)["amount"].size())

    out["deg_out_unique"] = out_deg.astype("float32")
    out["deg_in_unique"] = in_deg.astype("float32")
    out["deg_out_txns"] = out_cnt.astype("float32")
    out["deg_in_txns"] = in_cnt.astype("float32")

    # Fan patterns are exactly a mismatch between distinct counterparties and
    # transaction volume, so the ratio carries more than either alone.
    out["deg_out_ratio"] = (out_deg / out_cnt.clip(lower=1)).astype("float32")
    out["deg_in_ratio"] = (in_deg / in_cnt.clip(lower=1)).astype("float32")

    pair_cnt = df.groupby(["from_account", "to_account"], observed=True)["amount"].transform("size")
    out["deg_pair_txns"] = pair_cnt.astype("float32")
    out["deg_pair_share"] = (pair_cnt / out_cnt.clip(lower=1)).astype("float32")
    return out


def _account_codes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Map both account columns onto one shared integer vocabulary.

    `from_account` and `to_account` are separate categoricals with different
    category sets, so their codes are not comparable. A single vocabulary lets
    the graph work run on int64 arrays instead of Python string tuples, which
    turns the cycle detection from millions of dict lookups into vectorised
    numpy — the difference between minutes and seconds at 4.5M rows.
    """
    accounts = pd.Index(
        pd.unique(np.concatenate([df["from_account"].astype(str).to_numpy(),
                                  df["to_account"].astype(str).to_numpy()])))
    src = accounts.get_indexer(df["from_account"].astype(str))
    dst = accounts.get_indexer(df["to_account"].astype(str))
    return src.astype("int64"), dst.astype("int64"), len(accounts)


def _structuring_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deposits in the $2k-$10k band arriving at one account inside 7 days.

    This is the direct computational analogue of 31 USC 5324: the offence is
    breaking one reportable transaction into several unreportable ones, so the
    detector counts band-restricted arrivals per destination account and sums
    them. A group summing past $10,000 is the thing the statute describes.
    """
    n = len(df)
    band = df["amount"].between(*STRUCTURING_BAND, inclusive="left").to_numpy()

    cnt_arr = np.zeros(n, dtype="float32")
    sum_arr = np.zeros(n, dtype="float32")

    if band.any():
        work = df.loc[band, ["to_account", "timestamp", "amount"]].copy()
        work["_pos"] = np.flatnonzero(band)
        work = work.sort_values(["to_account", "timestamp"], kind="mergesort")
        pos = work["_pos"].to_numpy()

        rolled = (work.set_index("timestamp")
                  .groupby("to_account", observed=True, sort=False)["amount"]
                  .rolling("7D"))
        cnt_arr[pos] = rolled.count().to_numpy()
        sum_arr[pos] = rolled.sum().to_numpy()

    out = pd.DataFrame(index=df.index)
    out["struct_band_cnt_7d"] = cnt_arr
    out["struct_band_sum_7d"] = sum_arr
    # The aggregate crossing $10,000 is what would have triggered a CTR had it
    # been a single transaction -- the definition of the evasion.
    out["struct_aggregates_over_ctr"] = (out["struct_band_sum_7d"] >= CTR_THRESHOLD).astype("int8")
    out["struct_aggregates_over_sar"] = (out["struct_band_sum_7d"] >= SAR_THRESHOLD).astype("int8")
    return out


def _multihop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass-through behaviour and cycle membership.

    Two things no per-transaction feature can see:

    * **Pass-through** — an account receiving and forwarding similar value in a
      short window is behaving as a conduit, which is the middle leg of
      scatter-gather. Computed from per-account in/out aggregates rather than by
      enumerating paths, so cost stays linear.
    * **Cycles** — funds returning to their origin. Computed on the *distinct
      edge set* (810k edges), not on transactions (4.5M), which is what keeps
      the joins small. Measured 2-hop pair count is ~3.19M, so no degree cap is
      needed on this graph.
    """
    out = pd.DataFrame(index=df.index)

    # --- pass-through ------------------------------------------------------
    in_sum = df.groupby("to_account", observed=True)["amount"].sum()
    out_sum = df.groupby("from_account", observed=True)["amount"].sum()
    in_cnt = df.groupby("to_account", observed=True)["amount"].size()
    out_cnt = df.groupby("from_account", observed=True)["amount"].size()

    nodes = in_sum.index.union(out_sum.index)
    flow = pd.DataFrame({
        "in_sum": in_sum.reindex(nodes).fillna(0.0),
        "out_sum": out_sum.reindex(nodes).fillna(0.0),
        "in_cnt": in_cnt.reindex(nodes).fillna(0),
        "out_cnt": out_cnt.reindex(nodes).fillna(0),
    })
    denom = flow[["in_sum", "out_sum"]].max(axis=1).clip(lower=1e-6)
    flow["pass_ratio"] = flow[["in_sum", "out_sum"]].min(axis=1) / denom
    flow["flow_imbalance"] = (flow["in_sum"] - flow["out_sum"]).abs() / denom

    for side, col in (("src", "from_account"), ("dst", "to_account")):
        mapped = df[col].map(flow["pass_ratio"])
        out[f"hop_{side}_pass_ratio"] = mapped.fillna(0.0).astype("float32")
        out[f"hop_{side}_flow_imbalance"] = df[col].map(flow["flow_imbalance"]).fillna(0.0).astype("float32")

    # --- cycles on the distinct edge set -----------------------------------
    # Everything below runs on int64 codes. An edge (a, b) is encoded as the
    # single integer a * n + b, so edge membership becomes np.isin over a sorted
    # array rather than millions of tuple hashes.
    src, dst, n_accounts = _account_codes(df)
    edge_key = src * n_accounts + dst
    unique_edges = np.unique(edge_key)

    # 2-cycle: does the reciprocal edge exist anywhere in the graph?
    out["hop_reciprocal_edge"] = np.isin(dst * n_accounts + src, unique_edges).astype("int8")

    # 3-cycle: a -> b -> c -> a, computed on the 810k distinct edges rather than
    # the 4.5M transactions. Measured two-hop pair count is ~3.19M and max
    # degree is 73, so this needs no degree capping on this graph.
    e_src = (unique_edges // n_accounts).astype("int64")
    e_dst = (unique_edges % n_accounts).astype("int64")
    edges = pd.DataFrame({"a": e_src, "b": e_dst})

    two_hop = edges.merge(edges.rename(columns={"a": "b", "b": "c"}), on="b")
    closes = np.isin(two_hop["c"].to_numpy() * n_accounts + two_hop["a"].to_numpy(),
                     unique_edges)
    cycle_edges = np.unique(
        two_hop.loc[closes, "a"].to_numpy() * n_accounts + two_hop.loc[closes, "b"].to_numpy())

    out["hop_in_3cycle"] = np.isin(edge_key, cycle_edges).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, format_categories: list[str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the feature matrix and a parallel metadata frame.

    Returns (X, meta). `meta` carries ids, label, split and typology so the
    evaluation can report per-typology recall — none of it ever reaches X.
    Keeping them in separate frames makes leakage a deliberate act rather than
    an oversight.
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    parts = [
        _amount_features(df),
        _time_features(df),
        _format_features(df, format_categories),
        _velocity_features(df),
        _degree_features(df),
        _structuring_features(df),
        _multihop_features(df),
    ]
    X = pd.concat(parts, axis=1)

    # float32 halves the matrix: at 4.5M rows and ~50 features that is the
    # difference between ~1.8 GB and ~900 MB, which matters on a 16 GB laptop
    # already holding the source frame.
    X = X.astype("float32")

    leaked = LEAK_COLUMNS & set(X.columns)
    if leaked:
        raise AssertionError(f"leak columns reached the feature matrix: {sorted(leaked)}")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise AssertionError(f"NaNs in features: {bad}")

    meta_cols = [c for c in ("transaction_id", "customer_id", "timestamp", "is_laundering",
                             "split", "typology", "amount", "payment_format",
                             "from_account", "to_account") if c in df.columns]
    meta = df[meta_cols].copy()

    return X, meta


def family_of(column: str) -> str:
    for family, prefix in FAMILIES.items():
        if column.startswith(prefix):
            return family
    return "other"


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
FEATURE_DIR = DATA_DIR / "features"
X_PATH = FEATURE_DIR / "aml_X.npy"
COLUMNS_PATH = FEATURE_DIR / "aml_columns.json"
META_PATH = FEATURE_DIR / "aml_meta.csv"


def cache_features() -> None:
    """
    Build features once over ALL transactions and write them to disk.

    Built on the full table rather than per split on purpose. Velocity, degree
    and cycle features describe an account's position in the whole payment
    graph, and merchant and peer accounts are shared across customers — so
    computing them per split would truncate an account's history at an
    arbitrary boundary and understate exactly the structure we are trying to
    detect. A production monitoring system sees every transaction regardless of
    which customer is under review, and this mirrors that.

    This is not label leakage: no feature reads `is_laundering`. The split is
    applied afterwards, at training time.
    """
    import json
    import time

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading all transactions ...", flush=True)
    df = load_transactions("csv")
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("building features over the full graph ...", flush=True)
    X, meta = build_features(df)
    print(f"  {X.shape[0]:,} x {X.shape[1]} in {time.time()-t0:.1f}s", flush=True)

    np.save(X_PATH, X.to_numpy(dtype="float32"))
    COLUMNS_PATH.write_text(json.dumps(list(X.columns), indent=1))
    meta.to_csv(META_PATH, index=False)

    size_gb = X_PATH.stat().st_size / 1e9
    print(f"\ncached to {FEATURE_DIR}")
    print(f"  aml_X.npy       {size_gb:.2f} GB")
    print(f"  aml_meta.csv    {META_PATH.stat().st_size/1e6:.0f} MB")


def load_cached(split: str | list[str] | None = None, mmap: bool = True):
    """
    Load the cached matrix, optionally filtered to one or more splits.

    `mmap=True` keeps the array on disk and pages in only what is touched, so a
    1 GB matrix does not have to fit in RAM alongside everything else. Filtering
    by split materialises just that slice.
    """
    import json

    if not X_PATH.exists():
        raise FileNotFoundError(
            f"No feature cache at {X_PATH}. Run: python -m ml_models.aml_features --cache")

    columns = json.loads(COLUMNS_PATH.read_text())
    meta = pd.read_csv(META_PATH, parse_dates=["timestamp"])
    X = np.load(X_PATH, mmap_mode="r" if mmap else None)

    if split is not None:
        wanted = [split] if isinstance(split, str) else list(split)
        mask = meta["split"].isin(wanted).to_numpy()
        X = np.asarray(X[mask])
        meta = meta.loc[mask].reset_index(drop=True)

    return pd.DataFrame(X, columns=columns), meta


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="AML feature engineering")
    parser.add_argument("--cache", action="store_true",
                        help="build features over all transactions and write them to data/features/")
    parser.add_argument("--split", default="train", help="split to inspect when not caching")
    args = parser.parse_args()

    if args.cache:
        cache_features()
        raise SystemExit(0)

    t0 = time.time()
    print(f"loading {args.split} split from CSV ...")
    df = load_transactions("csv", split=args.split)
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    X, meta = build_features(df)
    print(f"\nfeatures built in {time.time()-t0:.1f}s")
    print(f"  matrix    : {X.shape[0]:,} x {X.shape[1]}  ({X.memory_usage().sum()/1e9:.2f} GB)")
    print(f"  positives : {int(meta['is_laundering'].sum()):,} "
          f"({100*meta['is_laundering'].mean():.4f}%)")

    print("\nfeatures by family:")
    fam = pd.Series([family_of(c) for c in X.columns]).value_counts()
    for name, count in fam.items():
        print(f"  {name:<12} {count}")

    print("\nseparation check (mean by label, top 12 by ratio):")
    pos = X[meta["is_laundering"].to_numpy() == 1].mean()
    neg = X[meta["is_laundering"].to_numpy() == 0].mean()
    ratio = (pos / neg.replace(0, np.nan)).abs().sort_values(ascending=False)
    for col in ratio.head(12).index:
        print(f"  {col:<28} laundering {pos[col]:>12,.3f} | normal {neg[col]:>12,.3f} "
              f"| {ratio[col]:>7.2f}x")
