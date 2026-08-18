"""
data_splits.py
==============
Splits the labelled credit data four ways -- train, validation, test and system --
by CUSTOMER rather than by row, using a deterministic hash so the partition is
identical on every machine and every run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR, SPLIT_DIR  # noqa: E402

SOURCE_CSV = DATA_DIR / "cleaned_credit" / "cleaned_train.csv"
ID_COL = "Customer_ID"
LABEL_COL = "Credit_Score"

# Cumulative bucket boundaries over 100 MD5 buckets.
SPLIT_BOUNDS: list[tuple[str, int]] = [
    ("train", 60),    # buckets  0–59
    ("val", 75),      # buckets 60–74
    ("test", 90),     # buckets 75–89
    ("system", 100),  # buckets 90–99
]
SPLIT_NAMES = [name for name, _ in SPLIT_BOUNDS]


def assign_split(customer_id: str) -> str:
    """Map a customer ID to one of the four splits via a stable hash."""
    bucket = int(hashlib.md5(str(customer_id).encode("utf-8")).hexdigest(), 16) % 100
    for name, upper in SPLIT_BOUNDS:
        if bucket < upper:
            return name
    return SPLIT_NAMES[-1]


def load_source() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise SystemExit(
            f"Source not found: {SOURCE_CSV}\n"
            "Run EDA/credit_eda.ipynb to produce data/cleaned_credit/cleaned_train.csv."
        )
    df = pd.read_csv(SOURCE_CSV)
    for col in (ID_COL, LABEL_COL):
        if col not in df.columns:
            raise SystemExit(f"Required column '{col}' missing from {SOURCE_CSV.name}")
    return df


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    side = df[ID_COL].map(assign_split)
    return {name: df[side == name].copy() for name in SPLIT_NAMES}


def get_split(name: str) -> pd.DataFrame:
    """
    Load one split, preferring a written CSV and falling back to recomputing.

    The fallback matters: because assignment is a pure hash, recomputing gives
    byte-identical membership to whatever was written earlier. Callers never
    have to care whether --write was run first.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{name}'. Expected one of {SPLIT_NAMES}")

    cached = SPLIT_DIR / f"{name}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    return split_frame(load_source())[name]


def write_splits(splits: dict[str, pd.DataFrame]) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(SPLIT_DIR / f"{name}.csv", index=False)


def report(df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    total_rows, total_cust = len(df), df[ID_COL].nunique()
    print(f"Source: {SOURCE_CSV.name}  —  {total_rows:,} rows / {total_cust:,} customers\n")
    print(f"{'split':<8}{'rows':>10}{'row %':>9}{'customers':>12}{'positive rate':>16}")
    print("-" * 55)
    for name in SPLIT_NAMES:
        frame = splits[name]
        rate = frame[LABEL_COL].mean() if LABEL_COL in frame.columns and len(frame) else float("nan")
        print(f"{name:<8}{len(frame):>10,}{len(frame) / total_rows:>8.1%}"
              f"{frame[ID_COL].nunique():>12,}{rate:>15.3f}")

    # Leakage check. A non-zero overlap means the split is broken and every
    # metric produced downstream is worthless, so it is asserted, not printed
    # and ignored.
    sets = {name: set(splits[name][ID_COL]) for name in SPLIT_NAMES}
    overlaps = {
        f"{a}/{b}": len(sets[a] & sets[b])
        for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]
    }
    print("\ncustomer overlap between splits (all must be 0):")
    for pair, count in overlaps.items():
        print(f"  {pair:<16}{count}")
    assert not any(overlaps.values()), "customer leakage between splits"

    covered = sum(len(splits[n]) for n in SPLIT_NAMES)
    assert covered == total_rows, f"row accounting mismatch: {covered} vs {total_rows}"
    print(f"\nAll {total_rows:,} rows accounted for, no customer in two splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-way customer-level split")
    parser.add_argument("--write", action="store_true",
                        help=f"write the four CSVs to {SPLIT_DIR}")
    args = parser.parse_args()

    df = load_source()
    splits = split_frame(df)
    report(df, splits)

    if args.write:
        write_splits(splits)
        print(f"\nWritten to {SPLIT_DIR}")
        for name in SPLIT_NAMES:
            print(f"  {name}.csv")
    else:
        print("\n(dry run — pass --write to save the CSVs)")


if __name__ == "__main__":
    main()
