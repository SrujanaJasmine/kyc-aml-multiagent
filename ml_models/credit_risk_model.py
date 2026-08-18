"""
credit_risk_model.py
====================
CreditScorer loads the trained credit model with its feature contract and decision
threshold, and turns raw application rows into risk probabilities and labels. Note
the model is trained on UNSCALED features -- no scaler belongs in the inference
path.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Artifacts live next to this file, in ml_models/artifacts/. Anchoring to
# __file__ means the scorer works no matter which directory the process was
# launched from — the agents are imported from the repo root, but the
# evaluation script may be run from ml_models/.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "xgboost_credit_model.joblib"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "best_threshold.joblib"

# The five columns the notebook one-hot encoded (cell 3).
ENCODE_COLS = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

CLASS_NAMES = {0: "Good / Low Risk", 1: "High Risk"}


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise feature names to clean identifiers.

    The source dataset carries names with spaces and hyphens -- "Auto Loan",
    "Credit-Builder Loan", "Not Specified" -- and one-hot expansion adds more
    ("Credit_Mix_Unknown Credit Mix"). Those are legal pandas columns but awful
    SQL identifiers: every query touching them needs double quotes, and one
    missed quote is a runtime error rather than a lint failure.

    Applied identically at training and inference, so the saved feature
    contract and the scoring frame always agree. It must stay in one place for
    that reason -- two copies of a normalisation rule drift, and the symptom is
    silently misaligned columns rather than an exception.
    """
    return df.rename(columns=lambda c: re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_"))


class CreditScorer:
    """
    Loads the trained XGBoost classifier plus the feature contract and turns
    a dataframe of raw credit-application rows into calibrated risk scores.

    The dataframe passed to `score()` should use the ORIGINAL dataset column
    names (Age, Annual_Income, Credit_Mix, ...), not the snake_case SQLite
    names. `agents/feature_mapping.py` handles that translation, so the
    scorer stays a pure model-contract object with no database knowledge.
    """

    def __init__(self, model, feature_columns: list[str], threshold: float):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.threshold = float(threshold)
        self.class_names = CLASS_NAMES

    # ------------------------------------------------------------------
    @classmethod
    def load(cls) -> "CreditScorer":
        """Read the three artifacts the notebook saved. Raises FileNotFoundError
        with an actionable message if training hasn't been run yet."""
        missing = [p.name for p in (MODEL_PATH, FEATURE_COLUMNS_PATH, THRESHOLD_PATH)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {ARTIFACT_DIR}: {', '.join(missing)}. "
                "Run the final cell of ml_models/cda_model.ipynb to regenerate them."
            )
        return cls(
            model=joblib.load(MODEL_PATH),
            feature_columns=joblib.load(FEATURE_COLUMNS_PATH),
            threshold=float(joblib.load(THRESHOLD_PATH)),
        )

    # ------------------------------------------------------------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode, then reindex onto the exact 59 training columns.

        Note `drop_first=False` here, unlike training. At training time
        drop_first=True removed one reference category per variable to avoid
        collinearity. At inference we cannot drop the first category of a
        single row — the row might BE that category, and dropping it would
        delete the only column present. Instead we expand everything and let
        the reindex discard the reference columns, which leaves them as an
        all-zeros row. That is exactly what drop_first encodes, arrived at
        from the other direction.
        """
        df = df.copy()

        encode_here = [c for c in ENCODE_COLS if c in df.columns]
        if encode_here:
            df = pd.get_dummies(df, columns=encode_here, drop_first=False)

        # Same normalisation the training path applied, so the reindex below
        # matches on identical names rather than near-misses.
        df = sanitize_columns(df)

        # Any training column absent from this row becomes 0; any extra
        # column here (application_id, Credit_Score label, ...) is dropped.
        # This single call is what guarantees the model never sees the label.
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # get_dummies yields bools and reindex yields ints; XGBoost wants a
        # uniform numeric matrix.
        return df.astype(float)

    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.build_features(df))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns one row per input row: probability of High Risk, the label
        implied by the tuned threshold, and the human-readable class name.

        Uses `self.threshold` (0.30, tuned for best F1 in notebook cell 14)
        rather than argmax/0.5. The dataset is imbalanced, so 0.5 would under-
        flag risky applications — the threshold is the whole point of having
        saved it.
        """
        proba = self.predict_proba(df)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "high_risk_probability": proba,
                "prediction": pred,
                "risk_label": [self.class_names[p] for p in pred],
            },
            index=df.index,
        )


# Backwards-compatible alias: older code (and any pickle referencing the old
# name) can still import CreditRiskPredictor without breaking.
CreditRiskPredictor = CreditScorer


if __name__ == "__main__":
    scorer = CreditScorer.load()
    print(f"Model:            {type(scorer.model).__name__}")
    print(f"Features:         {len(scorer.feature_columns)}")
    print(f"Decision threshold: {scorer.threshold:.4f}")
    print(f"First 8 features: {scorer.feature_columns[:8]}")
