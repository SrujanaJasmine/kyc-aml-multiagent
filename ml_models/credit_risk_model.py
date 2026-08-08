from __future__ import annotations
import numpy as np
import pandas as pd


class CreditRiskPredictor:
    """
    Wraps: one-hot encoding -> column alignment -> correlated-feature
    drop -> scaling -> trained classifier, so a downstream agent can
    call `.predict()` / `.predict_proba()` directly on a raw dataframe
    that looks like the original cleaned_train.csv (minus the label).

    Attributes
    ----------
    model : fitted sklearn/xgboost/lightgbm classifier
    scaler : fitted sklearn StandardScaler
    encode_cols : list[str]          columns that get one-hot encoded
    train_columns : list[str]        final column order the model expects (post scaling)
    dropped_corr_cols : list[str]    columns removed for multicollinearity
    loan_cols : list[str]            loan indicator columns filled with 0 if missing
    id_cols : list[str]              identifier columns dropped if present
    label_col : str
    class_names : dict               {0: "Good/Normal", 1: "High Risk"}
    metrics : dict                   evaluation metrics captured at training time
    model_name : str
    """

    def __init__(
        self,
        model,
        scaler,
        encode_cols,
        pre_scale_columns,
        train_columns,
        dropped_corr_cols,
        loan_cols,
        id_cols,
        label_col="Credit_Score",
        class_names=None,
        metrics=None,
        model_name="model",
    ):
        self.model = model
        self.scaler = scaler
        self.encode_cols = list(encode_cols)
        self._pre_scale_columns = list(pre_scale_columns)
        self.train_columns = list(train_columns)
        self.dropped_corr_cols = list(dropped_corr_cols)
        self.loan_cols = list(loan_cols)
        self.id_cols = list(id_cols)
        self.label_col = label_col
        self.class_names = class_names or {0: "Good / Low Risk", 1: "High Risk"}
        self.metrics = metrics or {}
        self.model_name = model_name

    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # drop identifier / label columns if present
        drop_these = [c for c in self.id_cols + [self.label_col] if c in df.columns]
        df = df.drop(columns=drop_these, errors="ignore")

        # loan indicator columns: fill missing with 0
        for c in self.loan_cols:
            if c not in df.columns:
                df[c] = 0
        df[self.loan_cols] = df[self.loan_cols].fillna(0)

        # one-hot encode categoricals
        encode_here = [c for c in self.encode_cols if c in df.columns]
        df = pd.get_dummies(df, columns=encode_here, drop_first=True)

        # align to the exact column set/order seen at training time
        # (this happens BEFORE the correlation-drop + scaling step, so we
        # reindex against the pre-drop training column list stored below)
        df = df.reindex(columns=self._pre_scale_columns, fill_value=0)

        # drop the multicollinear columns exactly as done at train time
        df = df.drop(columns=self.dropped_corr_cols, errors="ignore")

        # scale
        scaled = self.scaler.transform(df[self.train_columns])
        scaled_df = pd.DataFrame(scaled, columns=self.train_columns, index=df.index)
        return scaled_df

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._preprocess(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._preprocess(df)
        return self.model.predict_proba(X)

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for the CDA agent: returns a tidy dataframe
        with predicted label, human-readable class, and probability of
        the 'High Risk' class."""
        proba = self.predict_proba(df)
        pred = np.argmax(proba, axis=1)
        return pd.DataFrame(
            {
                "prediction": pred,
                "risk_label": [self.class_names[p] for p in pred],
                "high_risk_probability": proba[:, 1],
            },
            index=df.index,
        )
