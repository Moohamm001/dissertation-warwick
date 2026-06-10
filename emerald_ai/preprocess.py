"""Phase 2b — leakage-safe preprocessing pipeline.

A single ``ColumnTransformer`` that ONLY ever sees the vetted permitted columns. Categoricals are
split by cardinality: low-card -> one-hot, high-card -> target encoding (fit inside CV folds by the
caller, never on the full data). A hard ``assert_no_leakage`` guard refuses to build a pipeline
over any forbidden column — the automated check the roadmap (Gate, Phase 2) promises.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import feature_audit as FA

HIGH_CARD_THRESHOLD = 20  # categoricals with more levels are target-encoded, not one-hot


class LeakageError(AssertionError):
    """Raised when a forbidden / non-permitted column reaches the preprocessor."""


def assert_no_leakage(columns) -> None:
    """Guarantee every input column is on the permitted allowlist. The core safety check."""
    permitted = set(FA.permitted_columns())
    offenders = [c for c in columns if c not in permitted]
    if offenders:
        raise LeakageError(
            f"non-permitted column(s) reached the pipeline: {offenders}. "
            "Inputs must be a subset of feature_audit.permitted_columns()."
        )


def split_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the permitted columns present in ``df`` into numeric / low-card / high-card."""
    permitted = [c for c in FA.permitted_columns() if c in df.columns]
    assert_no_leakage(permitted)
    numeric, low_card, high_card = [], [], []
    for c in permitted:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        elif df[c].nunique(dropna=True) > HIGH_CARD_THRESHOLD:
            high_card.append(c)
        else:
            low_card.append(c)
    return {"numeric": numeric, "low_card": low_card, "high_card": high_card}


def build_preprocessor(df: pd.DataFrame, scale: bool = False) -> tuple[ColumnTransformer, dict]:
    """Construct the leakage-safe ColumnTransformer. ``scale`` on only for distance/linear models.

    Returns (transformer, feature_type_map). The transformer is unfitted — the caller fits it
    inside each CV fold so target encoding never sees held-out rows.
    """
    types = split_feature_types(df)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    low_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
    ])
    high_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("target", TargetEncoder(random_state=20260609)),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, types["numeric"]),
            ("low", low_card_pipe, types["low_card"]),
            ("high", high_card_pipe, types["high_card"]),
        ],
        remainder="drop",  # anything not explicitly listed is dropped — belt and braces vs leakage
        verbose_feature_names_out=False,
    )
    return transformer, types
