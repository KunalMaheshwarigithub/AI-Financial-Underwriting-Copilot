"""Data-aware fuzzy indicators and historical-cohort comparison.

This module is deliberately supplementary: it never changes the trained
model's prediction or the deterministic policy decision. Peer statistics and
fuzzy indicators are exploratory explanations, not validated credit scores.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_DIR / "models" / "applicant_feature_store.parquet"
load_dotenv(PROJECT_DIR / ".env")

# Candidate features are limited to fields that the new-application form can
# reasonably supply and that are also available in the Home Credit cohort.
NUMERIC_COMPARISON_FEATURES = [
    ("CREDIT_TO_INCOME", 0.22),
    ("ANNUITY_TO_INCOME", 0.18),
    ("AGE_YEARS", 0.14),
    ("AMT_INCOME_TOTAL", 0.10),
    ("EMPLOYMENT_YEARS", 0.10),
    ("AMT_CREDIT", 0.08),
    ("CNT_CHILDREN", 0.04),
]

CATEGORICAL_COMPARISON_FEATURES = [
    ("NAME_INCOME_TYPE", 0.045),
    ("NAME_EDUCATION_TYPE", 0.035),
    ("NAME_CONTRACT_TYPE", 0.025),
    ("NAME_HOUSING_TYPE", 0.025),
    ("FLAG_OWN_CAR", 0.01),
    ("FLAG_OWN_REALTY", 0.01),
]

TRAINING_PROFILE_FALLBACK = {
    "dataset_name": "Home Credit Default Risk",
    "training_source": "application_train.csv (Kaggle competition dataset)",
    "training_applicants": 307511,
    "target_name": "TARGET",
    "target_definition": "TARGET=1 indicates the historical default outcome; TARGET=0 indicates no recorded default outcome.",
    "default_count": 24825,
    "non_default_count": 282686,
    "default_rate_percent": 8.0729,
    "non_default_rate_percent": 91.9271,
    "model_type": "HistGradientBoostingClassifier",
    "predictor_count": 244,
    "numeric_predictor_count": 228,
    "categorical_predictor_count": 16,
    "model_version": "historical_features_v1",
    "evaluation_metrics": {
        "ROC-AUC": 0.782369,
        "PR-AUC": 0.278541,
        "Brier score": 0.066097,
    },
    "important_limitations": [
        "The metrics describe evaluation on historical Home Credit data; they do not establish performance on new applicants with many unavailable historical features.",
        "The model uses imputation for missing features, so a numeric prediction can be produced even when important source data are unavailable.",
        "Fuzzy indicators and nearest-applicant comparisons are exploratory and do not override the model output or policy rules.",
        "The TARGET label is for retrospective evaluation and must never be used as an input feature for a prediction.",
    ],
}


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def _parse_payload_column(series: pd.Series) -> pd.DataFrame:
    records = []
    for value in series:
        if isinstance(value, dict):
            records.append(value)
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                records.append(parsed if isinstance(parsed, dict) else {})
            except (json.JSONDecodeError, TypeError):
                records.append({})
        else:
            records.append({})
    return pd.json_normalize(records)


def _derive_reference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add comparable derived variables when source columns are present."""
    df = frame.copy()
    lower_to_original = {str(c).casefold(): c for c in df.columns}

    # Normalize only the TARGET casing; preserve feature names expected by the
    # rest of the app.
    if "TARGET" not in df.columns and "target" in lower_to_original:
        df = df.rename(columns={lower_to_original["target"]: "TARGET"})

    def numeric_column(name: str) -> pd.Series | None:
        col = lower_to_original.get(name.casefold())
        if col is None:
            col = {str(c).casefold(): c for c in df.columns}.get(name.casefold())
        if col is None:
            return None
        return pd.to_numeric(df[col], errors="coerce")

    # Derive demographic and financial ratios only where they are missing.
    if "AGE_YEARS" not in df.columns:
        days_birth = numeric_column("DAYS_BIRTH")
        if days_birth is not None:
            df["AGE_YEARS"] = -days_birth / 365.25

    if "EMPLOYMENT_YEARS" not in df.columns:
        days_employed = numeric_column("DAYS_EMPLOYED")
        if days_employed is not None:
            employment = -days_employed / 365.25
            # Home Credit uses 365243 as a sentinel for unavailable/not
            # applicable employment duration; do not interpret it as years.
            employment = employment.where(days_employed < 0)
            df["EMPLOYMENT_YEARS"] = employment

    if "CREDIT_TO_INCOME" not in df.columns:
        credit = numeric_column("AMT_CREDIT")
        income = numeric_column("AMT_INCOME_TOTAL")
        if credit is not None and income is not None:
            df["CREDIT_TO_INCOME"] = credit / income.where(income > 0)

    if "ANNUITY_TO_INCOME" not in df.columns:
        annuity = numeric_column("AMT_ANNUITY")
        income = numeric_column("AMT_INCOME_TOTAL")
        if annuity is not None and income is not None:
            df["ANNUITY_TO_INCOME"] = annuity / income.where(income > 0)

    if "CREDIT_TO_GOODS_PRICE" not in df.columns:
        credit = numeric_column("AMT_CREDIT")
        goods = numeric_column("AMT_GOODS_PRICE")
        if credit is not None and goods is not None:
            df["CREDIT_TO_GOODS_PRICE"] = credit / goods.where(goods > 0)

    for name in [
        "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "AGE_YEARS",
        "AMT_INCOME_TOTAL", "EMPLOYMENT_YEARS", "AMT_CREDIT",
        "CNT_CHILDREN", "TARGET",
    ]:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")

    return df


def _load_reference_from_postgres() -> tuple[pd.DataFrame | None, str]:
    """Fallback to the labeled historical applicant rows stored in PostgreSQL.

    Only a compact set of profile fields and TARGET are selected. TARGET is
    retained for cohort outcome summaries, never for similarity matching.
    """
    password = os.getenv("DB_PASSWORD")
    if not password:
        return None, "No labeled parquet and DB_PASSWORD is not configured for PostgreSQL fallback."

    field_names = [
        "DAYS_BIRTH", "DAYS_EMPLOYED", "AGE_YEARS", "EMPLOYMENT_YEARS",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_GOODS_PRICE",
        "CNT_CHILDREN", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_CONTRACT_TYPE", "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    ]
    select_fields = [
        f"f.feature_payload ->> '{name}' AS \"{name}\""
        for name in field_names
    ]
    sql = (
        'SELECT a.target AS "TARGET", ' + ", ".join(select_fields) +
        ' FROM applications AS a JOIN applicant_features AS f USING (applicant_id)'
        ' WHERE a.target IN (0, 1)'
    )

    connection = None
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "underwriting_copilot"),
            user=os.getenv("DB_USER", "postgres"),
            password=password,
            connect_timeout=5,
        )
        with connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        if not rows:
            return None, "PostgreSQL contained no rows with TARGET=0 or TARGET=1."
        frame = pd.DataFrame(rows, columns=["TARGET"] + field_names)
        return _derive_reference_features(frame), "loaded from PostgreSQL labeled historical applicants"
    except Exception as exc:
        return None, f"PostgreSQL reference fallback failed: {type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()


@lru_cache(maxsize=1)
def _load_reference_frame() -> tuple[pd.DataFrame | None, str]:
    """Load just the fields needed for cohort analysis, if the artifact exists."""
    if not REFERENCE_PATH.exists():
        return _load_reference_from_postgres()

    try:
        # Read only the small subset needed, rather than loading all 244 model
        # features into memory. Fall back to reading the full file for unusual
        # nested feature-store layouts.
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(REFERENCE_PATH)
        column_names = list(parquet.schema.names)
        target_col = _first_existing(column_names, ["TARGET", "target"])
        payload_col = _first_existing(
            column_names,
            ["feature_payload", "features", "feature_vector"],
        )

        wanted = [
            "TARGET", "target", "DAYS_BIRTH", "DAYS_EMPLOYED",
            "AGE_YEARS", "EMPLOYMENT_YEARS", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
            "CREDIT_TO_GOODS_PRICE", "CNT_CHILDREN",
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_CONTRACT_TYPE", "NAME_HOUSING_TYPE",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        ]
        wanted_actual = [
            _first_existing(column_names, [name])
            for name in wanted
        ]
        wanted_actual = list(dict.fromkeys(c for c in wanted_actual if c))

        # If this is a flattened feature store, read a limited column subset.
        has_flat_features = any(
            _first_existing(column_names, [name])
            for name in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
        )

        if has_flat_features and target_col:
            frame = pd.read_parquet(
                REFERENCE_PATH,
                columns=wanted_actual,
            )
        elif payload_col:
            payload_frame = pd.read_parquet(
                REFERENCE_PATH,
                columns=[c for c in [target_col, payload_col] if c],
            )
            expanded = _parse_payload_column(payload_frame[payload_col])
            if target_col and target_col in payload_frame.columns:
                expanded["TARGET"] = payload_frame[target_col].to_numpy()
            frame = expanded
        else:
            frame = pd.read_parquet(REFERENCE_PATH)

    except Exception as exc:
        parquet_error = (
            "Could not load the historical reference parquet. "
            f"Details: {type(exc).__name__}: {exc}"
        )
        db_frame, db_status = _load_reference_from_postgres()
        if db_frame is not None:
            return db_frame, db_status
        return None, f"{parquet_error} {db_status}"

    frame = _derive_reference_features(frame)
    if "TARGET" not in frame.columns:
        db_frame, db_status = _load_reference_from_postgres()
        if db_frame is not None:
            return db_frame, db_status
        return None, (
            "The reference parquet does not contain TARGET labels. "
            f"{db_status}"
        )

    frame = frame.loc[frame["TARGET"].isin([0, 1])].copy()
    if frame.empty:
        return None, "No rows with valid binary TARGET labels were found."

    return frame.reset_index(drop=True), "loaded"


def get_training_profile() -> dict[str, Any]:
    """Return the known training profile plus actual local reference-file status."""
    profile = dict(TRAINING_PROFILE_FALLBACK)
    profile["evaluation_metrics"] = dict(
        TRAINING_PROFILE_FALLBACK["evaluation_metrics"]
    )
    profile["important_limitations"] = list(
        TRAINING_PROFILE_FALLBACK["important_limitations"]
    )

    frame, status = _load_reference_frame()
    profile["reference_file"] = REFERENCE_PATH.name
    profile["reference_file_status"] = status

    if frame is not None:
        target = frame["TARGET"].dropna()
        if not target.empty:
            positive = int((target == 1).sum())
            negative = int((target == 0).sum())
            profile["reference_labeled_rows"] = int(len(target))
            profile["reference_default_count"] = positive
            profile["reference_non_default_count"] = negative
            profile["reference_default_rate_percent"] = round(
                float(target.mean() * 100), 4
            )

    return profile


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _applicant_values(feature_payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(feature_payload)

    if _finite_float(values.get("AGE_YEARS")) is None:
        days_birth = _finite_float(values.get("DAYS_BIRTH"))
        if days_birth is not None:
            values["AGE_YEARS"] = -days_birth / 365.25

    if _finite_float(values.get("EMPLOYMENT_YEARS")) is None:
        days_employed = _finite_float(values.get("DAYS_EMPLOYED"))
        if days_employed is not None and days_employed < 0:
            values["EMPLOYMENT_YEARS"] = -days_employed / 365.25

    income = _finite_float(values.get("AMT_INCOME_TOTAL"))
    credit = _finite_float(values.get("AMT_CREDIT"))
    annuity = _finite_float(values.get("AMT_ANNUITY"))
    goods = _finite_float(values.get("AMT_GOODS_PRICE"))

    if _finite_float(values.get("CREDIT_TO_INCOME")) is None:
        if income and income > 0 and credit is not None:
            values["CREDIT_TO_INCOME"] = credit / income

    if _finite_float(values.get("ANNUITY_TO_INCOME")) is None:
        if income and income > 0 and annuity is not None:
            values["ANNUITY_TO_INCOME"] = annuity / income

    if _finite_float(values.get("CREDIT_TO_GOODS_PRICE")) is None:
        if goods and goods > 0 and credit is not None:
            values["CREDIT_TO_GOODS_PRICE"] = credit / goods

    return values


def _quantiles_for(frame: pd.DataFrame | None, feature: str) -> tuple[float, float, float] | None:
    if frame is None or feature not in frame.columns:
        return None
    values = pd.to_numeric(frame[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(values) < 20:
        return None
    q25, q50, q75 = values.quantile([0.25, 0.50, 0.75]).tolist()
    q25, q50, q75 = float(q25), float(q50), float(q75)
    if not all(math.isfinite(x) for x in (q25, q50, q75)):
        return None
    if q25 == q50 or q50 == q75:
        spread = max(abs(q50) * 0.01, 1e-6)
        q25 = min(q25, q50 - spread)
        q75 = max(q75, q50 + spread)
    return q25, q50, q75


def _membership(value: float, quantiles: tuple[float, float, float], inverse_risk: bool = False) -> dict[str, float]:
    """Fuzzy Low/Medium/High memberships around data-derived quartiles."""
    q25, q50, q75 = quantiles

    if value <= q25:
        low = 1.0
    elif value >= q50:
        low = 0.0
    else:
        low = (q50 - value) / max(q50 - q25, 1e-12)

    if value <= q25 or value >= q75:
        medium = 0.0
    elif value <= q50:
        medium = (value - q25) / max(q50 - q25, 1e-12)
    else:
        medium = (q75 - value) / max(q75 - q50, 1e-12)

    if value <= q50:
        high = 0.0
    elif value >= q75:
        high = 1.0
    else:
        high = (value - q50) / max(q75 - q50, 1e-12)

    if inverse_risk:
        low, high = high, low

    # Numeric edge cases can cause tiny floating-point overshoots.
    low = float(min(1.0, max(0.0, low)))
    medium = float(min(1.0, max(0.0, medium)))
    high = float(min(1.0, max(0.0, high)))

    return {
        "low_risk_membership": round(low, 4),
        "medium_risk_membership": round(medium, 4),
        "high_risk_membership": round(high, 4),
    }


def build_fuzzy_assessment(
    feature_payload: dict[str, Any],
    predicted_pd: float,
) -> dict[str, Any]:
    """Build an explainable, exploratory fuzzy risk indicator.

    The output is intentionally not used to override the model/policy decision.
    Thresholds for profile variables come from the local labeled reference
    dataset when available; probability cutoffs match the project's risk-band
    assumptions.
    """
    values = _applicant_values(feature_payload)
    frame, reference_status = _load_reference_frame()

    factors: list[dict[str, Any]] = []
    weighted_score = 0.0
    available_weight = 0.0

    pd_value = _finite_float(predicted_pd)
    if pd_value is not None:
        memberships = _membership(
            pd_value,
            (0.05, 0.10, 0.20),
            inverse_risk=False,
        )
        # Convert fuzzy memberships into a transparent heuristic score.
        factor_score = (
            memberships["low_risk_membership"] * 15.0
            + memberships["medium_risk_membership"] * 50.0
            + memberships["high_risk_membership"] * 85.0
        )
        weight = 0.65
        weighted_score += factor_score * weight
        available_weight += weight
        factors.append({
            "feature": "predicted_pd",
            "value": round(pd_value, 6),
            "display_value": f"{pd_value * 100:.2f}%",
            "weight": weight,
            "risk_score": round(factor_score, 2),
            **memberships,
            "threshold_source": "project risk-band assumptions: 5%, 10%, 20%",
            "interpretation": "The trained model's predicted probability is the largest contributor to this heuristic indicator.",
        })

    ratio_factors = [
        ("CREDIT_TO_INCOME", 0.15, False, "credit relative to annual income"),
        ("ANNUITY_TO_INCOME", 0.15, False, "annuity relative to annual income"),
        ("EMPLOYMENT_YEARS", 0.05, True, "employment duration; shorter duration maps to higher heuristic risk"),
    ]

    for feature, weight, inverse_risk, description in ratio_factors:
        value = _finite_float(values.get(feature))
        quantiles = _quantiles_for(frame, feature)
        if value is None or quantiles is None:
            continue

        memberships = _membership(
            value,
            quantiles,
            inverse_risk=inverse_risk,
        )
        factor_score = (
            memberships["low_risk_membership"] * 15.0
            + memberships["medium_risk_membership"] * 50.0
            + memberships["high_risk_membership"] * 85.0
        )
        weighted_score += factor_score * weight
        available_weight += weight

        factors.append({
            "feature": feature,
            "value": round(value, 6),
            "display_value": f"{value:.3f}" if "RATIO" in feature or "INCOME" in feature else f"{value:.1f} years",
            "weight": weight,
            "risk_score": round(factor_score, 2),
            **memberships,
            "training_quartiles": {
                "q25": round(quantiles[0], 6),
                "q50": round(quantiles[1], 6),
                "q75": round(quantiles[2], 6),
            },
            "threshold_source": "quartiles of the local historical reference data",
            "interpretation": description,
        })

    if available_weight > 0:
        score = weighted_score / available_weight
        if score < 30:
            band = "Low"
        elif score < 50:
            band = "Moderate"
        elif score < 70:
            band = "Elevated"
        else:
            band = "High"
        available = True
    else:
        score = None
        band = "Unavailable"
        available = False

    if factors:
        highest = sorted(
            factors,
            key=lambda item: item.get("risk_score", 0),
            reverse=True,
        )[:3]
        explanations = [
            f"{item['feature']} contributes a heuristic risk score of {item['risk_score']:.1f}/100"
            for item in highest
        ]
    else:
        explanations = [
            "Insufficient applicant values or reference-data distributions for a fuzzy profile."
        ]

    return {
        "available": available,
        "method": "transparent fuzzy membership indicator",
        "fuzzy_risk_score": round(float(score), 2) if score is not None else None,
        "fuzzy_risk_band": band,
        "factors": factors,
        "explanations": explanations,
        "reference_data_status": reference_status,
        "decision_use": "EXPLANATORY_ONLY_NOT_USED_TO_CHANGE_DECISION",
        "limitation": (
            "This heuristic is not trained or calibrated as a separate predictive model. "
            "It is for explanation and experimentation only, and must not be interpreted "
            "as a validated probability of default or used alone for lending decisions."
        ),
    }


def compare_to_training_cohort(
    feature_payload: dict[str, Any],
    max_comparables: int = 200,
) -> dict[str, Any]:
    """Compare an applicant with similar labeled historical profiles.

    TARGET is used only to summarize outcomes after matching; it is never
    included in the similarity calculation.
    """
    frame, status = _load_reference_frame()
    if frame is None:
        return {
            "available": False,
            "method": "weighted profile similarity",
            "reference_data_status": status,
            "limitation": "The historical reference file with TARGET labels could not be used.",
        }

    values = _applicant_values(feature_payload)
    target = pd.to_numeric(frame["TARGET"], errors="coerce").to_numpy(dtype=float)
    valid_target = np.isfinite(target) & np.isin(target, [0.0, 1.0])
    reference = frame.loc[valid_target].reset_index(drop=True)
    target = target[valid_target]

    weighted_sum = np.zeros(len(reference), dtype=float)
    total_weight = np.zeros(len(reference), dtype=float)
    features_used: list[str] = []

    for feature, weight in NUMERIC_COMPARISON_FEATURES:
        applicant_value = _finite_float(values.get(feature))
        if applicant_value is None or feature not in reference.columns:
            continue

        column = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(column)
        if finite.sum() < 20:
            continue

        q25, q75 = np.nanquantile(column[finite], [0.25, 0.75])
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = float(np.nanstd(column[finite]))
        if not math.isfinite(scale) or scale <= 1e-9:
            continue

        similarity = np.exp(-np.minimum(np.abs(column - applicant_value) / scale, 30.0))
        weighted_sum[finite] += similarity[finite] * weight
        total_weight[finite] += weight
        features_used.append(feature)

    for feature, weight in CATEGORICAL_COMPARISON_FEATURES:
        applicant_value = values.get(feature)
        if applicant_value is None or feature not in reference.columns:
            continue

        column = reference[feature].astype("string").str.strip().str.casefold()
        applicant_text = str(applicant_value).strip().casefold()
        finite = column.notna().to_numpy()
        match = (column == applicant_text).fillna(False).to_numpy(dtype=bool)
        weighted_sum[finite] += match[finite].astype(float) * weight
        total_weight[finite] += weight
        features_used.append(feature)

    if not features_used:
        return {
            "available": False,
            "method": "weighted profile similarity",
            "reference_data_status": status,
            "limitation": "No matching profile features were available for comparison.",
        }

    similarities = np.divide(
        weighted_sum,
        total_weight,
        out=np.full(len(reference), np.nan, dtype=float),
        where=total_weight > 0,
    )
    valid = np.isfinite(similarities)
    if not valid.any():
        return {
            "available": False,
            "method": "weighted profile similarity",
            "reference_data_status": status,
            "features_used": features_used,
            "limitation": "No historical rows had enough available comparison fields.",
        }

    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
    # Use reasonably similar rows for the exploratory rate. If very few rows
    # meet the cutoff, still show the nearest profiles but mark the estimate
    # insufficient rather than imply reliable evidence.
    cutoff_indices = order[similarities[order] >= 0.50]
    selected = cutoff_indices[:max_comparables]
    if len(selected) == 0:
        selected = order[:min(20, len(order))]

    selected_similarity = similarities[selected]
    selected_target = target[selected]
    similarity_mean = float(np.mean(selected_similarity)) if len(selected) else None
    default_rate = None
    if len(selected) >= 30:
        default_rate = round(float(np.mean(selected_target) * 100), 4)

    baseline_rate = round(float(np.mean(target) * 100), 4) if len(target) else None
    high_similarity_count = int(np.sum(similarities[valid] >= 0.65))

    if len(selected) >= 100 and similarity_mean is not None and similarity_mean >= 0.65:
        reliability_label = "Moderate exploratory evidence; not independently validated"
    elif len(selected) >= 30:
        reliability_label = "Limited exploratory evidence; not independently validated"
    else:
        reliability_label = "Insufficient comparable cases for a stable default-rate estimate"

    top_match = int(order[0])
    return {
        "available": True,
        "method": "weighted numeric similarity plus categorical exact-match signals; TARGET excluded from matching",
        "reference_data_status": status,
        "reference_labeled_applicants": int(len(reference)),
        "features_used": features_used,
        "nearest_similarity_percent": round(float(similarities[top_match] * 100), 2),
        "high_similarity_applicant_count": high_similarity_count,
        "comparison_sample_size": int(len(selected)),
        "mean_similarity_percent": round(similarity_mean * 100, 2) if similarity_mean is not None else None,
        "similar_cohort_observed_default_rate_percent": default_rate,
        "overall_reference_default_rate_percent": baseline_rate,
        "evidence_quality": reliability_label,
        "decision_use": "COMPARISON_ONLY_NOT_USED_TO_CHANGE_DECISION",
        "limitation": (
            "The cohort rate is an exploratory retrospective statistic, not a calibrated prediction "
            "for this applicant. Results depend on the saved reference file, the available form fields, "
            "and the similarity method."
        ),
    }
