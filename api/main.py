# api/main.py
# AI Financial Underwriting Copilot — FastAPI backend
# PostgreSQL -> Feature retrieval -> ML -> Policy -> Explanations -> Audit

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import Json

from api.copilot import router as copilot_router
from api.new_applications import router as new_applications_router
from api.risk_analytics import (
    build_fuzzy_assessment,
    compare_to_training_cohort,
    get_training_profile,
)


# =========================================================
# 1. Paths and environment
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
load_dotenv(PROJECT_DIR / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "underwriting_copilot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MODEL_PATH = MODELS_DIR / "underwriting_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
POLICY_CONFIG_PATH = MODELS_DIR / "policy_config.json"
RISK_BAND_CONFIG_PATH = MODELS_DIR / "risk_band_config.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


# =========================================================
# 2. Load saved model artifacts
# =========================================================

required_files = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_CONFIG_PATH,
    POLICY_CONFIG_PATH,
    RISK_BAND_CONFIG_PATH,
    MODEL_METADATA_PATH,
]

for artifact_path in required_files:
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Required project artifact not found: {artifact_path}"
        )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with FEATURE_CONFIG_PATH.open("r", encoding="utf-8") as file:
    feature_config = json.load(file)
with POLICY_CONFIG_PATH.open("r", encoding="utf-8") as file:
    policy_config = json.load(file)
with RISK_BAND_CONFIG_PATH.open("r", encoding="utf-8") as file:
    risk_band_config = json.load(file)
with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
    model_metadata = json.load(file)

FEATURE_COLUMNS = feature_config["feature_columns"]
NUMERIC_FEATURES = feature_config["numeric_features"]
CATEGORICAL_FEATURES = feature_config["categorical_features"]

if len(FEATURE_COLUMNS) != 244:
    raise RuntimeError(
        f"Expected 244 model features, got {len(FEATURE_COLUMNS)}."
    )
if "TARGET" in FEATURE_COLUMNS or "SK_ID_CURR" in FEATURE_COLUMNS:
    raise RuntimeError("TARGET and SK_ID_CURR must not be model predictors.")


# =========================================================
# 3. FastAPI and routers
# =========================================================

app = FastAPI(
    title="AI Financial Underwriting Copilot API",
    version="0.2.0",
    description=(
        "Prototype decision-support API using PostgreSQL, a trained ML model, "
        "deterministic policy checks, fuzzy explanations, and historical-cohort comparison."
    ),
)
app.include_router(copilot_router)
app.include_router(new_applications_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# =========================================================
# 4. PostgreSQL connection
# =========================================================

def get_db_connection():
    if not DB_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="DB_PASSWORD is not configured in the project-root .env file.",
        )
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5,
        )
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "PostgreSQL is unavailable. Check the service and the .env connection settings."
            ),
        ) from exc


# =========================================================
# 5. General helpers
# =========================================================

def clean_numeric_value(value: Any) -> float | None:
    """Convert a finite numeric value to float, otherwise return None."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def assign_risk_band(pd_value: float) -> str:
    """Apply the existing project's risk-band assumptions."""
    if pd_value < risk_band_config["Low"]["max_pd"]:
        return "Low"
    if pd_value < risk_band_config["Medium"]["max_pd"]:
        return "Medium"
    if pd_value < risk_band_config["High"]["max_pd"]:
        return "High"
    return "Very High"


def count_non_missing_model_features(feature_payload: dict) -> dict[str, Any]:
    """Measure raw payload completeness; availability flags are values too."""
    present = sum(feature_payload.get(name) is not None for name in FEATURE_COLUMNS)
    total = len(FEATURE_COLUMNS)
    return {
        "available_model_feature_count": int(present),
        "model_feature_count": int(total),
        "non_missing_feature_percent": round(100 * present / total, 2) if total else 0.0,
    }


# =========================================================
# 6. Policy rules
# =========================================================

POLICY_RULES = [
    ("BUREAU_DEBT_TO_CREDIT", "High Bureau Debt Burden"),
    ("PREV_REFUSAL_RATE", "High Previous Refusal Rate"),
    ("INSTALLMENT_LATE_RATE", "High Installment Late Rate"),
    ("INSTALLMENT_UNDERPAYMENT_RATE", "High Installment Underpayment Rate"),
    ("POS_DPD_RATE", "POS/CASH Delinquency Rate"),
    ("POS_DPD_DEF_RATE", "POS/CASH Default-Delinquency Rate"),
    ("CC_MAX_UTILIZATION", "High Credit Card Utilization"),
    ("BB_DPD_RATE", "Bureau Balance Delinquency Rate"),
    ("BB_SERIOUS_DPD_RATE", "Bureau Serious Delinquency Rate"),
]


def evaluate_policy(feature_payload: dict) -> list[dict]:
    """Missing historical values are NOT_EVALUATED, not policy violations."""
    results = []
    for feature_name, rule_name in POLICY_RULES:
        value = clean_numeric_value(feature_payload.get(feature_name))

        if value is None:
            status = "NOT_EVALUATED"
            reason = "Required historical information is unavailable; this is not a policy failure."
        else:
            thresholds = policy_config.get(feature_name)
            if not thresholds:
                raise RuntimeError(
                    f"Missing policy configuration for {feature_name}."
                )
            warning = float(thresholds["warning"])
            critical = float(thresholds["critical"])
            if value >= critical:
                status = "CRITICAL"
            elif value >= warning:
                status = "WARNING"
            else:
                status = "PASSED"
            reason = f"{feature_name} = {value:.4f}"

        results.append({
            "rule": rule_name,
            "feature": feature_name,
            "value": value,
            "status": status,
            "reason": reason,
        })
    return results


def make_decision(predicted_pd: float, policy_results: list[dict]) -> dict:
    """Apply project decision rules. NOT_EVALUATED rules are not failures."""
    critical_count = sum(row["status"] == "CRITICAL" for row in policy_results)
    warning_count = sum(row["status"] == "WARNING" for row in policy_results)
    not_evaluated_count = sum(
        row["status"] == "NOT_EVALUATED" for row in policy_results
    )

    if critical_count > 0 or predicted_pd >= 0.20:
        decision = "REJECT"
        if critical_count > 0:
            reason = "One or more evaluated policy rules triggered a critical threshold."
        else:
            reason = "Predicted probability of default is at or above the project's 20% threshold."
    elif warning_count > 0 or predicted_pd >= 0.10:
        decision = "REVIEW"
        if warning_count > 0:
            reason = "One or more evaluated policy rules triggered a warning threshold."
        else:
            reason = "Predicted probability of default is within the project's 10% to 20% Review range."
    else:
        decision = "APPROVE"
        reason = (
            "Predicted probability of default is below the project's 10% Review threshold, "
            "and no evaluated policy rule triggered a warning or critical threshold."
        )

    return {
        "decision": decision,
        "critical_rule_count": int(critical_count),
        "warning_rule_count": int(warning_count),
        "not_evaluated_rule_count": int(not_evaluated_count),
        "decision_reason": reason,
    }


def apply_new_application_safeguards(
    decision_result: dict,
    schema_version: str,
) -> tuple[dict, bool, str]:
    """Report missing history without forcibly changing the decision to REVIEW."""
    is_new_application = schema_version == "new_application_v1"
    not_evaluated_count = int(decision_result.get("not_evaluated_rule_count", 0))

    if is_new_application and not_evaluated_count > 0:
        data_quality_status = "HISTORICAL_DATA_INCOMPLETE"
        note = (
            " Historical credit/repayment checks were NOT_EVALUATED because the source data "
            "are not connected. Missing history was not treated as a policy violation or used "
            "by itself to force REVIEW. The model score for this incomplete feature profile "
            "has not been validated for real lending use."
        )
        decision_result["decision_reason"] = (
            decision_result.get("decision_reason", "").strip() + note
        ).strip()
    elif not_evaluated_count > 0:
        data_quality_status = "SOME_POLICY_CHECKS_UNAVAILABLE"
    else:
        data_quality_status = "ALL_POLICY_CHECKS_EVALUATED"

    # This flag now means the rule-based outcome is REVIEW. Missing history
    # is separately communicated through data_quality_status and notes.
    requires_manual_review = decision_result.get("decision") == "REVIEW"
    return decision_result, requires_manual_review, data_quality_status


def json_safe_policy_value(value):
    """Return SQL NULL for missing values, otherwise a JSONB-compatible value."""
    return None if value is None else Json(value)


# =========================================================
# 7. Health and model info endpoints
# =========================================================

@app.get("/health")
def health():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return {
            "status": "ok",
            "database": "connected",
            "model": model_metadata.get("model_name", "HistGradientBoostingClassifier"),
            "model_version": model_metadata.get("model_version", "historical_features_v1"),
            "predictor_count": len(FEATURE_COLUMNS),
        }
    finally:
        connection.close()


@app.get("/model-info")
def model_info():
    """Explain the historical training data and known evaluation metrics."""
    return get_training_profile()


# =========================================================
# 8. Applicant lookup endpoint
# =========================================================

@app.get("/applicants/{applicant_id}")
def get_applicant(applicant_id: int):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.applicant_id, a.application_payload, f.feature_schema_version
                FROM applications AS a
                JOIN applicant_features AS f USING (applicant_id)
                WHERE a.applicant_id = %s
                """,
                (applicant_id,),
            )
            row = cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Applicant not found.")

        return {
            "applicant_id": row[0],
            "profile": row[1],
            "feature_schema_version": row[2],
        }
    finally:
        connection.close()


# =========================================================
# 9. Underwriting endpoint
# =========================================================

@app.post("/applicants/{applicant_id}/underwrite")
def underwrite_applicant(applicant_id: int):
    connection = get_db_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT feature_payload, feature_schema_version
                    FROM applicant_features
                    WHERE applicant_id = %s
                    """,
                    (applicant_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Applicant features were not found in the local database.",
                    )

                feature_payload, schema_version = row
                if not isinstance(feature_payload, dict):
                    raise HTTPException(
                        status_code=500,
                        detail="Stored feature payload is invalid.",
                    )

                missing_features = [
                    name for name in FEATURE_COLUMNS if name not in feature_payload
                ]
                if missing_features:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Stored feature payload is incomplete: {missing_features[:10]}",
                    )

                X = pd.DataFrame(
                    [{name: feature_payload[name] for name in FEATURE_COLUMNS}],
                    columns=FEATURE_COLUMNS,
                )

                try:
                    processed = preprocessor.transform(X)
                    processed = np.asarray(processed, dtype=np.float32)
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Feature preprocessing failed. Check that the stored payload "
                            "matches the saved feature configuration and preprocessor."
                        ),
                    ) from exc

                if not np.isfinite(processed).all():
                    raise HTTPException(
                        status_code=500,
                        detail="Preprocessed features contain invalid numeric values.",
                    )

                try:
                    predicted_pd = float(model.predict_proba(processed)[0, 1])
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail="The saved model could not score this feature vector.",
                    ) from exc

                if not math.isfinite(predicted_pd) or not 0 <= predicted_pd <= 1:
                    raise HTTPException(
                        status_code=500,
                        detail="The model returned an invalid probability value.",
                    )

                risk_band = assign_risk_band(predicted_pd)
                policy_results = evaluate_policy(feature_payload)
                decision_result = make_decision(predicted_pd, policy_results)
                (
                    decision_result,
                    requires_manual_review,
                    data_quality_status,
                ) = apply_new_application_safeguards(
                    decision_result=decision_result,
                    schema_version=schema_version,
                )

                # Supplementary analysis: these outputs explain the result but
                # explicitly do not override the trained model or policy logic.
                fuzzy_assessment = build_fuzzy_assessment(
                    feature_payload,
                    predicted_pd,
                )
                cohort_comparison = compare_to_training_cohort(feature_payload)
                feature_coverage = count_non_missing_model_features(feature_payload)

                decision_rationale = [
                    f"The trained model returned a predicted default probability of {predicted_pd * 100:.2f}%.",
                    f"The model's project risk band is {risk_band}.",
                    decision_result["decision_reason"],
                ]
                if decision_result["not_evaluated_rule_count"] > 0:
                    decision_rationale.append(
                        f"{decision_result['not_evaluated_rule_count']} historical policy checks were NOT_EVALUATED; they were not counted as failures."
                    )
                if fuzzy_assessment.get("available"):
                    decision_rationale.append(
                        f"The separate exploratory fuzzy indicator is {fuzzy_assessment['fuzzy_risk_band']} "
                        f"({fuzzy_assessment['fuzzy_risk_score']:.1f}/100); it does not change the decision."
                    )
                if cohort_comparison.get("available"):
                    if cohort_comparison.get("similar_cohort_observed_default_rate_percent") is not None:
                        decision_rationale.append(
                            "The exploratory historical comparison found "
                            f"{cohort_comparison['comparison_sample_size']} comparable records with an observed "
                            f"default rate of {cohort_comparison['similar_cohort_observed_default_rate_percent']:.2f}%; "
                            "this statistic does not change the decision."
                        )
                    else:
                        decision_rationale.append(
                            "Historical comparison was available, but there were too few comparable cases for a stable cohort default-rate estimate."
                        )

                # Save prediction.
                cursor.execute(
                    """
                    INSERT INTO predictions
                        (applicant_id, model_name, model_version, predicted_pd, risk_band)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING prediction_id
                    """,
                    (
                        applicant_id,
                        model_metadata.get("model_name", "HistGradientBoostingClassifier"),
                        model_metadata.get("model_version", "historical_features_v1"),
                        predicted_pd,
                        risk_band,
                    ),
                )
                prediction_id = cursor.fetchone()[0]

                # Save all policy results, including NOT_EVALUATED results.
                for rule in policy_results:
                    cursor.execute(
                        """
                        INSERT INTO policy_results
                            (applicant_id, prediction_id, rule_name, feature_name,
                             feature_value, status, reason)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            applicant_id,
                            prediction_id,
                            rule["rule"],
                            rule["feature"],
                            json_safe_policy_value(rule["value"]),
                            rule["status"],
                            rule["reason"],
                        ),
                    )

                # Save the final deterministic decision.
                cursor.execute(
                    """
                    INSERT INTO underwriting_decisions
                        (applicant_id, prediction_id, decision,
                         critical_rule_count, warning_rule_count,
                         not_evaluated_rule_count, decision_reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING decision_id
                    """,
                    (
                        applicant_id,
                        prediction_id,
                        decision_result["decision"],
                        decision_result["critical_rule_count"],
                        decision_result["warning_rule_count"],
                        decision_result["not_evaluated_rule_count"],
                        decision_result["decision_reason"],
                    ),
                )
                decision_id = cursor.fetchone()[0]

                audit_details = {
                    "prediction_id": prediction_id,
                    "decision_id": decision_id,
                    "model_version": model_metadata.get("model_version", "historical_features_v1"),
                    "risk_band": risk_band,
                    "decision": decision_result["decision"],
                    "decision_reason": decision_result["decision_reason"],
                    "feature_schema_version": schema_version,
                    "requires_manual_review": requires_manual_review,
                    "data_quality_status": data_quality_status,
                    "feature_coverage": feature_coverage,
                    "fuzzy_assessment": fuzzy_assessment,
                    "training_cohort_comparison": cohort_comparison,
                    "decision_rationale": decision_rationale,
                    "critical_rule_count": decision_result["critical_rule_count"],
                    "warning_rule_count": decision_result["warning_rule_count"],
                    "not_evaluated_rule_count": decision_result["not_evaluated_rule_count"],
                }
                cursor.execute(
                    """
                    INSERT INTO audit_log (applicant_id, event_type, event_details)
                    VALUES (%s, %s, %s)
                    """,
                    (applicant_id, "UNDERWRITING_ASSESSMENT", Json(audit_details)),
                )

        # Return only after successful transaction commit.
        return {
            "applicant_id": applicant_id,
            "prediction_id": prediction_id,
            "decision_id": decision_id,
            "predicted_pd": predicted_pd,
            "predicted_pd_percent": round(predicted_pd * 100, 4),
            "risk_band": risk_band,
            "decision": decision_result["decision"],
            "decision_reason": decision_result["decision_reason"],
            "critical_rule_count": decision_result["critical_rule_count"],
            "warning_rule_count": decision_result["warning_rule_count"],
            "not_evaluated_rule_count": decision_result["not_evaluated_rule_count"],
            "requires_manual_review": requires_manual_review,
            "data_quality_status": data_quality_status,
            "feature_coverage": feature_coverage,
            "fuzzy_assessment": fuzzy_assessment,
            "training_cohort_comparison": cohort_comparison,
            "decision_rationale": decision_rationale,
            "model_version": model_metadata.get("model_version", "historical_features_v1"),
            "notice": (
                "Prototype decision support only. Missing historical checks are not treated as violations. "
                "Fuzzy indicators and similar-applicant rates are exploratory and do not change the saved "
                "model/policy decision. Predictions for new applicants with incomplete historical features "
                "have not been validated for real lending use."
            ),
        }

    except HTTPException:
        raise
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Database operation failed. Check the PostgreSQL schema and API logs.",
        ) from exc
    finally:
        connection.close()
