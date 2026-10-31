"""Validated new-applicant intake endpoint for the underwriting prototype."""
from __future__ import annotations

import math
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from psycopg2.extras import Json

router = APIRouter(tags=["New Applications"])
FEATURE_SCHEMA_VERSION = "new_application_v1"
EXPECTED_FEATURE_COUNT = 244


# ============================================================
# 1. VALIDATED APPLICATION REQUEST
# ============================================================

class NewApplicationRequest(BaseModel):
    """Fields collected by the current new-application intake form."""

    application_reference: str | None = Field(default=None, max_length=80)
    annual_income: float = Field(gt=0, allow_inf_nan=False)
    credit_amount: float = Field(gt=0, allow_inf_nan=False)
    annuity_amount: float = Field(gt=0, allow_inf_nan=False)
    goods_price: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    age_years: int = Field(ge=18, le=100)
    employment_years: float = Field(
        ge=0,
        le=80,
        allow_inf_nan=False,
    )
    children_count: int = Field(default=0, ge=0, le=30)

    contract_type: Literal[
        "Cash loans",
        "Revolving loans",
    ] = "Cash loans"

    income_type: Literal[
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ] = "Working"

    education_type: Literal[
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ] = "Secondary / secondary special"

    housing_type: Literal[
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ] = "House / apartment"

    occupation_type: str | None = Field(default=None, max_length=100)
    organization_type: str | None = Field(default=None, max_length=100)

    owns_car: Literal["Y", "N"] | None = None
    owns_realty: Literal["Y", "N"] | None = None

    type_suite: str | None = Field(default=None, max_length=50)

    weekday_application: Literal[
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ] | None = None


# ============================================================
# 2. FEATURE GENERATION
# ============================================================

def generate_new_application_features(
    request: NewApplicationRequest,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    """Build the model's expected feature schema without fabricating history.

    The feature payload always has all 244 model keys. Values not supplied by
    the intake form or an authorized source remain None. The saved preprocessor
    handles missing values according to its configured behavior.
    """

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Expected {EXPECTED_FEATURE_COUNT} model features, "
                f"but the loaded configuration contains "
                f"{len(feature_columns)}."
            ),
        )

    feature_set = set(feature_columns)
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)

    if len(feature_set) != len(feature_columns):
        raise HTTPException(
            status_code=500,
            detail="Feature configuration contains duplicate feature names.",
        )

    if numeric_set & categorical_set:
        raise HTTPException(
            status_code=500,
            detail="A feature cannot be both numeric and categorical.",
        )

    if numeric_set | categorical_set != feature_set:
        raise HTTPException(
            status_code=500,
            detail=(
                "Numeric and categorical feature lists must cover "
                "the complete model feature configuration."
            ),
        )

    # Initialize every expected model feature.
    # Unavailable historical values deliberately remain None.
    features: dict[str, int | float | str | None] = {
        name: None for name in feature_columns
    }

    def set_feature(
        name: str,
        value: int | float | str | None,
    ) -> None:
        """Set only features belonging to the saved model schema."""
        if name in features:
            features[name] = value

    # --------------------------------------------------------
    # Applicant-provided values
    # --------------------------------------------------------

    set_feature("AMT_INCOME_TOTAL", request.annual_income)
    set_feature("AMT_CREDIT", request.credit_amount)
    set_feature("AMT_ANNUITY", request.annuity_amount)
    set_feature("AMT_GOODS_PRICE", request.goods_price)

    set_feature("AGE_YEARS", request.age_years)
    set_feature("EMPLOYMENT_YEARS", request.employment_years)
    set_feature("CNT_CHILDREN", request.children_count)

    set_feature("NAME_CONTRACT_TYPE", request.contract_type)
    set_feature("NAME_INCOME_TYPE", request.income_type)
    set_feature("NAME_EDUCATION_TYPE", request.education_type)
    set_feature("NAME_HOUSING_TYPE", request.housing_type)
    set_feature("OCCUPATION_TYPE", request.occupation_type)
    set_feature("ORGANIZATION_TYPE", request.organization_type)
    set_feature("FLAG_OWN_CAR", request.owns_car)
    set_feature("FLAG_OWN_REALTY", request.owns_realty)
    set_feature("NAME_TYPE_SUITE", request.type_suite)
    set_feature(
        "WEEKDAY_APPR_PROCESS_START",
        request.weekday_application,
    )

    # Home Credit expresses age as negative days before application.
    # This is an approximation derived from the age entered on the form.
    set_feature(
        "DAYS_BIRTH",
        -round(request.age_years * 365.25),
    )

    # Home Credit uses 365243 as a sentinel for unemployed applicants.
    # For other income types, approximate employment duration in days.
    if request.income_type == "Unemployed":
        set_feature("DAYS_EMPLOYED", 365243)
    else:
        set_feature(
            "DAYS_EMPLOYED",
            -round(request.employment_years * 365.25),
        )

    # --------------------------------------------------------
    # Derived financial features
    # --------------------------------------------------------

    set_feature(
        "CREDIT_TO_INCOME",
        request.credit_amount / request.annual_income,
    )

    set_feature(
        "ANNUITY_TO_INCOME",
        request.annuity_amount / request.annual_income,
    )

    if request.goods_price is not None:
        set_feature(
            "CREDIT_TO_GOODS_PRICE",
            request.credit_amount / request.goods_price,
        )

    set_feature(
        "APPROX_TERM_MONTHS",
        request.credit_amount / request.annuity_amount,
    )

    # --------------------------------------------------------
    # External credit-source scores are not entered on the form
    # --------------------------------------------------------

    for name in (
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
    ):
        set_feature(name, None)

    set_feature("EXT_SOURCE_AVAILABLE", 0)
    set_feature("EXT_SOURCE_1_MISSING", 1)
    set_feature("EXT_SOURCE_2_MISSING", 1)
    set_feature("EXT_SOURCE_3_MISSING", 1)

    # --------------------------------------------------------
    # Credit-bureau enquiries are not connected to this intake
    # --------------------------------------------------------

    for name in (
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "CREDIT_BUREAU_ENQUIRIES_TOTAL",
    ):
        set_feature(name, None)

    set_feature("CREDIT_BUREAU_ENQUIRIES_AVAILABLE", 0)
    set_feature("CREDIT_BUREAU_ENQUIRIES_MISSING", 1)

    # --------------------------------------------------------
    # Historical data availability flags
    # --------------------------------------------------------

    # These indicate that the prototype has not connected these data sources.
    # They do NOT mean that the person has never borrowed or repaid a loan.
    for name in (
        "BUREAU_HISTORY_AVAILABLE",
        "INSTALLMENT_HISTORY_AVAILABLE",
        "POS_HISTORY_AVAILABLE",
        "CC_HISTORY_AVAILABLE",
        "BUREAU_BALANCE_HISTORY_AVAILABLE",
    ):
        set_feature(name, 0)

    # --------------------------------------------------------
    # Payload validation
    # --------------------------------------------------------

    missing_keys = [
        name
        for name in feature_columns
        if name not in features
    ]

    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=(
                "Generated feature payload is incomplete: "
                f"{missing_keys[:10]}"
            ),
        )

    for name in numeric_features:
        value = features[name]

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid numeric feature generated: {name}",
            )

        if not math.isfinite(float(value)):
            raise HTTPException(
                status_code=500,
                detail=f"Non-finite numeric feature generated: {name}",
            )

    for name in categorical_features:
        value = features[name]

        if value is not None and not isinstance(value, str):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid categorical feature generated: {name}",
            )

    return features


# ============================================================
# 3. REGISTER A NEW APPLICATION
# ============================================================

@router.post(
    "/applications",
    status_code=201,
    tags=["Applications"],
)
def create_application(request: NewApplicationRequest):
    """Persist an application and feature payload; scoring is a separate call."""

    if request.employment_years > request.age_years:
        raise HTTPException(
            status_code=422,
            detail=(
                "Employment duration cannot exceed the applicant's age."
            ),
        )

    # Lazy import avoids a circular import when api.main registers this router.
    import api.main as main_module

    features = generate_new_application_features(
        request=request,
        feature_columns=main_module.FEATURE_COLUMNS,
        numeric_features=main_module.NUMERIC_FEATURES,
        categorical_features=main_module.CATEGORICAL_FEATURES,
    )

    application_reference = (
        request.application_reference.strip()
        if request.application_reference
        else None
    )

    if application_reference == "":
        application_reference = None

    # Keep original inputs and commonly used derived values for profile
    # display. Model inference uses the separate 244-column feature payload.
    application_payload = {
        "APPLICATION_REFERENCE": application_reference,
        "NAME_CONTRACT_TYPE": request.contract_type,
        "NAME_INCOME_TYPE": request.income_type,
        "NAME_EDUCATION_TYPE": request.education_type,
        "NAME_HOUSING_TYPE": request.housing_type,
        "OCCUPATION_TYPE": request.occupation_type,
        "ORGANIZATION_TYPE": request.organization_type,
        "FLAG_OWN_CAR": request.owns_car,
        "FLAG_OWN_REALTY": request.owns_realty,
        "NAME_TYPE_SUITE": request.type_suite,
        "WEEKDAY_APPR_PROCESS_START": request.weekday_application,
        "CNT_CHILDREN": request.children_count,
        "AMT_INCOME_TOTAL": request.annual_income,
        "AMT_CREDIT": request.credit_amount,
        "AMT_ANNUITY": request.annuity_amount,
        "AMT_GOODS_PRICE": request.goods_price,
        "AGE_YEARS": request.age_years,
        "EMPLOYMENT_YEARS": request.employment_years,
        "CREDIT_TO_INCOME": features.get("CREDIT_TO_INCOME"),
        "ANNUITY_TO_INCOME": features.get("ANNUITY_TO_INCOME"),
        "CREDIT_TO_GOODS_PRICE": features.get("CREDIT_TO_GOODS_PRICE"),
        "APPROX_TERM_MONTHS": features.get("APPROX_TERM_MONTHS"),
    }

    non_missing_feature_count = sum(
        value is not None
        for value in features.values()
    )

    feature_coverage_percent = round(
        100 * non_missing_feature_count / len(features),
        2,
    )

    connection = main_module.get_db_connection()

    try:
        with connection:
            with connection.cursor() as cursor:
                # nextval() atomically assigns a unique applicant ID.
                cursor.execute(
                    "SELECT nextval('public.new_application_applicant_id_seq')"
                )
                applicant_id = cursor.fetchone()[0]

                # TARGET remains SQL NULL for a new applicant because
                # the actual repayment outcome is not known.
                cursor.execute(
                    """
                    INSERT INTO applications
                        (
                            applicant_id,
                            target,
                            application_payload
                        )
                    VALUES (%s, NULL, %s)
                    """,
                    (
                        applicant_id,
                        Json(application_payload),
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO applicant_features
                        (
                            applicant_id,
                            feature_payload,
                            feature_schema_version
                        )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        applicant_id,
                        Json(features),
                        FEATURE_SCHEMA_VERSION,
                    ),
                )

                audit_details = {
                    "application_reference": application_reference,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "model_feature_count": len(features),
                    "non_missing_model_feature_count": (
                        non_missing_feature_count
                    ),
                    "feature_coverage_percent": feature_coverage_percent,
                    "historical_data_connected": False,
                    "underwriting_status": "NOT_RUN",
                }

                cursor.execute(
                    """
                    INSERT INTO audit_log
                        (
                            applicant_id,
                            event_type,
                            event_details
                        )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        applicant_id,
                        "NEW_APPLICATION_CREATED",
                        Json(audit_details),
                    ),
                )

        return {
            "applicant_id": applicant_id,
            "status": "APPLICATION_CREATED",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_feature_count": len(features),
            "non_missing_model_feature_count": (
                non_missing_feature_count
            ),
            "feature_coverage_percent": feature_coverage_percent,
            "historical_data_connected": False,
            "underwriting_status": "NOT_RUN",
            "next_step": (
                f"POST /applicants/{applicant_id}/underwrite"
            ),
            "notice": (
                "Prototype intake only. Historical bureau and repayment data "
                "are not connected. Missing features remain unavailable and "
                "may reduce prediction reliability. The subsequent assessment "
                "provides separate model, fuzzy-profile, and historical-cohort "
                "outputs where reference data are available; those additional "
                "outputs do not constitute a validated lending decision."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        # The transaction rolls back if any database insert fails.
        # The full exception is chained for server logs, not exposed to users.
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save the new application. "
                "Check the API terminal and PostgreSQL schema."
            ),
        ) from exc

    finally:
        connection.close()