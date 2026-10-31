"""Load the Home Credit applicant feature store into local PostgreSQL.

Place this file in the extracted project root, alongside the models/,
knowledge_base/, and database/ folders. It is safe to rerun: applicant
rows, feature rows, and RAG documents use ON CONFLICT upserts.
"""

from __future__ import annotations

import getpass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_PATH = MODELS_DIR / "applicant_feature_store.parquet"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
RAG_PATH = BASE_DIR / "knowledge_base" / "rag_knowledge_base.json"

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "underwriting_copilot"
DB_USER = "postgres"

BATCH_SIZE = 1000
MODEL_FEATURE_SCHEMA_VERSION = "historical_features_v1"

# Compact profile shown in the application.
# Full model predictors are stored separately in JSONB.
PROFILE_FIELDS = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "EXT_SOURCE_MEAN",
]


def clean_value(value):
    """Convert pandas/NumPy values into strict JSON-safe values."""

    if value is None or value is pd.NA:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, np.str_):
        return str(value)

    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass

    return value


def strict_json_dumps(value):
    """Serialize JSON without allowing NaN or infinity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False
    )


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:

    # -----------------------------------------------------
    # 1. Check required files
    # -----------------------------------------------------

    for path in (
        FEATURE_STORE_PATH,
        FEATURE_CONFIG_PATH
    ):
        if not path.is_file():
            fail(
                f"File not found: {path}\n"
                "Place this script beside the models/ folder "
                "in your extracted project directory."
            )

    with FEATURE_CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        feature_config = json.load(f)

    feature_columns = feature_config.get(
        "feature_columns",
        []
    )

    if len(feature_columns) != 244:
        fail(
            "Expected 244 model feature names in "
            f"feature_config.json; found {len(feature_columns)}."
        )

    if (
        "TARGET" in feature_columns
        or "SK_ID_CURR" in feature_columns
    ):
        fail(
            "feature_config.json incorrectly includes "
            "TARGET or SK_ID_CURR as a predictor."
        )

    # -----------------------------------------------------
    # 2. Read and validate the Parquet feature store
    # -----------------------------------------------------

    print("Reading applicant feature store ...")

    df = pd.read_parquet(
        FEATURE_STORE_PATH
    )

    required_columns = {
        "SK_ID_CURR",
        "TARGET",
        *feature_columns
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        fail(
            "Feature store is missing columns: "
            f"{missing_columns[:20]}"
        )

    if df["SK_ID_CURR"].isna().any():
        fail(
            "Feature store contains missing applicant IDs."
        )

    if df["SK_ID_CURR"].duplicated().any():
        fail(
            "Feature store contains duplicate applicant IDs."
        )

    if len(df) != 307511:
        fail(
            f"Expected 307,511 applicants; found {len(df):,}."
        )

    print(f"Applicants: {len(df):,}")
    print(f"Predictor features: {len(feature_columns)}")
    print(f"Database: {DB_NAME} on {DB_HOST}:{DB_PORT}")
    print(
        "Enter the password you created for the "
        "PostgreSQL 'postgres' user."
    )

    password = getpass.getpass(
        "PostgreSQL password (input hidden): "
    )

    # -----------------------------------------------------
    # 3. Connect to PostgreSQL
    # -----------------------------------------------------

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=password,
            connect_timeout=10,
        )

    except Exception as exc:
        fail(
            f"Could not connect to PostgreSQL: {exc}\n"
            "Verify PostgreSQL is running, the database is named "
            "'underwriting_copilot', and the password is correct."
        )

    # -----------------------------------------------------
    # 4. Prepare SQL upserts
    # -----------------------------------------------------

    app_sql = """
        INSERT INTO applications
            (applicant_id, target, application_payload)
        VALUES %s
        ON CONFLICT (applicant_id) DO UPDATE SET
            target = EXCLUDED.target,
            application_payload = EXCLUDED.application_payload
    """

    features_sql = """
        INSERT INTO applicant_features
            (applicant_id, feature_payload, feature_schema_version)
        VALUES %s
        ON CONFLICT (applicant_id) DO UPDATE SET
            feature_payload = EXCLUDED.feature_payload,
            feature_schema_version = EXCLUDED.feature_schema_version,
            feature_generated_at = CURRENT_TIMESTAMP
    """

    try:

        # -------------------------------------------------
        # 5. Load applicants in batches
        # -------------------------------------------------

        with conn:
            with conn.cursor() as cur:

                # Fail early if the correct schema isn't installed.
                cur.execute("""
                    SELECT
                        to_regclass('public.applications'),
                        to_regclass('public.applicant_features')
                """)

                tables = cur.fetchone()

                if not all(tables):
                    fail(
                        "Required tables were not found. Apply the "
                        "latest database/schema.sql to the "
                        "underwriting_copilot database first."
                    )

                total = len(df)

                for start in range(
                    0,
                    total,
                    BATCH_SIZE
                ):

                    batch = df.iloc[
                        start:start + BATCH_SIZE
                    ]

                    application_rows = []
                    feature_rows = []

                    for record in batch.to_dict(
                        orient="records"
                    ):

                        applicant_id = int(
                            record["SK_ID_CURR"]
                        )

                        target_value = clean_value(
                            record.get("TARGET")
                        )

                        if target_value is not None:
                            target_value = int(target_value)

                        # Profile data for application lookup.
                        profile = {
                            key: clean_value(record.get(key))
                            for key in PROFILE_FIELDS
                            if key in record
                        }

                        # Full 244-feature model input.
                        features = {
                            key: clean_value(record.get(key))
                            for key in feature_columns
                        }

                        application_rows.append(
                            (
                                applicant_id,
                                target_value,
                                Json(
                                    profile,
                                    dumps=strict_json_dumps
                                ),
                            )
                        )

                        feature_rows.append(
                            (
                                applicant_id,
                                Json(
                                    features,
                                    dumps=strict_json_dumps
                                ),
                                MODEL_FEATURE_SCHEMA_VERSION,
                            )
                        )

                    execute_values(
                        cur,
                        app_sql,
                        application_rows,
                        page_size=500
                    )

                    execute_values(
                        cur,
                        features_sql,
                        feature_rows,
                        page_size=500
                    )

                    # Commit each batch so progress is preserved.
                    conn.commit()

                    completed = min(
                        start + len(batch),
                        total
                    )

                    if (
                        completed == total
                        or completed % 10000 < BATCH_SIZE
                    ):
                        print(
                            f"Loaded/updated "
                            f"{completed:,} / {total:,} applicants"
                        )

                # -----------------------------------------
                # 6. Load RAG documents
                # -----------------------------------------

                if RAG_PATH.is_file():

                    with RAG_PATH.open(
                        "r",
                        encoding="utf-8"
                    ) as f:
                        rag_docs = json.load(f)

                    rag_sql = """
                        INSERT INTO rag_documents
                            (document_id, title, category, content)
                        VALUES %s
                        ON CONFLICT (document_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            category = EXCLUDED.category,
                            content = EXCLUDED.content
                    """

                    rag_rows = [
                        (
                            doc["doc_id"],
                            doc["title"],
                            doc.get("category"),
                            doc["content"],
                        )
                        for doc in rag_docs
                    ]

                    if rag_rows:
                        execute_values(
                            cur,
                            rag_sql,
                            rag_rows,
                            page_size=100
                        )

                    conn.commit()

                    print(
                        f"Loaded/updated "
                        f"{len(rag_rows)} RAG documents"
                    )

                else:
                    print(
                        f"RAG file not found; skipped: {RAG_PATH}"
                    )

                # -----------------------------------------
                # 7. Verify database counts
                # -----------------------------------------

                cur.execute(
                    "SELECT COUNT(*) FROM applications"
                )
                app_count = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM applicant_features"
                )
                feature_count = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM rag_documents"
                )
                rag_count = cur.fetchone()[0]

                cur.execute("""
                    SELECT target, COUNT(*)
                    FROM applications
                    GROUP BY target
                    ORDER BY target
                """)
                target_counts = cur.fetchall()

        # -------------------------------------------------
        # 8. Final report
        # -------------------------------------------------

        print("\nDATABASE LOAD COMPLETE")

        print(
            f"Applications rows:       {app_count:,}"
        )

        print(
            f"Applicant feature rows:  {feature_count:,}"
        )

        print(
            f"RAG documents:            {rag_count:,}"
        )

        print("Target counts:")

        for target, count in target_counts:
            print(
                f"  TARGET={target}: {count:,}"
            )

        if (
            app_count < total
            or feature_count < total
        ):
            print(
                "WARNING: table row counts are below "
                "the expected applicant count."
            )
        else:
            print(
                "Verified: all expected applicants are present "
                "in both core tables."
            )

    except SystemExit:
        conn.rollback()
        raise

    except Exception as exc:
        conn.rollback()

        fail(
            f"Import failed: {exc}\n"
            "The loader uses upserts, so after fixing the "
            "problem you can rerun it safely."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()