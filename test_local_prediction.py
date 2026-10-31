# test_local_prediction.py
# Test local ML inference using PostgreSQL + exported Kaggle artifacts

import json
import getpass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODELS_DIR / "underwriting_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


# =========================================================
# 2. PostgreSQL configuration
# =========================================================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "underwriting_copilot"
DB_USER = "postgres"

APPLICANT_ID = 100020


# =========================================================
# 3. Verify artifact files
# =========================================================

required_files = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_METADATA_PATH
]

for file_path in required_files:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {file_path}"
        )

print("All required model artifacts found.")


# =========================================================
# 4. Load model, preprocessor and feature configuration
# =========================================================

print("\nLoading saved Kaggle artifacts...")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with FEATURE_CONFIG_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    feature_config = json.load(file)

with MODEL_METADATA_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    model_metadata = json.load(file)

feature_columns = feature_config["feature_columns"]
numeric_features = feature_config["numeric_features"]
categorical_features = feature_config["categorical_features"]

if len(feature_columns) != 244:
    raise ValueError(
        f"Expected 244 model features; got {len(feature_columns)}."
    )

if "TARGET" in feature_columns or "SK_ID_CURR" in feature_columns:
    raise ValueError(
        "TARGET or SK_ID_CURR incorrectly appears as a model predictor."
    )

print("Model loaded:", model_metadata["model_name"])
print("Model version:", model_metadata["model_version"])
print("Predictor count:", len(feature_columns))


# =========================================================
# 5. Connect to PostgreSQL
# =========================================================

password = getpass.getpass(
    "\nEnter PostgreSQL password (input hidden): "
)

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        connect_timeout=10
    )
except psycopg2.Error as exc:
    raise RuntimeError(
        "Could not connect to PostgreSQL. Check that the server "
        "is running and that the database name and password are correct."
    ) from exc


# =========================================================
# 6. Retrieve the applicant's complete feature vector
# =========================================================

try:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                a.target,
                f.feature_payload,
                f.feature_schema_version
            FROM applications AS a
            JOIN applicant_features AS f
                ON a.applicant_id = f.applicant_id
            WHERE a.applicant_id = %s
            """,
            (APPLICANT_ID,)
        )

        result = cursor.fetchone()

finally:
    conn.close()


if result is None:
    raise ValueError(
        f"Applicant {APPLICANT_ID} was not found in PostgreSQL."
    )

actual_target, feature_payload, schema_version = result

if not isinstance(feature_payload, dict):
    raise TypeError(
        "PostgreSQL feature_payload was not returned as a JSON object."
    )

missing_features = [
    col for col in feature_columns
    if col not in feature_payload
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features[:20]}"
    )

if len(feature_payload) != len(feature_columns):
    print(
        "Warning: JSON feature count differs from the configured "
        "predictor count. The required 244 features will still be used."
    )

print("\nApplicant retrieved successfully.")
print("Applicant ID:", APPLICANT_ID)
print("Stored historical TARGET:", actual_target)
print("Feature schema version:", schema_version)
print("Features available:", len(feature_payload))


# =========================================================
# 7. Restore the original model feature order
# =========================================================

X_applicant = pd.DataFrame(
    [
        {
            col: feature_payload[col]
            for col in feature_columns
        }
    ],
    columns=feature_columns
)

if X_applicant.shape != (1, 244):
    raise ValueError(
        f"Unexpected input shape: {X_applicant.shape}"
    )

# Ensure numeric/categorical columns follow the saved configuration.
if set(numeric_features).intersection(categorical_features):
    raise ValueError(
        "Numeric and categorical feature lists overlap."
    )

if set(numeric_features).union(categorical_features) != set(feature_columns):
    raise ValueError(
        "Feature configuration does not match the model feature list."
    )


# =========================================================
# 8. Apply the saved Kaggle preprocessor
# =========================================================

X_processed = preprocessor.transform(X_applicant)

X_processed = np.asarray(
    X_processed,
    dtype=np.float32
)

expected_processed_features = (
    feature_config["processed_feature_count"]
)

if X_processed.shape != (1, expected_processed_features):
    raise ValueError(
        f"Unexpected processed shape: {X_processed.shape}"
    )

if not np.isfinite(X_processed).all():
    raise ValueError(
        "Processed features contain NaN or infinite values."
    )

print("\nPreprocessing successful.")
print("Processed input shape:", X_processed.shape)


# =========================================================
# 9. Calculate Probability of Default
# =========================================================

predicted_pd = float(
    model.predict_proba(X_processed)[0, 1]
)


# =========================================================
# 10. Assign the project-defined risk band
# =========================================================

risk_band_config_path = MODELS_DIR / "risk_band_config.json"

with risk_band_config_path.open(
    "r",
    encoding="utf-8"
) as file:
    risk_band_config = json.load(file)


def assign_risk_band_local(pd_value):

    if pd_value < risk_band_config["Low"]["max_pd"]:
        return "Low"

    if pd_value < risk_band_config["Medium"]["max_pd"]:
        return "Medium"

    if pd_value < risk_band_config["High"]["max_pd"]:
        return "High"

    return "Very High"


risk_band = assign_risk_band_local(predicted_pd)


# =========================================================
# 11. Compare against the Kaggle reference prediction
# =========================================================

kaggle_reference_pd = 0.6029776665857008

difference = abs(
    predicted_pd - kaggle_reference_pd
)

print("\n" + "=" * 55)
print("LOCAL UNDERWRITING MODEL TEST")
print("=" * 55)

print("Applicant ID:", APPLICANT_ID)
print("Actual historical TARGET:", actual_target)
print(f"Local predicted PD:       {predicted_pd:.10f}")
print(f"Local predicted PD (%):   {predicted_pd * 100:.4f}%")
print("Local risk band:", risk_band)

print("\nKaggle reference PD:", f"{kaggle_reference_pd:.10f}")
print("Absolute difference:", f"{difference:.10f}")

if difference < 1e-5:
    print("\nSUCCESS: Local prediction matches the Kaggle reference.")
else:
    print(
        "\nCHECK REQUIRED: Local prediction differs from the "
        "Kaggle reference. We should investigate before proceeding."
    )