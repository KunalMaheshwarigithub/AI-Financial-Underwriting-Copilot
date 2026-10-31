# Architecture

## System context

The AI Financial Underwriting Copilot is a local, portfolio-oriented decision-support prototype. It combines a Streamlit user interface, a FastAPI service, PostgreSQL persistence, an offline-trained scikit-learn model, deterministic policy checks, experimental risk explanations, and a retrieval-augmented LLM assistant.

## Component responsibilities

| Component | Responsibility |
|---|---|
| Streamlit frontend | Application intake, applicant profile, underwriting results, and Copilot chat |
| FastAPI backend | Request validation, feature retrieval, orchestration, model inference, policy evaluation, and persistence |
| PostgreSQL | Applications, model-compatible applicant features, predictions, policy results, underwriting decisions, and audit events |
| Model artifacts | Trained classifier, preprocessing pipeline, feature configuration, model metadata, risk-band configuration, and policy configuration |
| Risk analytics | Additional fuzzy indicators and historical cohort analysis, subject to input availability and validation |
| Knowledge base | Project documentation and policy reference documents for retrieval |
| Cloudflare Workers AI | Hosted LLM inference for the natural-language Copilot |

## Underwriting request flow

1. The frontend sends an applicant ID to the FastAPI underwriting endpoint.
2. The API retrieves the applicant's feature payload and schema version from PostgreSQL.
3. It orders the feature values using the saved 244-feature configuration.
4. The saved preprocessing pipeline transforms the values.
5. The classifier returns a probability estimate; the configured project risk bands map that value to a label.
6. The policy engine evaluates available deterministic rules. A required value that is absent is marked `NOT_EVALUATED`, not treated as a failed rule.
7. Additional fuzzy and historical-cohort explanations may be produced when their input data are available.
8. Prediction, policy outcomes, decision output, and audit event are persisted.
9. The API returns the assessment payload to Streamlit.

## Copilot request flow

1. The user submits a natural-language question.
2. The TF-IDF retriever ranks knowledge-base documents using cosine similarity.
3. When an applicant ID is supplied, the API retrieves selected applicant fields and the latest saved prediction and decision.
4. Retrieved text and selected applicant context are sent to Cloudflare Workers AI.
5. The response and retrieved-document metadata are returned to Streamlit.

## Current API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Check API and PostgreSQL connectivity |
| POST | `/applications` | Validate and store a new application |
| GET | `/applicants/{applicant_id}` | Retrieve an applicant profile |
| POST | `/applicants/{applicant_id}/underwrite` | Run model inference and underwriting workflow |
| POST | `/copilot/ask` | Retrieve references and generate a Copilot answer |
| GET | `/model-info` | Expose model/training metadata, if enabled in the deployed API version |

The interactive OpenAPI page is available at `http://127.0.0.1:8000/docs` when the local API is running.

## Data model (high level)

The PostgreSQL schema includes the following core entities:

- `applications`: applicant identifier, optional historical target, and application payload.
- `applicant_features`: model feature payload and feature schema version.
- `predictions`: model name/version, predicted probability, and risk band.
- `policy_results`: individual rule outcomes and supporting values.
- `underwriting_decisions`: project decision, rule counts, and rationale.
- `audit_log`: application and assessment event history.

The exact columns and constraints are defined in `database/schema.sql`.

## Data boundaries and trust

- `TARGET` is an outcome label for historical supervised training; it must never be used as a predictor.
- Applicant identifiers are database keys, not predictors.
- New applications have no known `TARGET` at intake.
- Missing credit-bureau or repayment history must remain distinguishable from a negative historical outcome.
- Only a small, explicitly selected subset of applicant context should be sent to the LLM.
- Secrets and applicant-level datasets must remain outside the public repository.

## Design limitations

The current form collects fewer fields than the historical model's 244-feature input schema. A preprocessing pipeline may technically produce an inference from imputed values, but that does not prove the prediction is calibrated or reliable for the new-applicant profile. Fuzzy indicators and similarity-based comparisons are supplementary explanations and must not be presented as independently validated credit scores.

Production concerns such as robust authentication, role-based access, tenant isolation, encryption, secrets management, monitoring, deployment, model registry, drift monitoring, and formal governance are future work unless implemented and tested separately.
