
-- ======================================================
-- AI FINANCIAL UNDERWRITING COPILOT
-- PostgreSQL Schema v2
-- ======================================================

-- 1. Applications and historical target labels
CREATE TABLE IF NOT EXISTS applications (
    applicant_id BIGINT PRIMARY KEY,
    target SMALLINT CHECK (target IN (0, 1) OR target IS NULL),
    application_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Complete model feature vector: all 244 predictors
CREATE TABLE IF NOT EXISTS applicant_features (
    applicant_id BIGINT PRIMARY KEY
        REFERENCES applications(applicant_id),
    feature_payload JSONB NOT NULL,
    feature_schema_version VARCHAR(100) NOT NULL
        DEFAULT 'historical_features_v1',
    feature_generated_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);

-- 3. Historical-source availability and summaries
CREATE TABLE IF NOT EXISTS historical_data_summary (
    id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    source_name VARCHAR(100) NOT NULL,
    history_available BOOLEAN NOT NULL,
    record_count BIGINT,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 4. Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    model_name VARCHAR(150) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    predicted_pd DOUBLE PRECISION NOT NULL
        CHECK (predicted_pd >= 0 AND predicted_pd <= 1),
    risk_band VARCHAR(30) NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);

-- 5. Policy results
CREATE TABLE IF NOT EXISTS policy_results (
    policy_result_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    rule_name VARCHAR(150) NOT NULL,
    feature_name VARCHAR(150),
    feature_value JSONB,
    status VARCHAR(30) NOT NULL
        CHECK (status IN ('PASSED', 'WARNING',
                          'CRITICAL', 'NOT_EVALUATED')),
    reason TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 6. Underwriting decisions
CREATE TABLE IF NOT EXISTS underwriting_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    decision VARCHAR(30) NOT NULL
        CHECK (decision IN ('APPROVE', 'REVIEW', 'REJECT')),
    critical_rule_count INTEGER NOT NULL DEFAULT 0,
    warning_rule_count INTEGER NOT NULL DEFAULT 0,
    not_evaluated_rule_count INTEGER NOT NULL DEFAULT 0,
    decision_reason TEXT,
    decision_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 7. Individual SHAP explanations
CREATE TABLE IF NOT EXISTS shap_explanations (
    shap_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    feature_name VARCHAR(150) NOT NULL,
    business_description TEXT,
    applicant_value JSONB,
    shap_value DOUBLE PRECISION NOT NULL,
    direction VARCHAR(30) NOT NULL
        CHECK (direction IN ('RISK_INCREASING', 'RISK_REDUCING')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 8. Business-level explanations
CREATE TABLE IF NOT EXISTS shap_business_groups (
    group_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    business_group VARCHAR(150) NOT NULL,
    shap_contribution DOUBLE PRECISION NOT NULL,
    absolute_shap_impact DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 9. Generated underwriting reports
CREATE TABLE IF NOT EXISTS underwriting_reports (
    report_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT NOT NULL
        REFERENCES applications(applicant_id),
    decision_id BIGINT REFERENCES underwriting_decisions(decision_id),
    llm_provider VARCHAR(100),
    llm_model VARCHAR(150),
    report_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 10. RAG knowledge base
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id VARCHAR(100) PRIMARY KEY,
    title VARCHAR(250) NOT NULL,
    category VARCHAR(100),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 11. Copilot question-and-answer history
CREATE TABLE IF NOT EXISTS copilot_queries (
    query_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT REFERENCES applications(applicant_id),
    question TEXT NOT NULL,
    answer TEXT,
    retrieved_documents JSONB NOT NULL DEFAULT '[]'::jsonb,
    llm_provider VARCHAR(100),
    llm_model VARCHAR(150),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 12. Audit trail
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    applicant_id BIGINT REFERENCES applications(applicant_id),
    event_type VARCHAR(100) NOT NULL,
    event_details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_predictions_applicant
    ON predictions(applicant_id, prediction_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_policy_results_applicant
    ON policy_results(applicant_id);

CREATE INDEX IF NOT EXISTS idx_decisions_applicant
    ON underwriting_decisions(applicant_id, decision_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_shap_applicant
    ON shap_explanations(applicant_id);

CREATE INDEX IF NOT EXISTS idx_copilot_applicant
    ON copilot_queries(applicant_id);

CREATE INDEX IF NOT EXISTS idx_audit_applicant
    ON audit_log(applicant_id);

CREATE INDEX IF NOT EXISTS idx_applicant_features_json
    ON applicant_features USING GIN(feature_payload);
