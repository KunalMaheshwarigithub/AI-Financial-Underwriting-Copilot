import requests
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Financial Underwriting Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. API CONFIGURATION
# ============================================================
st.sidebar.title("🏦 Underwriting Copilot")
st.sidebar.caption("ML Risk Assessment · Fuzzy Logic · RAG · GenAI")
API_BASE_URL = st.sidebar.text_input(
    "FastAPI backend URL",
    value="http://127.0.0.1:8000",
).strip().rstrip("/")

applicant_id = int(st.sidebar.number_input(
    "Applicant ID",
    min_value=1,
    max_value=9_007_199_254_740_991,
    value=100020,
    step=1,
    format="%d",
))


def call_api(method, path, payload=None, timeout=30):
    """Call FastAPI; return (decoded data, error message)."""
    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            json=payload,
            timeout=timeout,
        )
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text}

        if not response.ok:
            detail = data.get("detail", data) if isinstance(data, dict) else data
            return None, f"HTTP {response.status_code}: {detail}"
        return data, None
    except requests.Timeout:
        return None, "The request timed out. Check the backend and try again."
    except requests.RequestException as exc:
        return None, f"Could not connect to the FastAPI backend: {exc}"


if st.sidebar.button("Check API health", use_container_width=True):
    health, error = call_api("GET", "/health", timeout=10)
    if error:
        st.sidebar.error(error)
    else:
        st.sidebar.success("API is responding")
        with st.sidebar.expander("Health details"):
            st.json(health)


# ============================================================
# 3. SESSION STATE
# ============================================================
STATE_DEFAULTS = {
    "applicant_profile": None,
    "profile_applicant_id": None,
    "underwriting_result": None,
    "underwriting_applicant_id": None,
    "copilot_messages": [],
    "new_application_result": None,
    "new_application_underwriting_result": None,
}
for key, default_value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# 4. DISPLAY HELPERS
# ============================================================
def display_risk_analytics(result: dict):
    """Show fuzzy indicators, cohort comparison and missing-data status."""
    quality = result.get("data_quality_status")
    coverage = result.get("feature_coverage", {})

    if quality == "HISTORICAL_DATA_INCOMPLETE":
        st.warning(
            "Historical credit/repayment data are unavailable. The missing checks are "
            "marked NOT_EVALUATED rather than treated as policy failures. The numeric "
            "model score is experimental for this incomplete feature profile."
        )
    elif quality == "SOME_POLICY_CHECKS_UNAVAILABLE":
        st.info("Some policy checks were unavailable; inspect the individual policy results.")

    if coverage:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Non-missing model features",
                f"{coverage.get('available_model_feature_count', '—')} / {coverage.get('model_feature_count', '—')}",
            )
        with col2:
            st.metric(
                "Feature completeness",
                f"{coverage.get('non_missing_feature_percent', '—')}%",
            )
        with col3:
            st.metric(
                "Checks not evaluated",
                result.get("not_evaluated_rule_count", 0),
            )
        st.caption(
            "Feature completeness counts non-null fields, including availability flags. "
            "It is not a measure of prediction accuracy."
        )

    fuzzy = result.get("fuzzy_assessment") or {}
    st.markdown("#### Fuzzy risk explanation")
    if fuzzy.get("available"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Fuzzy risk indicator", f"{fuzzy.get('fuzzy_risk_score', '—')} / 100")
        with c2:
            st.metric("Fuzzy profile", fuzzy.get("fuzzy_risk_band", "Unavailable"))
        st.caption(
            "This is an explanatory heuristic, not a probability of default and not a validated model. "
            "It does not override the ML prediction or project policy decision."
        )
        factors = fuzzy.get("factors", [])
        if factors:
            factor_rows = []
            for item in factors:
                factor_rows.append({
                    "Feature": item.get("feature"),
                    "Value": item.get("display_value", item.get("value")),
                    "Fuzzy risk score": item.get("risk_score"),
                    "Low membership": item.get("low_risk_membership"),
                    "Medium membership": item.get("medium_risk_membership"),
                    "High membership": item.get("high_risk_membership"),
                    "Interpretation": item.get("interpretation"),
                })
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
        for explanation in fuzzy.get("explanations", []):
            st.write(f"- {explanation}")
    else:
        st.info(
            "The fuzzy indicator could not be calculated. Check whether the local historical "
            "reference parquet is available and contains the required feature distributions."
        )

    cohort = result.get("training_cohort_comparison") or {}
    st.markdown("#### Similar historical applicants")
    if cohort.get("available"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Comparable sample", cohort.get("comparison_sample_size", "—"))
        with c2:
            sim = cohort.get("mean_similarity_percent")
            st.metric("Mean similarity", f"{sim:.1f}%" if isinstance(sim, (int, float)) else "—")
        with c3:
            rate = cohort.get("similar_cohort_observed_default_rate_percent")
            st.metric("Observed default rate in comparison group", f"{rate:.2f}%" if isinstance(rate, (int, float)) else "Insufficient sample")
        baseline = cohort.get("overall_reference_default_rate_percent")
        if isinstance(baseline, (int, float)):
            st.caption(f"Overall labeled reference cohort default rate: {baseline:.2f}%.")
        st.caption(cohort.get("evidence_quality", "Exploratory comparison"))
        st.caption(
            "This is a retrospective cohort statistic, not a calibrated prediction for the applicant. "
            "It is excluded from the final decision rule."
        )
        with st.expander("Comparison method and limitations"):
            st.write(cohort.get("method", "Weighted profile similarity"))
            st.write("Features used:", ", ".join(cohort.get("features_used", [])))
            st.write(cohort.get("limitation", ""))
    else:
        st.info(cohort.get(
            "limitation",
            "Historical comparison is unavailable. Make sure models/applicant_feature_store.parquet "
            "is present and includes labeled TARGET values.",
        ))

    rationale = result.get("decision_rationale", [])
    if rationale:
        st.markdown("#### Why the system produced this result")
        for reason in rationale:
            st.write(f"- {reason}")


def display_underwriting_result(result: dict):
    pd_percent = result.get("predicted_pd_percent")
    if pd_percent is None and isinstance(result.get("predicted_pd"), (int, float)):
        pd_percent = result["predicted_pd"] * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Probability of Default", f"{pd_percent:.2f}%" if isinstance(pd_percent, (int, float)) else "Unavailable")
    with c2:
        st.metric("Model Risk Band", result.get("risk_band", "Unknown"))
    with c3:
        st.metric("Project Decision", result.get("decision", "Unknown"))

    if isinstance(pd_percent, (int, float)):
        st.progress(min(max(pd_percent / 100.0, 0.0), 1.0), text="Model-predicted probability of default")

    st.markdown("#### Decision rationale")
    st.write(result.get("decision_reason", "No decision reason was returned."))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Critical rules", result.get("critical_rule_count", 0))
    with c2:
        st.metric("Warning rules", result.get("warning_rule_count", 0))
    with c3:
        st.metric("Not evaluated", result.get("not_evaluated_rule_count", 0))

    st.markdown("#### Policy evaluation results")
    policy_results = result.get("policy_results", [])
    if policy_results:
        st.dataframe(pd.DataFrame(policy_results), use_container_width=True, hide_index=True)
    else:
        st.info("No policy results were returned.")

    display_risk_analytics(result)

    with st.expander("Complete underwriting response"):
        st.json(result)
    st.caption(result.get("notice", "Prototype decision support only; not a real lending determination."))


# ============================================================
# 5. PAGE HEADER AND TABS
# ============================================================
st.title("AI Financial Underwriting Copilot")
st.markdown(
    "A prototype combining model-based risk prediction, policy checks, fuzzy explanations, "
    "historical-applicant comparisons, PostgreSQL, and a RAG-based AI Copilot."
)
st.caption("ML inference · PostgreSQL · FastAPI · Fuzzy Logic · Historical Cohort Analysis · RAG · Cloud LLM")
st.divider()

(
    overview_tab,
    new_application_tab,
    profile_tab,
    underwriting_tab,
    copilot_tab,
) = st.tabs([
    "🏠 Overview",
    "➕ New Application",
    "👤 Applicant Profile",
    "📊 Underwriting",
    "💬 AI Copilot",
])


# ============================================================
# 6. OVERVIEW
# ============================================================
with overview_tab:
    st.subheader("Welcome to your underwriting workspace")
    st.write(
        "Create an application, run the trained model, inspect the policy outcomes, "
        "and review separate fuzzy/cohort explanations."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model predictors", "244")
    with c2:
        st.metric("RAG documents", "10+")
    with c3:
        st.metric("Decision layers", "3", help="ML prediction, deterministic policies, supplementary explanations")

    st.divider()
    st.subheader("Workflow")
    st.markdown(
        """
        1. **New Application:** Save applicant inputs and generate the model-compatible feature payload.
        2. **ML Prediction:** Run the saved model and show its probability and project risk band.
        3. **Policy Checks:** Evaluate rules with available historical data; unavailable checks remain `NOT_EVALUATED`.
        4. **Fuzzy Explanation:** Display data-derived fuzzy memberships for the model score and supported financial features.
        5. **Historical Comparison:** Compare against the local labeled historical reference set when available.
        6. **AI Copilot:** Ask about the Home Credit training data, metrics, policy rules, or a particular applicant.
        """
    )
    st.info(
        "Missing history no longer forces the decision to REVIEW. The result still has a data-quality warning, "
        "and the fuzzy/cohort analysis is exploratory rather than a validated lending score."
    )
    if st.button("View model training profile", key="load_model_info"):
        info, error = call_api("GET", "/model-info", timeout=30)
        if error:
            st.error(error)
        else:
            st.session_state["model_info_result"] = info
    if st.session_state.get("model_info_result"):
        with st.expander("Model training profile", expanded=True):
            st.json(st.session_state.model_info_result)


# ============================================================
# 7. NEW APPLICATION
# ============================================================
with new_application_tab:
    st.subheader("Register a New Loan Application")
    st.write(
        "Enter the applicant details available at intake. Historical credit scores and repayment behaviour "
        "are not invented when they are unavailable."
    )

    with st.form("new_application_form"):
        st.markdown("#### Application details")
        application_reference = st.text_input("Application reference (optional)", max_chars=80, placeholder="Example: DEMO-NEW-002")
        c1, c2 = st.columns(2)
        with c1:
            annual_income = st.number_input("Annual income", min_value=1.0, value=108000.0, step=5000.0)
            credit_amount = st.number_input("Requested credit amount", min_value=1.0, value=300000.0, step=10000.0)
            annuity_amount = st.number_input("Annuity amount (use same units as training data)", min_value=1.0, value=18000.0, step=1000.0)
            goods_price_input = st.number_input("Goods or asset price (0 if unavailable)", min_value=0.0, value=280000.0, step=10000.0)
        with c2:
            age_years = st.number_input("Applicant age", min_value=18, max_value=100, value=35, step=1)
            employment_years = st.number_input("Years of employment", min_value=0.0, max_value=80.0, value=5.0, step=0.5)
            children_count = st.number_input("Number of children", min_value=0, max_value=30, value=0, step=1)

        st.markdown("#### Applicant profile")
        c1, c2 = st.columns(2)
        with c1:
            contract_type = st.selectbox("Contract type", ["Cash loans", "Revolving loans"])
            income_type = st.selectbox("Income type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"])
            education_type = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
            housing_type = st.selectbox("Housing type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
        with c2:
            occupation_type = st.text_input("Occupation (optional)", placeholder="Example: Drivers")
            organization_type = st.text_input("Organization type (optional)", placeholder="Use a category consistent with training data")
            owns_car_choice = st.selectbox("Owns a car", ["Unknown", "Y", "N"])
            owns_realty_choice = st.selectbox("Owns real estate", ["Unknown", "Y", "N"])
            type_suite = st.text_input("Who accompanied the applicant? (optional)")
            weekday_choice = st.selectbox("Application weekday", ["Unknown", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"])

        st.caption("Verify monetary units and input meaning before submitting real data.")
        submitted_application = st.form_submit_button("Create Application", type="primary", use_container_width=True)

    if submitted_application:
        if employment_years > age_years:
            st.error("Employment duration cannot exceed the applicant's age.")
        else:
            payload = {
                "application_reference": application_reference.strip() or None,
                "annual_income": annual_income,
                "credit_amount": credit_amount,
                "annuity_amount": annuity_amount,
                "goods_price": goods_price_input if goods_price_input > 0 else None,
                "age_years": int(age_years),
                "employment_years": employment_years,
                "children_count": int(children_count),
                "contract_type": contract_type,
                "income_type": income_type,
                "education_type": education_type,
                "housing_type": housing_type,
                "occupation_type": occupation_type.strip() or None,
                "organization_type": organization_type.strip() or None,
                "owns_car": None if owns_car_choice == "Unknown" else owns_car_choice,
                "owns_realty": None if owns_realty_choice == "Unknown" else owns_realty_choice,
                "type_suite": type_suite.strip() or None,
                "weekday_application": None if weekday_choice == "Unknown" else weekday_choice,
            }
            with st.spinner("Creating application..."):
                data, error = call_api("POST", "/applications", payload=payload, timeout=30)
            if error:
                st.error(error)
            else:
                st.session_state.new_application_result = data
                st.session_state.new_application_underwriting_result = None
                st.success(f"Application created. Applicant ID: {data['applicant_id']}")

    new_application = st.session_state.new_application_result
    if new_application:
        new_id = int(new_application["applicant_id"])
        st.divider()
        st.markdown("#### Created application")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Applicant ID", new_id)
        with c2:
            st.metric("Status", new_application.get("status", "APPLICATION_CREATED"))
        with c3:
            st.metric("Model feature count", new_application.get("model_feature_count", 244))
        with st.expander("Application creation details"):
            st.json(new_application)

        st.warning(
            "The next action creates a prediction, policy results, a decision record, and an audit event. "
            "For new applicants, model performance is not validated when many history-based features are missing."
        )
        if st.button("Run Underwriting for This Applicant", key="underwrite_new_applicant", type="primary"):
            with st.spinner("Running model and supplementary analysis..."):
                data, error = call_api("POST", f"/applicants/{new_id}/underwrite", payload={}, timeout=90)
            if error:
                st.error(error)
            else:
                st.session_state.new_application_underwriting_result = data

        new_underwriting = st.session_state.new_application_underwriting_result
        if new_underwriting:
            st.divider()
            st.markdown("#### Underwriting result")
            display_underwriting_result(new_underwriting)


# ============================================================
# 8. APPLICANT PROFILE
# ============================================================
with profile_tab:
    st.subheader(f"Applicant Profile · {applicant_id}")
    st.write("Retrieve the selected applicant's saved profile from PostgreSQL.")

    if st.button("Retrieve Applicant", key="retrieve_applicant", type="primary"):
        with st.spinner("Retrieving applicant data..."):
            data, error = call_api("GET", f"/applicants/{applicant_id}")
        if error:
            st.error(error)
        else:
            st.session_state.applicant_profile = data
            st.session_state.profile_applicant_id = applicant_id

    profile_data = st.session_state.applicant_profile
    if profile_data and st.session_state.profile_applicant_id == applicant_id:
        profile = profile_data.get("profile", {})
        st.success("Applicant retrieved successfully.")
        c1, c2, c3, c4 = st.columns(4)
        income = profile.get("AMT_INCOME_TOTAL")
        credit = profile.get("AMT_CREDIT")
        annuity = profile.get("AMT_ANNUITY")
        ratio = profile.get("CREDIT_TO_INCOME")
        with c1:
            st.metric("Annual Income", f"{income:,.0f}" if isinstance(income, (int, float)) else "Unavailable")
        with c2:
            st.metric("Credit Amount", f"{credit:,.0f}" if isinstance(credit, (int, float)) else "Unavailable")
        with c3:
            st.metric("Annuity", f"{annuity:,.0f}" if isinstance(annuity, (int, float)) else "Unavailable")
        with c4:
            st.metric("Credit / Income", f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "Unavailable")
        profile_df = pd.DataFrame([{"Feature": key, "Value": value} for key, value in profile.items()])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        with st.expander("Complete applicant API response"):
            st.json(profile_data)
    elif st.session_state.profile_applicant_id != applicant_id:
        st.info("Click 'Retrieve Applicant' to load the selected applicant.")


# ============================================================
# 9. UNDERWRITING
# ============================================================
with underwriting_tab:
    st.subheader(f"Underwriting Assessment · {applicant_id}")
    st.warning("Each run creates another prediction and decision history record.")
    if st.button("Run Underwriting Assessment", key="run_underwriting", type="primary"):
        with st.spinner("Running the ML model and policy rules..."):
            data, error = call_api("POST", f"/applicants/{applicant_id}/underwrite", payload={}, timeout=90)
        if error:
            st.error(error)
        else:
            st.session_state.underwriting_result = data
            st.session_state.underwriting_applicant_id = applicant_id

    result = st.session_state.underwriting_result
    if result and st.session_state.underwriting_applicant_id == applicant_id:
        st.success("Assessment completed.")
        display_underwriting_result(result)
    elif st.session_state.underwriting_applicant_id != applicant_id:
        st.info("Click 'Run Underwriting Assessment' to generate a new assessment.")


# ============================================================
# 10. AI COPILOT
# ============================================================
with copilot_tab:
    st.subheader("Ask the AI Copilot")
    st.write("Ask about the training dataset, model metrics, policy outcomes, or a recorded applicant assessment.")

    options = []
    latest_application = st.session_state.new_application_result
    if latest_application:
        options.append(int(latest_application["applicant_id"]))
    if applicant_id not in options:
        options.append(applicant_id)

    copilot_applicant_id = st.selectbox(
        "Applicant context",
        options=options,
        index=0,
        format_func=lambda value: f"Applicant {value}",
    )
    use_applicant_context = st.checkbox(
        "Include applicant context",
        value=True,
        help="Turn this off for a general question about the training data or project policies.",
    )

    for message in st.session_state.copilot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander("Retrieved reference documents"):
                    for source in sources:
                        st.markdown(
                            f"- **{source.get('title', 'Document')}** "
                            f"({source.get('category', 'unknown')}; similarity: {source.get('similarity', '—')})"
                        )

    with st.form("copilot_question_form"):
        question = st.text_area(
            "Your question",
            placeholder=(
                "What data was the model trained on?\n"
                "Why was this applicant approved or referred for review?\n"
                "What does the fuzzy risk indicator mean?"
            ),
            max_chars=2000,
        )
        top_k = st.slider("Knowledge documents to retrieve", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Ask Copilot", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            payload = {
                "question": question.strip(),
                "applicant_id": int(copilot_applicant_id) if use_applicant_context else None,
                "top_k": top_k,
            }
            st.session_state.copilot_messages.append({"role": "user", "content": question.strip()})
            with st.spinner("Retrieving evidence and generating an answer..."):
                data, error = call_api("POST", "/copilot/ask", payload=payload, timeout=120)
            if error:
                st.session_state.copilot_messages.append({"role": "assistant", "content": f"Request failed: {error}"})
                st.error(error)
            else:
                st.session_state.copilot_messages.append({
                    "role": "assistant",
                    "content": data.get("answer", "The API did not return an answer."),
                    "sources": data.get("retrieved_documents", []),
                })
            st.rerun()

    st.caption(
        "The Copilot explains model outputs and project assumptions. It does not override the saved decision."
    )
