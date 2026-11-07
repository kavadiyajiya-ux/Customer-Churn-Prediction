
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_rf_model.pkl"

# =========================================================
# 2. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the prediction model: {e}")
    st.stop()

# =========================================================
# 3. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: #F9FAFB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #111827;
    font-weight: 800;
}

p, label {
    color: #64748B;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF2;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 800;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E5EAF2;
    padding: 24px;
    border-radius: 14px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E5EAF2;
    border-radius: 12px;
}

hr {
    border-color: #E5EAF2;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="padding:10px 0 20px 0;">
    <div style="font-size:27px;font-weight:800;color:white;">
        📊 ChurnIQ
    </div>
    <div style="font-size:12px;color:#9CA3AF;">
        CUSTOMER INTELLIGENCE PLATFORM
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("### About the Project")

st.sidebar.write(
    "An intelligent customer churn prediction "
    "and retention analytics platform."
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Dataset:** Telco Customer Churn")
st.sidebar.write("**Prediction:** Customer Churn")

st.sidebar.divider()

st.sidebar.markdown("### Technologies")

st.sidebar.write(
    "Python · Pandas · Scikit-learn · SHAP · Streamlit"
)

st.sidebar.divider()

st.sidebar.caption(
    "Built for customer retention intelligence."
)

# =========================================================
# 5. HOMEPAGE HEADER
# =========================================================



st.markdown(
    """
    <div style="background-color:#111827; padding:35px; border-radius:18px;">
        <h1 style="color:white; font-size:32px;">
            Customer Churn Intelligence
        </h1>
        <p style="color:#CBD5E1; font-size:15px;">
            Predict customer churn, identify high-risk customers
            and generate actionable retention strategies using
            Machine Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)




# =========================================================
# 6. MODEL OVERVIEW
# =========================================================

st.markdown("### Prediction Workspace")

st.caption(
    "Enter customer information to estimate their likelihood of leaving."
)



st.divider()


# =========================================================
# 7. CUSTOMER INPUT FORM
# =========================================================

st.markdown("### Customer Information")

with st.form("customer_prediction_form"):

    st.markdown("#### 01 — Personal & Billing Details")

    col1, col2 = st.columns(2)

    with col1:
        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        partner = st.selectbox(
            "Has Partner?", ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Has Dependents?", ["No", "Yes"]
        )

    with col2:
        tenure = st.slider(
            "Tenure (Months)", 0, 72, 12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=800.0
        )

    st.divider()

    st.markdown("#### 02 — Phone & Internet Services")

    col1, col2 = st.columns(2)

    with col1:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    st.markdown("#### 03 — Contract & Payment")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 8. PREDICTION ENGINE
# =========================================================

if submitted:

    input_data = {
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,

        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,

        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,

        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,

        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,

        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,

        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,

        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,

        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,

        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,

        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,

        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0
    }

    try:
        input_df = pd.DataFrame([input_data])

        # Match the exact feature order used during model training
        input_df = input_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.session_state["churn_result"] = {
            "prediction": int(prediction),
            "probability": float(probability),
            "tenure": tenure,
            "contract": contract,
            "monthly_charges": monthly_charges,
            "tech_support": tech_support,
            "internet_service": internet_service
        }

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================================================
# 9. PREDICTION RESULT
# =========================================================

if "churn_result" in st.session_state:

    result = st.session_state["churn_result"]

    prediction = result["prediction"]
    probability = result["probability"]

    st.divider()

    st.markdown("### Prediction Results")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        st.caption(
            "Estimated probability of this customer churning."
        )

    with col2:

        if probability < 0.30:
            risk = "Low Risk"
            st.success("LOW RISK")
            st.markdown("### Customer likely to stay")

        elif probability < 0.60:
            risk = "Medium Risk"
            st.warning("MEDIUM RISK")
            st.markdown("### Customer needs attention")

        else:
            risk = "High Risk"
            st.error("HIGH RISK")
            st.markdown("### Customer may leave")

    st.write(f"**Risk Category:** {risk}")

    if prediction == 1:
        st.error("Model Prediction: Customer likely to churn.")
    else:
        st.success("Model Prediction: Customer likely to stay.")

    st.caption(
        "Note: The model prediction and the probability-based risk category "
        "are separate outputs. They may differ depending on the model's "
        "classification threshold."
    )

    # =====================================================
    # 10. RETENTION RECOMMENDATIONS
    # =====================================================

    st.divider()

    st.markdown("### Personalized Retention Recommendations")

    recommendations = []

    if prediction == 1:

        if result["tenure"] < 12:
            recommendations.append(
                "Offer a welcome benefit or loyalty discount."
            )

        if result["contract"] == "Month-to-month":
            recommendations.append(
                "Offer a discount for switching to a longer-term contract."
            )

        if result["monthly_charges"] > 80:
            recommendations.append(
                "Review the bill and offer a suitable lower-cost plan."
            )

        if result["tech_support"] == "No":
            recommendations.append(
                "Offer technical support assistance."
            )

        if result["internet_service"] == "Fiber optic":
            recommendations.append(
                "Check internet service quality and resolve connectivity complaints."
            )

        if not recommendations:
            recommendations.append(
                "Contact the customer to understand their concerns "
                "and offer personalized support."
            )

        for item in recommendations:
            st.info("📌 " + item)

    else:
        st.success(
            "This customer is currently predicted to stay. "
            "Continue providing good service and monitor satisfaction."
        )


# =========================================================
# 11. MODEL INSIGHTS
# =========================================================

st.divider()

st.markdown("### Model Insights")

st.caption(
    "Understand which features contribute most to the Random Forest model."
)

st.markdown("#### Top 10 Features Influencing Customer Churn")

feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

st.bar_chart(
    feature_importance,
    x="Feature",
    y="Importance",
    horizontal=True,
    color="#3B82F6"
)

st.caption(
    "Feature importance indicates how much a feature contributes "
    "to the Random Forest model. It does not establish causation."
)


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:15px;color:#94A3B8;">
    <b>ChurnIQ</b> · AI-Powered Customer Retention Analytics
    <br>
    <span style="font-size:12px;">
        Built with Python, Machine Learning and Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
