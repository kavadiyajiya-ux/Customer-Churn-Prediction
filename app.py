
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("best_rf_model.pkl")

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Churn Analytics")
st.sidebar.write("Customer Churn Prediction System")
st.sidebar.divider()

st.sidebar.subheader("About Project")
st.sidebar.write("""
This project uses Machine Learning
to predict customer churn.

**Model:** Random Forest

**Dataset:** Telco Customer Churn

**Technologies:**
- Python
- Pandas
- Scikit-learn
- SHAP
- Streamlit
""")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")
st.divider()

# -----------------------------
# CUSTOMER DETAILS
# -----------------------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
    )

with col2:
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, max_value=10000.0, value=800.0
    )

st.divider()
st.subheader("Services")

col1, col2 = st.columns(2)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn", type="primary"):

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

    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # PREDICTION RESULT
    st.divider()
    st.header("Prediction Result")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    if probability < 0.30:
        risk = "Low Risk"
        st.success("🟢 Low Risk - Customer is likely to stay.")

    elif probability < 0.60:
        risk = "Medium Risk"
        st.warning("🟡 Medium Risk - Customer needs attention.")

    else:
        risk = "High Risk"
        st.error("🔴 High Risk - Customer may leave.")

    st.write(f"**Risk Category:** {risk}")
    st.progress(float(probability))

    if prediction == 1:
        st.write("**Model Prediction:** Customer likely to churn.")
    else:
        st.write("**Model Prediction:** Customer likely to stay.")

    # RETENTION RECOMMENDATIONS
    st.divider()
    st.subheader("💡 Customer Retention Recommendations")

    recommendations = []

    if prediction == 1:

        if tenure < 12:
            recommendations.append(
                "Offer a welcome benefit or loyalty discount."
            )

        if contract == "Month-to-month":
            recommendations.append(
                "Offer a discount for switching to a longer-term contract."
            )

        if monthly_charges > 80:
            recommendations.append(
                "Review the bill and offer a suitable lower-cost plan."
            )

        if tech_support == "No":
            recommendations.append(
                "Offer technical support assistance."
            )

        if internet_service == "Fiber optic":
            recommendations.append(
                "Check internet service quality and resolve connectivity complaints."
            )

        if not recommendations:
            recommendations.append(
                "Contact the customer to understand their concerns and offer personalized support."
            )

        for item in recommendations:
            st.info("📌 " + item)

    else:
        st.success(
            "Customer is currently predicted to stay. "
            "Continue providing good service and monitor satisfaction."
        )

# -----------------------------
# MODEL INSIGHTS
# -----------------------------
st.divider()
st.header("📊 Model Insights")
st.subheader("Top 10 Features Influencing Customer Churn")

feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

st.bar_chart(
    feature_importance.set_index("Feature")
)

st.caption(
    "Feature importance shows how much each feature contributes "
    "to the Random Forest model's predictions. It does not show causation."
)
