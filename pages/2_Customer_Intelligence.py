
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence | Churn IQ",
    page_icon="👤",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.recommendations import generate_recommendations

DATA_PATH = (
    BASE_DIR / "data" / "archive"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PREDICTION_PATH = (
    BASE_DIR / "data" / "customer_risk_predictions.csv"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -1px;
}

.subtitle {
    color: #6B7280;
    font-size: 14px;
    margin-bottom: 24px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #111827;
    margin-top: 16px;
    margin-bottom: 12px;
}

.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E8ECF3;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    min-height: 115px;
}

.kpi-label {
    color: #6B7280;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

.kpi-value {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
}

.profile-card {
    background: white;
    border: 1px solid #E5E7EB;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.recommendation-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-left: 5px solid #3B82F6;
    border-radius: 12px;
    padding: 17px;
    margin-bottom: 12px;
}

.high-priority {
    border-left-color: #EF4444;
}

.medium-priority {
    border-left-color: #F59E0B;
}

.low-priority {
    border-left-color: #10B981;
}

.stDownloadButton button {
    background-color: #1D4ED8;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 600;
}

.stDownloadButton button:hover {
    background-color: #1E40AF;
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("CHURN IQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### Customer Intelligence")
    st.write(
        "Explore individual customer profiles, "
        "understand churn risk and discover retention actions."
    )

    st.divider()

    st.markdown("### Customer Analytics")
    st.caption("Machine Learning")
    st.write("Random Forest")

    st.caption("Recommendation Engine")
    st.write("Rule-based personalized actions")

    st.divider()
    st.caption("CUSTOMER RETENTION INTELLIGENCE")

# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Customer Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Customer Management / '
    'Profiles, Risk Analysis & Retention Actions</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error("Original customer dataset not found.")
    st.stop()

if not PREDICTION_PATH.exists():
    st.error("Customer prediction file not found.")
    st.info("Run: python src/predict.py")
    st.stop()

@st.cache_data
def load_customer_data(data_path, prediction_path):
    customers = pd.read_csv(data_path)
    predictions = pd.read_csv(prediction_path)

    prediction_columns = [
        "customerID",
        "churn_probability",
        "predicted_churn",
        "risk_level"
    ]

    missing_columns = [
        col for col in prediction_columns
        if col not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    return customers.merge(
        predictions[prediction_columns],
        on="customerID",
        how="inner"
    )

try:
    df = load_customer_data(
        str(DATA_PATH),
        str(PREDICTION_PATH)
    )
except Exception as error:
    st.error(f"Unable to load customer data: {error}")
    st.stop()

if df.empty:
    st.error("No matching customer records found.")
    st.stop()

# --------------------------------------------------
# CUSTOMER SEARCH
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Find a Customer</div>',
    unsafe_allow_html=True
)

search_col, info_col = st.columns([2, 1])

with search_col:
    customer_id = st.text_input(
        "Customer ID",
        placeholder="Enter ID, e.g. 7590-VHVEG",
        help="Enter the exact customer ID to view their profile."
    )

with info_col:
    st.write("")
    st.write("")
    st.caption(f"Total customer records available: {len(df):,}")

# --------------------------------------------------
# CUSTOMER PROFILE
# --------------------------------------------------

if customer_id:

    customer = df[
        df["customerID"].astype(str).str.upper()
        == customer_id.strip().upper()
    ]

    if customer.empty:
        st.error("Customer not found. Please check the Customer ID.")

    else:

        row = customer.iloc[0]

        probability = float(row["churn_probability"])
        probability = max(0.0, min(1.0, probability))

        is_churner = int(row["predicted_churn"]) == 1
        risk = str(row["risk_level"])

        st.divider()

        st.markdown(
            f'<div class="section-title">Customer Profile '
            f'· {row["customerID"]}</div>',
            unsafe_allow_html=True
        )

        # ------------------------------------------
        # CUSTOMER KPI CARDS
        # ------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (c1, "CHURN PROBABILITY", f"{probability * 100:.1f}%"),
            (c2, "RISK CATEGORY", risk),
            (c3, "MONTHLY CHARGES", f"${float(row['MonthlyCharges']):,.2f}"),
            (c4, "CUSTOMER TENURE", f"{row['tenure']} months")
        ]

        for col, label, value in metrics:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        # ------------------------------------------
        # RISK STATUS
        # ------------------------------------------

        left, right = st.columns([2, 1])

        with left:
            st.markdown("#### Churn Risk Assessment")
            st.progress(probability)
            st.caption(
                f"Model-estimated probability: {probability * 100:.2f}%"
            )

        with right:
            if risk == "High":
                st.error("HIGH RISK")
            elif risk == "Medium":
                st.warning("MEDIUM RISK")
            else:
                st.success("LOW RISK")

            if is_churner:
                st.write("Model prediction: **Likely to churn**")
            else:
                st.write("Model prediction: **Likely to stay**")

        st.divider()

        # ------------------------------------------
        # CUSTOMER DETAILS
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">Customer Information</div>',
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs([
            "Personal Details",
            "Services & Subscription",
            "Billing & Payment"
        ])

        with tab1:
            a, b = st.columns(2)

            with a:
                st.markdown("**Gender**")
                st.write(row["gender"])

                st.markdown("**Senior Citizen**")
                st.write("Yes" if row["SeniorCitizen"] == 1 else "No")

                st.markdown("**Partner**")
                st.write(row["Partner"])

            with b:
                st.markdown("**Dependents**")
                st.write(row["Dependents"])

                st.markdown("**Phone Service**")
                st.write(row["PhoneService"])

                st.markdown("**Multiple Lines**")
                st.write(row["MultipleLines"])

        with tab2:
            a, b = st.columns(2)

            with a:
                st.markdown("**Internet Service**")
                st.write(row["InternetService"])

                st.markdown("**Online Security**")
                st.write(row["OnlineSecurity"])

                st.markdown("**Online Backup**")
                st.write(row["OnlineBackup"])

                st.markdown("**Device Protection**")
                st.write(row["DeviceProtection"])

            with b:
                st.markdown("**Technical Support**")
                st.write(row["TechSupport"])

                st.markdown("**Streaming TV**")
                st.write(row["StreamingTV"])

                st.markdown("**Streaming Movies**")
                st.write(row["StreamingMovies"])

        with tab3:
            a, b = st.columns(2)

            with a:
                st.markdown("**Contract Type**")
                st.write(row["Contract"])

                st.markdown("**Paperless Billing**")
                st.write(row["PaperlessBilling"])

            with b:
                st.markdown("**Payment Method**")
                st.write(row["PaymentMethod"])

                st.markdown("**Total Charges**")
                st.write(f"${float(row['TotalCharges']):,.2f}")

        st.divider()

        # ------------------------------------------
        # RETENTION RECOMMENDATIONS
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">Personalized Retention Strategy</div>',
            unsafe_allow_html=True
        )

        recommendations = generate_recommendations(row)

        if recommendations is None:
            recommendations = []

        st.caption(
            f"{len(recommendations)} personalized retention actions "
            "generated for this customer."
        )

        if recommendations:

            for index, recommendation in enumerate(
                recommendations, start=1
            ):

                priority = recommendation.get(
                    "priority", "Low"
                )

                if priority == "High":
                    css_class = "high-priority"
                    priority_icon = "🔴"

                elif priority == "Medium":
                    css_class = "medium-priority"
                    priority_icon = "🟡"

                else:
                    css_class = "low-priority"
                    priority_icon = "🟢"

                st.markdown(f"""
                <div class="recommendation-card {css_class}">
                    <div style="font-weight:700; color:#111827;">
                        {priority_icon} ACTION {index}
                        &nbsp; · &nbsp; {priority.upper()} PRIORITY
                    </div>
                    <br>
                    <div style="font-weight:600; color:#374151;">
                        Suggested Action
                    </div>
                    <div style="color:#4B5563;">
                        {recommendation.get('action', 'No action provided')}
                    </div>
                    <br>
                    <div style="font-weight:600; color:#374151;">
                        Business Reason
                    </div>
                    <div style="color:#6B7280;">
                        {recommendation.get('reason', 'No reason provided')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.success(
                "No specific retention actions are currently suggested "
                "for this customer."
            )

        st.caption(
            "Recommendations are generated using customer attributes "
            "and predefined business rules. They are suggestions, "
            "not guaranteed retention outcomes."
        )

# --------------------------------------------------
# DEFAULT CUSTOMER OVERVIEW
# --------------------------------------------------

else:

    st.divider()

    st.markdown(
        '<div class="section-title">Highest-Risk Customers</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Customers ranked by their model-estimated churn probability."
    )

    top_customers = (
        df.sort_values(
            by="churn_probability",
            ascending=False
        )
        .head(10)
    )

    display_columns = [
        "customerID",
        "churn_probability",
        "risk_level",
        "MonthlyCharges",
        "Contract"
    ]

    display_df = top_customers[display_columns].copy()

    display_df["churn_probability"] = (
        display_df["churn_probability"] * 100
    ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "customerID": "Customer ID",
            "churn_probability": st.column_config.NumberColumn(
                "Churn Probability",
                format="%.2f%%"
            ),
            "risk_level": "Risk Level",
            "MonthlyCharges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="$%.2f"
            ),
            "Contract": "Contract Type"
        }
    )

    st.info(
        "Search for a customer above to explore their complete profile "
        "and personalized retention strategy."
    )

# --------------------------------------------------
# EXPORT
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Export Customer Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Download the complete customer intelligence dataset."
)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Customer Intelligence Report",
    data=csv,
    file_name="customer_intelligence_report.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CHURN IQ | Customer Retention Intelligence Platform"
)

st.caption(
    "All churn probabilities are machine learning estimates. "
    "Retention recommendations are business-rule-based suggestions."
)
