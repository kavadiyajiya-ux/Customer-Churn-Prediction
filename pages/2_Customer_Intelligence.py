import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="👤",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

# Allow importing modules from src
sys.path.insert(0, str(BASE_DIR))

from src.recommendations import generate_recommendations

DATA_PATH = (
    BASE_DIR / "data" / "archive"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PREDICTION_PATH = (
    BASE_DIR / "data" / "customer_risk_predictions.csv"
)

# ----------------------------------
# 2. PAGE TITLE
# ----------------------------------

st.title("👤 Customer Intelligence")
st.caption(
    "Search, understand and prioritize individual customers."
)

st.divider()

# ----------------------------------
# 3. LOAD DATA
# ----------------------------------

if not DATA_PATH.exists():
    st.error("Original customer dataset not found.")
    st.stop()

if not PREDICTION_PATH.exists():
    st.error("Customer prediction file not found.")
    st.info(
        "Run: python src/predict.py "
        "data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    st.stop()

customers = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTION_PATH)

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
    st.error(
        f"Missing columns in prediction file: {missing_columns}"
    )
    st.stop()

df = customers.merge(
    predictions[prediction_columns],
    on="customerID",
    how="inner"
)

if df.empty:
    st.error("No matching customer records found.")
    st.stop()

# ----------------------------------
# 4. CUSTOMER SEARCH
# ----------------------------------

st.subheader("🔍 Search Customer")

customer_id = st.text_input(
    "Enter Customer ID",
    placeholder="Example: 7590-VHVEG"
)

# ----------------------------------
# 5. DISPLAY CUSTOMER PROFILE
# ----------------------------------

if customer_id:

    customer = df[
        df["customerID"].astype(str).str.upper()
        == customer_id.strip().upper()
    ]

    if customer.empty:

        st.error(
            "Customer not found. Please check the Customer ID."
        )

    else:

        row = customer.iloc[0]

        st.divider()

        st.subheader(
            f"Customer Profile: {row['customerID']}"
        )

        # Customer metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{row['churn_probability'] * 100:.2f}%"
        )

        col2.metric(
            "Risk Category",
            str(row["risk_level"])
        )

        col3.metric(
            "Predicted Churn",
            "Yes" if row["predicted_churn"] == 1 else "No"
        )

        # Probability progress bar
        st.write("**Churn Risk Probability**")

        st.progress(
            max(0.0, min(1.0, float(row["churn_probability"])))
        )

        st.divider()

        # ----------------------------------
        # 6. CUSTOMER DETAILS
        # ----------------------------------

        st.subheader("📋 Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Tenure**")
            st.write(f"{row['tenure']} months")

            st.write("**Monthly Charges**")
            st.write(f"${row['MonthlyCharges']}")

            st.write("**Contract**")
            st.write(row["Contract"])

            st.write("**Gender**")
            st.write(row["gender"])

        with col2:
            st.write("**Internet Service**")
            st.write(row["InternetService"])

            st.write("**Payment Method**")
            st.write(row["PaymentMethod"])

            st.write("**Partner**")
            st.write(row["Partner"])

            st.write("**Dependents**")
            st.write(row["Dependents"])

        with col3:
            st.write("**Online Security**")
            st.write(row["OnlineSecurity"])

            st.write("**Technical Support**")
            st.write(row["TechSupport"])

            st.write("**Online Backup**")
            st.write(row["OnlineBackup"])

            st.write("**Paperless Billing**")
            st.write(row["PaperlessBilling"])

        st.divider()

        # ----------------------------------
        # 7. RETENTION RECOMMENDATION ENGINE
        # ----------------------------------

        st.subheader("💡 Personalized Retention Recommendations")

        # Generate recommendations for the selected customer
        recommendations = generate_recommendations(row)

        st.write(
            f"Found **{len(recommendations)} personalized "
            "retention actions** for this customer."
        )

        # Display each recommendation
        for index, recommendation in enumerate(recommendations, start=1):

            priority = recommendation["priority"]

            if priority == "High":
                st.error(
                    f"🔴 Recommendation {index} | HIGH PRIORITY"
                )

            elif priority == "Medium":
                st.warning(
                    f"🟡 Recommendation {index} | MEDIUM PRIORITY"
                )

            else:
                st.info(
                    f"🟢 Recommendation {index} | LOW PRIORITY"
                )

            st.markdown(
                f"**Suggested Action:** "
                f"{recommendation['action']}"
            )

            st.markdown(
                f"**Reason:** "
                f"{recommendation['reason']}"
            )

            st.divider()

        st.caption(
            "Recommendations are generated using predefined "
            "customer attributes and business rules. They are "
            "suggestions, not guaranteed retention outcomes."
        )

# ----------------------------------
# 8. DEFAULT CUSTOMER TABLE
# ----------------------------------

else:

    st.subheader("🔥 Highest-Risk Customers")

    top_customers = df.sort_values(
        by="churn_probability",
        ascending=False
    ).head(10)

    display_columns = [
        "customerID",
        "churn_probability",
        "risk_level",
        "MonthlyCharges",
        "Contract"
    ]

    st.dataframe(
        top_customers[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Enter a Customer ID above to view their complete profile."
    )

# ----------------------------------
# 9. DOWNLOAD CUSTOMER DATA
# ----------------------------------

st.divider()

st.subheader("📥 Export Customer Information")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Intelligence Report",
    data=csv,
    file_name="customer_intelligence_report.csv",
    mime="text/csv"
)

st.caption(
    "Customer churn predictions are estimates generated "
    "by the trained ML model."
)