
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customer_risk_predictions.csv"

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Retention Intelligence")
st.caption("Executive Dashboard | Customer Churn Overview")

if not DATA_PATH.exists():
    st.error("Prediction file not found. Run src/predict.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Key metrics
total_customers = len(df)
predicted_churners = int(df["predicted_churn"].sum())
churn_rate = predicted_churners / total_customers * 100 if total_customers else 0
potential_exposure = df.loc[
    df["predicted_churn"] == 1,
    "MonthlyCharges"
].sum()

high_risk = int((df["risk_level"] == "High").sum())
medium_risk = int((df["risk_level"] == "Medium").sum())
low_risk = int((df["risk_level"] == "Low").sum())

st.subheader("Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Predicted Churners", f"{predicted_churners:,}")
c3.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
c4.metric("Potential Monthly Exposure", f"₹{potential_exposure:,.2f}")

st.divider()

# Risk distribution
st.subheader("Customer Risk Distribution")

left, right = st.columns(2)

risk_counts = pd.DataFrame({
    "Risk Level": ["High", "Medium", "Low"],
    "Customers": [high_risk, medium_risk, low_risk]
})

with left:
    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Customers",
        hole=0.55,
        title="Risk Category Distribution",
        color="Risk Level",
        color_discrete_map={
            "High": "#E76F51",
            "Medium": "#E9C46A",
            "Low": "#2A9D8F"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Customers",
        color="Risk Level",
        title="Customers by Risk Level",
        text="Customers",
        color_discrete_map={
            "High": "#E76F51",
            "Medium": "#E9C46A",
            "Low": "#2A9D8F"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Churn probability distribution
st.subheader("Churn Probability Analysis")

fig = px.histogram(
    df,
    x="churn_probability",
    nbins=25,
    title="Distribution of Customer Churn Probabilities",
    labels={"churn_probability": "Churn Probability"}
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Top risk customers
st.subheader("Customers Requiring Attention")

top_customers = df[
    df["predicted_churn"] == 1
].sort_values(
    "churn_probability",
    ascending=False
).head(20)

st.dataframe(
    top_customers,
    use_container_width=True,
    hide_index=True
)

# Download
st.subheader("Export Customer Analysis")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Risk Report",
    data=csv,
    file_name="customer_risk_report.csv",
    mime="text/csv"
)

st.caption(
    "Note: Predictions are model estimates, not confirmed churn events. "
    "Monthly exposure represents charges associated with predicted churners."
)
