
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard | Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customer_risk_predictions.csv"

# --------------------------------------------------
# CUSTOM STYLING
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

[data-testid="stSidebar"] h1 {
    color: white;
    font-weight: 800;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -1px;
    margin-bottom: 3px;
}

.subtitle {
    color: #6B7280;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #111827;
    margin-top: 15px;
    margin-bottom: 15px;
}

.kpi-card {
    background: white;
    padding: 21px;
    border-radius: 14px;
    border: 1px solid #E8ECF3;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    min-height: 125px;
}

.kpi-label {
    color: #6B7280;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.kpi-value {
    color: #111827;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.8px;
}

.kpi-note {
    color: #9CA3AF;
    font-size: 11px;
    margin-top: 8px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
}

.stDownloadButton button {
    background-color: #1D4ED8;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

.stDownloadButton button:hover {
    background-color: #1E40AF;
    color: white;
}

hr {
    border-color: #E5E7EB;
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

    st.markdown("### Executive Overview")
    st.write("Monitor customer churn, financial exposure and retention opportunities.")

    st.divider()

    st.markdown("### Platform Information")
    st.caption("Machine Learning Model")
    st.write("Random Forest")

    st.caption("Dataset")
    st.write("Telco Customer Churn")

    st.caption("Technology")
    st.write("Python · Scikit-learn · Plotly · Streamlit")

    st.divider()

    st.caption("CUSTOMER RETENTION INTELLIGENCE")
    st.caption("Executive Analytics")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Customer Retention Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Executive Overview  /  Customer Churn Analytics</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error("Prediction file not found. Please run src/predict.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "predicted_churn",
    "risk_level",
    "MonthlyCharges",
    "churn_probability"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns in prediction file: {missing_columns}")
    st.stop()

if df.empty:
    st.warning("The prediction dataset is empty.")
    st.stop()

# --------------------------------------------------
# BUSINESS METRICS
# --------------------------------------------------

total_customers = len(df)

predicted_churners = int(df["predicted_churn"].sum())

churn_rate = (
    predicted_churners / total_customers * 100
    if total_customers else 0
)

potential_exposure = df.loc[
    df["predicted_churn"] == 1,
    "MonthlyCharges"
].sum()

high_risk = int((df["risk_level"] == "High").sum())
medium_risk = int((df["risk_level"] == "Medium").sum())
low_risk = int((df["risk_level"] == "Low").sum())

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Business Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

kpis = [
    (c1, "TOTAL CUSTOMERS", f"{total_customers:,}", "Customers analyzed"),
    (c2, "PREDICTED CHURNERS", f"{predicted_churners:,}", "Customers at risk"),
    (c3, "PREDICTED CHURN RATE", f"{churn_rate:.1f}%", "Estimated churn rate"),
    (c4, "MONTHLY EXPOSURE", f"₹{potential_exposure:,.0f}", "Charges linked to predicted churners")
]

for column, label, value, note in kpis:
    with column:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Customer Risk Distribution</div>',
    unsafe_allow_html=True
)

risk_counts = pd.DataFrame({
    "Risk Level": ["High", "Medium", "Low"],
    "Customers": [high_risk, medium_risk, low_risk]
})

left, right = st.columns(2)

chart_layout = dict(
    template="plotly_white",
    font=dict(family="DM Sans", color="#374151"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=25, r=25, t=65, b=25),
    height=350
)

risk_colors = {
    "High": "#EF4444",
    "Medium": "#F59E0B",
    "Low": "#10B981"
}

with left:
    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Customers",
        hole=0.65,
        title="Risk Category Breakdown",
        color="Risk Level",
        color_discrete_map=risk_colors
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        marker=dict(line=dict(color="white", width=3))
    )

    fig.update_layout(
        **chart_layout,
        legend=dict(orientation="h", y=-0.08)
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Customers",
        color="Risk Level",
        text="Customers",
        title="Customers by Risk Level",
        color_discrete_map=risk_colors
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0
    )

    fig.update_layout(
        **chart_layout,
        showlegend=False,
        xaxis_title="Risk Category",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CHURN PROBABILITY ANALYSIS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Churn Probability Analysis</div>',
    unsafe_allow_html=True
)

fig = px.histogram(
    df,
    x="churn_probability",
    nbins=30,
    title="Distribution of Customer Churn Probabilities",
    labels={
        "churn_probability": "Churn Probability",
        "count": "Number of Customers"
    },
    color_discrete_sequence=["#3B82F6"]
)

fig.update_layout(
    **{**chart_layout, "height": 370},
    xaxis=dict(title="Churn Probability", tickformat=".0%"),
    yaxis_title="Number of Customers",
    bargap=0.08
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# HIGH-RISK CUSTOMER TABLE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Customers Requiring Attention</div>',
    unsafe_allow_html=True
)

st.caption(
    "The following customers have been identified as likely to churn, "
    "ranked by predicted churn probability."
)

top_customers = (
    df[df["predicted_churn"] == 1]
    .sort_values("churn_probability", ascending=False)
    .head(20)
)

if not top_customers.empty:

    display_df = top_customers.copy()

    display_df["churn_probability"] = (
        display_df["churn_probability"] * 100
    ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "churn_probability": st.column_config.NumberColumn(
                "Churn Probability",
                format="%.2f%%"
            ),
            "MonthlyCharges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="₹%.2f"
            )
        }
    )

else:
    st.success("No customers are currently classified as predicted churners.")

# --------------------------------------------------
# EXPORT REPORT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Export Customer Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Download the complete customer risk analysis for further business review."
)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Risk Report",
    data=csv,
    file_name="customer_risk_report.csv",
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
    "Predictions are machine learning estimates, not confirmed churn events. "
    "Monthly exposure represents charges associated with predicted churners."
)
