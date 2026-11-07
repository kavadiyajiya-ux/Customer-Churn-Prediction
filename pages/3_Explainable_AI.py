
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Explainable AI | Churn IQ",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = (
    BASE_DIR / "data" / "archive"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

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

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -1px;
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
    margin-top: 18px;
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
    font-size: 24px;
    font-weight: 800;
}

.info-card {
    background: white;
    border: 1px solid #E5E7EB;
    padding: 20px;
    border-radius: 14px;
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

    st.markdown("### Explainable AI")
    st.write(
        "Understand how machine learning identifies "
        "customers at risk of churn."
    )

    st.divider()

    st.markdown("### Explainability Engine")
    st.caption("Framework")
    st.write("SHAP")

    st.caption("Model")
    st.write("Random Forest / Logistic Regression")

    st.divider()
    st.caption("AI MODEL TRANSPARENCY")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Explainable AI Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Analytics / '
    'Model Transparency & Prediction Explanations</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return pipeline, metadata


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    pipeline, metadata = load_model()
    customers = load_data()

except Exception as error:
    st.error(f"Error loading model or dataset: {error}")
    st.stop()

if customers.empty:
    st.error("Customer dataset is empty.")
    st.stop()

# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

try:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]

    feature_columns = list(preprocessor.feature_names_in_)

    X = customers[feature_columns].copy()

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:
        if column in X.columns:
            X[column] = pd.to_numeric(
                X[column].astype(str).str.strip(),
                errors="coerce"
            )

    transformed = preprocessor.transform(X)

    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()

except Exception as error:
    st.error(f"Feature preparation failed: {error}")
    st.stop()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True
)

model_name = metadata.get("model_name", "Selected Model")

c1, c2, c3 = st.columns(3)

model_cards = [
    (c1, "SELECTED MODEL", str(model_name)),
    (c2, "EXPLAINABILITY", "SHAP"),
    (c3, "SHAP VERSION", str(shap.__version__))
]

for col, label, value in model_cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

st.info(
    "SHAP explains how individual features contribute to a model's "
    "prediction. These explanations describe model behavior; "
    "they do not establish real-world causation."
)

# --------------------------------------------------
# SELECT CUSTOMER
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Individual Customer Explanation</div>',
    unsafe_allow_html=True
)

customer_ids = customers["customerID"].astype(str).tolist()

selected_id = st.selectbox(
    "Select Customer ID",
    customer_ids
)

customer_index = customers.index[
    customers["customerID"].astype(str) == selected_id
][0]

customer_raw = X.loc[[customer_index]]
customer_transformed = transformed[[customer_index]]

probability = float(
    pipeline.predict_proba(customer_raw)[0][1]
)

probability = max(0.0, min(1.0, probability))

# --------------------------------------------------
# CUSTOMER PREDICTION CARD
# --------------------------------------------------

left, right = st.columns([2, 1])

with left:
    st.markdown(f"""
    <div class="info-card">
        <div style="color:#6B7280;font-size:12px;font-weight:600;">
            SELECTED CUSTOMER
        </div>
        <div style="font-size:25px;font-weight:800;color:#111827;">
            {selected_id}
        </div>
        <div style="color:#6B7280;font-size:13px;">
            Individual model prediction explanation
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="info-card">
        <div style="color:#6B7280;font-size:12px;font-weight:600;">
            CHURN PROBABILITY
        </div>
        <div style="font-size:25px;font-weight:800;color:#1D4ED8;">
            {probability * 100:.2f}%
        </div>
        <div style="color:#6B7280;font-size:13px;">
            Model-estimated probability
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.progress(probability)

# --------------------------------------------------
# CREATE SHAP EXPLAINER
# --------------------------------------------------

@st.cache_resource
def create_explainer(_classifier, model_type, background):

    if model_type == "RandomForest":
        return shap.TreeExplainer(_classifier)

    return shap.LinearExplainer(
        _classifier,
        background
    )


def get_positive_class_values(values):

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] > 1:
            return values[:, :, 1]
        return values[:, :, 0]

    return values


def get_positive_base_value(base_values):

    base_values = np.asarray(base_values)

    if base_values.ndim == 2:
        if base_values.shape[-1] > 1:
            return float(base_values[0, 1])
        return float(base_values[0, 0])

    if base_values.ndim == 1:
        if len(base_values) > 1:
            return float(base_values[1])
        return float(base_values[0])

    return float(base_values.item())


# --------------------------------------------------
# CALCULATE SHAP VALUES
# --------------------------------------------------

try:

    background_size = min(100, len(transformed))

    rng = np.random.default_rng(42)

    background_indices = rng.choice(
        len(transformed),
        size=background_size,
        replace=False
    )

    background = transformed[background_indices]

    if "RandomForest" in str(type(classifier)):
        model_type = "RandomForest"

    elif "LogisticRegression" in str(type(classifier)):
        model_type = "LogisticRegression"

    else:
        st.error(
            f"Unsupported model type: {type(classifier).__name__}"
        )
        st.stop()

    explainer = create_explainer(
        classifier,
        model_type,
        background
    )

    shap_result = explainer(customer_transformed)

    shap_values = get_positive_class_values(
        shap_result.values
    )[0]

    base_value = get_positive_base_value(
        shap_result.base_values
    )

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=customer_transformed[0],
        feature_names=feature_names
    )

except Exception as error:
    st.error(f"SHAP calculation failed: {error}")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# INDIVIDUAL SHAP EXPLANATION
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Why Is This Customer at Risk?</div>',
    unsafe_allow_html=True
)

st.caption(
    "The waterfall chart explains which features push the "
    "model output higher or lower for this customer."
)

try:

    fig = plt.figure(figsize=(12, 7))

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.markdown("""
    <div class="info-card">
        <b>How to interpret this chart</b>
        <br><br>
        <span style="color:#EF4444;">● Red features</span>
        push the model output higher.
        <br>
        <span style="color:#3B82F6;">● Blue features</span>
        push the model output lower.
        <br><br>
        The baseline represents the explainer's expected model output.
        The final output represents this customer's explained prediction.
        The output scale depends on the SHAP explainer.
    </div>
    """, unsafe_allow_html=True)

except Exception as error:
    st.error(f"Waterfall chart failed: {error}")
    st.exception(error)

# --------------------------------------------------
# GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Global Churn Drivers</div>',
    unsafe_allow_html=True
)

st.caption(
    "Discover which features have the greatest average impact "
    "on churn predictions across a sample of customers."
)

try:

    sample_size = min(150, len(transformed))

    sample_indices = rng.choice(
        len(transformed),
        size=sample_size,
        replace=False
    )

    sample_data = transformed[sample_indices]

    global_result = explainer(sample_data)

    global_values = get_positive_class_values(
        global_result.values
    )

    importance = np.abs(global_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value": importance
    }).sort_values(
        "Mean Absolute SHAP Value",
        ascending=False
    ).head(15)

    # ----------------------------------------------
    # GLOBAL IMPORTANCE CHART
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(
        importance_df["Feature"].astype(str)[::-1],
        importance_df["Mean Absolute SHAP Value"][::-1],
        height=0.65
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value",
        fontsize=11
    )

    ax.set_title(
        "Top 15 Features Influencing Churn Predictions",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # ----------------------------------------------
    # IMPORTANCE TABLE
    # ----------------------------------------------

    st.markdown("#### Feature Importance Details")

    st.dataframe(
        importance_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": "Model Feature",
            "Mean Absolute SHAP Value": st.column_config.NumberColumn(
                "Mean Absolute SHAP Value",
                format="%.5f"
            )
        }
    )

except Exception as error:
    st.error(f"Global feature importance failed: {error}")
    st.exception(error)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CHURN IQ | Explainable Artificial Intelligence"
)

st.caption(
    "SHAP explanations describe model behavior and should be "
    "interpreted alongside business knowledge."
)
