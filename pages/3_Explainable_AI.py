
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import matplotlib.pyplot as plt
from scipy import sparse

st.set_page_config(
    page_title="Explainable AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explainable AI - Customer Churn")
st.write(
    "Understand why customers are predicted to leave "
    "and which factors influence churn."
)

# -------------------------------
# 1. LOAD MODEL AND DATA
# -------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/churn_pipeline.pkl")

    with open("models/model_metadata.json", "r") as file:
        metadata = json.load(file)

    return model, metadata


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )


try:
    pipeline, metadata = load_model()
    customers = load_data()

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# -------------------------------
# 2. PREPARE FEATURES
# -------------------------------

feature_columns = (
    pipeline.named_steps["preprocessor"].feature_names_in_
)

X = customers[list(feature_columns)].copy()

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:
    X[column] = pd.to_numeric(
        X[column].astype(str).str.strip(),
        errors="coerce"
    )

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["model"]

try:
    transformed = preprocessor.transform(X)

    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()

except Exception as e:
    st.error(f"Feature preparation failed: {e}")
    st.stop()


# -------------------------------
# 3. MODEL INFORMATION
# -------------------------------

st.subheader("Model Information")

model_name = metadata.get("model_name", "Selected model")

col1, col2 = st.columns(2)

col1.metric("Selected Model", str(model_name))
col2.metric("SHAP Version", shap.__version__)

st.info(
    "SHAP explains how each feature contributes to the model's "
    "churn prediction. These are model explanations, not proof "
    "that a feature causes a customer to leave."
)


# -------------------------------
# 4. SELECT CUSTOMER
# -------------------------------

st.subheader("Individual Customer Explanation")

customer_ids = customers["customerID"].astype(str).tolist()

selected_id = st.selectbox(
    "Select a customer ID",
    customer_ids
)

customer_index = customers.index[
    customers["customerID"].astype(str) == selected_id
][0]

customer_raw = X.loc[[customer_index]]
customer_transformed = transformed[[customer_index]]

probability = pipeline.predict_proba(customer_raw)[0][1]

st.metric(
    "Predicted Churn Probability",
    f"{probability * 100:.2f}%"
)


# -------------------------------
# 5. CALCULATE SHAP VALUES
# -------------------------------

@st.cache_resource
def create_explainer(_classifier, model_type, background):
    if model_type == "RandomForest":
        return shap.TreeExplainer(_classifier)

    return shap.LinearExplainer(
        _classifier,
        background
    )


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
        model_type = "Other"

    if model_type == "Other":
        st.error(
            f"Unsupported model type: {type(classifier).__name__}"
        )
        st.stop()

    explainer = create_explainer(
        classifier,
        model_type,
        background
    )

    shap_result = explainer(
        customer_transformed
    )

    shap_values = np.asarray(shap_result.values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]

    base_values = np.asarray(shap_result.base_values)

    if base_values.ndim == 2:
        base_value = base_values[0, 1]
    elif base_values.ndim == 1:
        base_value = base_values[0]
    else:
        base_value = base_values.item()

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=customer_transformed[0],
        feature_names=feature_names
    )

    st.subheader("Why is this customer at risk?")

    fig = plt.figure(figsize=(12, 7))

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red features push the model output higher; "
        "blue features push it lower. The output scale depends "
        "on the model's SHAP explainer."
    )

except Exception as e:
    st.error(f"SHAP explanation failed: {e}")
    st.exception(e)
    st.stop()


# -------------------------------
# 6. GLOBAL FEATURE IMPORTANCE
# -------------------------------

st.subheader("Overall Churn Drivers")

st.write(
    "This chart shows which features have the largest average "
    "impact on model predictions across a sample of customers."
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

    global_values = np.asarray(global_result.values)

    if global_values.ndim == 3:
        global_values = global_values[:, :, 1]

    importance = np.abs(global_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value": importance
    }).sort_values(
        "Mean Absolute SHAP Value",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Mean Absolute SHAP Value"][::-1]
    )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Top 15 Features Influencing Churn Predictions")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error(f"Global feature importance failed: {e}")
    st.exception(e)
