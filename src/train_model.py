import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# 1. Load dataset
DATA_PATH = "data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Define features
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

required_columns = (
    numeric_features
    + categorical_features
    + ["customerID", "Churn"]
)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# 3. Clean numeric columns
for col in numeric_features:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.strip().replace("", np.nan),
        errors="coerce"
    )

# 4. Separate features and target
X = df[numeric_features + categorical_features].copy()

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError("Unexpected values found in the Churn column.")

print("\nChurn distribution:")
print(y.value_counts())

# 5. Split into training, validation and testing sets
# 60% training, 20% validation, 20% testing

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nData split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# 6. Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# 7. Logistic Regression pipeline
logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# 8. Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# 9. Tune Random Forest using training data only
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("\nStarting Random Forest hyperparameter tuning...")

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

# 10. Train Logistic Regression
print("\nTraining Logistic Regression...")

logistic_pipeline.fit(X_train, y_train)

# 11. Find the best threshold using validation data
def find_best_threshold(model, X_validation, y_validation):

    probabilities = model.predict_proba(X_validation)[:, 1]

    best_threshold = 0.50
    best_f1 = 0

    for threshold in np.arange(0.10, 0.91, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": best_rf_model
}

validation_results = {}

print("\nVALIDATION RESULTS")

for name, model in models.items():

    threshold, validation_f1 = find_best_threshold(
        model,
        X_val,
        y_val
    )

    validation_results[name] = {
        "model": model,
        "threshold": threshold,
        "f1": validation_f1
    }

    print(f"\n{name}")
    print("Best threshold:", round(threshold, 2))
    print("Validation F1:", round(validation_f1, 4))

# 12. Select the model using validation F1
best_model_name = max(
    validation_results,
    key=lambda name: validation_results[name]["f1"]
)

final_model = validation_results[best_model_name]["model"]
final_threshold = validation_results[best_model_name]["threshold"]

print("\nSelected model:", best_model_name)
print("Selected threshold:", round(final_threshold, 2))

# 13. Evaluate once on untouched test data
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= final_threshold
).astype(int)

test_metrics = {
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(y_test, test_predictions, zero_division=0),
    "recall": recall_score(y_test, test_predictions, zero_division=0),
    "f1_score": f1_score(y_test, test_predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, test_probabilities),
    "pr_auc": average_precision_score(y_test, test_probabilities)
}

print("\n========== FINAL TEST RESULTS ==========")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# 14. Save the complete model pipeline
os.makedirs("models", exist_ok=True)

joblib.dump(
    final_model,
    "models/churn_pipeline.pkl"
)

# 15. Save model information and threshold
metadata = {
    "model_name": best_model_name,
    "threshold": float(final_threshold),
    "validation_f1": float(
        validation_results[best_model_name]["f1"]
    ),
    "test_metrics": {
        key: float(value)
        for key, value in test_metrics.items()
    },
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

with open("models/model_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\nModel saved: models/churn_pipeline.pkl")
print("Metadata saved: models/model_metadata.json")
print("\nTRAINING COMPLETED SUCCESSFULLY!")