import json
import joblib
import pandas as pd

from pathlib import Path
import argparse

# Find project folders
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


def predict_customers(input_file):

    # Load saved model and metadata
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    threshold = metadata["threshold"]

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    required_features = numeric_features + categorical_features

    # Read customer data
    df = pd.read_csv(input_file)

    print("\nTotal customers uploaded:", len(df))

    # Preserve customer IDs
    if "customerID" in df.columns:
        customer_ids = df["customerID"]
    else:
        customer_ids = pd.Series(
            [f"Customer_{i+1}" for i in range(len(df))]
        )

    # Remove target column if present
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    # Check required columns
    missing = [
        col for col in required_features
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X = X[required_features].copy()

    # Convert numeric columns
    for col in numeric_features:
        X[col] = pd.to_numeric(
            X[col].astype(str).str.strip().replace("", float("nan")),
            errors="coerce"
        )

    # Predict churn probability
    probabilities = model.predict_proba(X)[:, 1]

    # Apply selected threshold
    predictions = (probabilities >= threshold).astype(int)

    # Create results
    results = pd.DataFrame({
        "customerID": customer_ids,
        "churn_probability": probabilities.round(4),
        "predicted_churn": predictions
    })

    # Assign risk categories
    results["risk_level"] = pd.cut(
        results["churn_probability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    # Add monthly charges
    results["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    ).fillna(0).values

    # Potential monthly charges associated with flagged customers
    results["potential_monthly_charges"] = (
        results["MonthlyCharges"] * results["predicted_churn"]
    )

    # Sort highest-risk customers first
    results = results.sort_values(
        by="churn_probability",
        ascending=False
    )

    # Save results
    output_path = (
        BASE_DIR / "data" / "customer_risk_predictions.csv"
    )

    results.to_csv(output_path, index=False)

    # Display summary
    print("\n========== CUSTOMER RISK SUMMARY ==========")

    print("High Risk:",
          (results["risk_level"] == "High").sum())

    print("Medium Risk:",
          (results["risk_level"] == "Medium").sum())

    print("Low Risk:",
          (results["risk_level"] == "Low").sum())

    print("\nPredicted Churners:",
          results["predicted_churn"].sum())

    print(
        "Potential monthly charges associated with flagged customers: ₹",
        round(results["potential_monthly_charges"].sum(), 2)
    )

    print("\nTop 10 High-Risk Customers:")
    print(results.head(10).to_string(index=False))

    print("\nResults saved to:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict customer churn risk from a CSV file."
    )

    parser.add_argument(
        "input_file",
        help="Path to the customer CSV file"
    )

    args = parser.parse_args()

    predict_customers(args.input_file)