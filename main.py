
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Step 1: Load the dataset
df = pd.read_csv("data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Display first 5 rows
print("\nFIRST 5 CUSTOMERS:")
print(df.head())

# Step 3: Check dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Step 4: Check column information
print("\nDATASET INFORMATION:")
print(df.info())

# Step 5: Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# Step 6: Display all column names
print("\nALL COLUMN NAMES:")
print(df.columns.tolist())

# Step 7: Check customer churn distribution
print("\nCUSTOMER CHURN COUNT:")
print(df["Churn"].value_counts())

# Step 8: Check churn percentage
print("\nCUSTOMER CHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Step 9: Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# ==========================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ==========================================

# 1. Check duplicate records
print("\nDUPLICATE RECORDS:")
print("Total duplicates:", df.duplicated().sum())

# 2. Check data types
print("\nDATA TYPES:")
print(df.dtypes)

# 3. Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 4. Check missing values
print("\nMISSING VALUES BEFORE CLEANING:")
print(df.isnull().sum())

# 5. Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# 6. Verify missing values
print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# 7. Remove duplicate records if any
df = df.drop_duplicates()

# 8. Final dataset information
print("\nFINAL DATASET INFORMATION:")
print("Dataset Shape:", df.shape)
print("Total Missing Values:", df.isnull().sum().sum())
print("Total Duplicate Rows:", df.duplicated().sum())

print("\nDATA CLEANING COMPLETED!")

# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Count customers who stayed and left
print("\nCUSTOMER CHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

# 2. Calculate churn percentage
print("\nCHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 3. Visualize churn distribution
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (Yes = Left, No = Stayed)")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.2: CHURN BASED ON CONTRACT TYPE
# ==========================================

# Count churn for each contract type
print("\nCHURN BASED ON CONTRACT:")
print(pd.crosstab(df["Contract"], df["Churn"]))

# Calculate churn percentage for each contract type
print("\nCHURN PERCENTAGE BY CONTRACT:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize contract vs churn
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.show()
# ==========================================
# STEP 3.3: CHURN BASED ON TENURE
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True
)

plt.title("Customer Churn Based on Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.4: CHURN BASED ON MONTHLY CHARGES
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    bins=30,
    kde=True
)

plt.title("Customer Churn Based on Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.5: CHURN BASED ON PAYMENT METHOD
# ==========================================

# Count customers by payment method and churn
print("\nCHURN BASED ON PAYMENT METHOD:")
print(pd.crosstab(df["PaymentMethod"], df["Churn"]))

# Calculate churn percentage
print("\nCHURN PERCENTAGE BY PAYMENT METHOD:")
print(
    pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn Based on Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.show()
# ==========================================
# STEP 3.6: CHURN BASED ON INTERNET SERVICE
# ==========================================

# Count customers
print("\nCHURN BASED ON INTERNET SERVICE:")
print(pd.crosstab(df["InternetService"], df["Churn"]))

# Calculate churn percentage
print("\nCHURN PERCENTAGE BY INTERNET SERVICE:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn Based on Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.7: CHURN BASED ON SENIOR CITIZENS
# ==========================================

# Compare senior citizen status with churn
print("\nCHURN BASED ON SENIOR CITIZENS:")
print(pd.crosstab(df["SeniorCitizen"], df["Churn"]))

# Calculate percentages
print("\nCHURN PERCENTAGE:")
print(
    pd.crosstab(
        df["SeniorCitizen"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn"
)

plt.title("Customer Churn Based on Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.xticks([0, 1], ["Non-Senior Citizen", "Senior Citizen"])

plt.show()
# ==========================================
# STEP 3.8: CHURN BASED ON TECH SUPPORT
# ==========================================

# Count customers
print("\nCHURN BASED ON TECH SUPPORT:")
print(pd.crosstab(df["TechSupport"], df["Churn"]))

# Calculate churn percentages
print("\nCHURN PERCENTAGE BY TECH SUPPORT:")
print(
    pd.crosstab(
        df["TechSupport"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="TechSupport",
    hue="Churn"
)

plt.title("Customer Churn Based on Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.9: CHURN BASED ON ONLINE SECURITY
# ==========================================

# Count customers
print("\nCHURN BASED ON ONLINE SECURITY:")
print(pd.crosstab(df["OnlineSecurity"], df["Churn"]))

# Calculate churn percentages
print("\nCHURN PERCENTAGE BY ONLINE SECURITY:")
print(
    pd.crosstab(
        df["OnlineSecurity"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OnlineSecurity",
    hue="Churn"
)

plt.title("Customer Churn Based on Online Security")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 3.10: CHURN BASED ON ONLINE BACKUP
# ==========================================

# Count customers
print("\nCHURN BASED ON ONLINE BACKUP:")
print(pd.crosstab(df["OnlineBackup"], df["Churn"]))

# Calculate churn percentages
print("\nCHURN PERCENTAGE BY ONLINE BACKUP:")
print(
    pd.crosstab(
        df["OnlineBackup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize the results
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OnlineBackup",
    hue="Churn"
)

plt.title("Customer Churn Based on Online Backup")
plt.xlabel("Online Backup")
plt.ylabel("Number of Customers")

plt.show()
# ==========================================
# STEP 4.1: OVERALL CUSTOMER CHURN
# ==========================================

churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Overall Customer Churn Percentage")

plt.show()
# ==========================================
# STEP 4.2: CORRELATION HEATMAP
# ==========================================

# Select numerical columns
numerical_df = df.select_dtypes(include=["number"])

# Calculate correlation
correlation = numerical_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()
# ==========================================
# STEP 5.1: REMOVE CUSTOMER ID
# ==========================================

df = df.drop("customerID", axis=1)

print("Updated Dataset:")
print(df.head())

print("\nUpdated Shape:")
print(df.shape)
# ==========================================
# STEP 5.2: ENCODING CATEGORICAL DATA
# ==========================================

# Convert Yes/No values into 1/0
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Separate categorical columns
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

# Apply One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Display updated dataset
print("\nENCODED DATASET:")
print(df.head())

print("\nUPDATED DATA TYPES:")
print(df.dtypes)

print("\nUPDATED DATASET SHAPE:")
print(df.shape)
# ==========================================
# STEP 6: TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("\nTRAIN-TEST SPLIT:")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
# ==========================================
# STEP 7: FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Learn scaling values from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to testing data
X_test_scaled = scaler.transform(X_test)

print("\nSCALING COMPLETED!")
print("Scaled training data:", X_train_scaled.shape)
print("Scaled testing data:", X_test_scaled.shape)
# ==========================================
# STEP 8: LOGISTIC REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression

# Create the model
log_model = LogisticRegression(max_iter=1000)

# Train the model
log_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log = log_model.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION MODEL")
print("Model training completed!")
print("First 10 predictions:", y_pred_log[:10])
print("Actual values:", y_test.values[:10])
# ==========================================
# STEP 9: MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_log)
precision = precision_score(y_test, y_pred_log)
recall = recall_score(y_test, y_pred_log)
f1 = f1_score(y_test, y_pred_log)

print("\nLOGISTIC REGRESSION EVALUATION")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_log))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_log))

# ==========================================
# DECISION TREE CLASSIFIER
# ==========================================

# Step 1: Import Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Step 2: Create the model
dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Step 3: Train the model
dt_model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred_dt = dt_model.predict(X_test)

# Step 5: Evaluate the model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print("\n========== DECISION TREE RESULTS ==========")

print("Accuracy:", round(dt_accuracy * 100, 2), "%")
print("Precision:", round(dt_precision * 100, 2), "%")
print("Recall:", round(dt_recall * 100, 2), "%")
print("F1 Score:", round(dt_f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Step 6: Visualise Decision Tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X_train.columns,
    class_names=["Stay", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Customer Churn - Decision Tree")
plt.show()


# ==========================================
# RANDOM FOREST CLASSIFIER
# ==========================================

# Step 1: Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Step 2: Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Step 3: Train the model
rf_model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred_rf = rf_model.predict(X_test)

# Step 5: Evaluate the model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy:", round(rf_accuracy * 100, 2), "%")
print("Precision:", round(rf_precision * 100, 2), "%")
print("Recall:", round(rf_recall * 100, 2), "%")
print("F1 Score:", round(rf_f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# MODEL COMPARISON
# ==========================================

import pandas as pd

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        precision,
        dt_precision,
        rf_precision
    ],
    "Recall": [
        recall,
        dt_recall,
        rf_recall
    ],
    "F1 Score": [
        f1,
        dt_f1,
        rf_f1
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(model_comparison.round(4))

print("\nBest Model Based on F1 Score:")
print(
    model_comparison.loc[
        model_comparison["F1 Score"].idxmax()
    ]
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance from Random Forest
feature_importance = rf_model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importance
})

# Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance_df.head(15))

# Visualise feature importance
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features Influencing Customer Churn")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ==========================================
# FEATURE SELECTION - TOP 15 FEATURES
# ==========================================

# Select top 15 features
top_15_features = importance_df.head(15)["Feature"].tolist()

# Create new datasets with selected features
X_train_selected = X_train[top_15_features]
X_test_selected = X_test[top_15_features]

# Train Random Forest using selected features
rf_selected = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_selected.fit(X_train_selected, y_train)

# Predictions
y_pred_selected = rf_selected.predict(X_test_selected)

# Evaluate performance
print("\n========== FEATURE SELECTION RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_selected))
print("Precision:", precision_score(y_test, y_pred_selected))
print("Recall:", recall_score(y_test, y_pred_selected))
print("F1 Score:", f1_score(y_test, y_pred_selected))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_selected))

# ==========================================
# HYPERPARAMETER TUNING - GRID SEARCH CV
# ==========================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Create Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define parameters to test
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Train using training data only
grid_search.fit(X_train, y_train)

# Best parameters
print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

# Best cross-validation F1 score
print("\nBest Cross-Validation F1 Score:")
print(grid_search.best_score_)

# Get best model
best_rf_model = grid_search.best_estimator_

# Predict on test data
y_pred_tuned = best_rf_model.predict(X_test)

# Evaluate tuned model
print("\n========== TUNED RANDOM FOREST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))
print("F1 Score:", f1_score(y_test, y_pred_tuned))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))

# ==========================================
# ROC-AUC EVALUATION
# ==========================================

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get churn probabilities from both models

y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC scores

log_auc = roc_auc_score(y_test, y_prob_log)

rf_auc = roc_auc_score(y_test, y_prob_rf)

# Print results

print("\n========== ROC-AUC SCORES ==========")

print("Logistic Regression ROC-AUC:", round(log_auc, 4))

print("Tuned Random Forest ROC-AUC:", round(rf_auc, 4))

# Calculate ROC curve values

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Plot ROC curve

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {log_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Tuned Random Forest (AUC = {rf_auc:.3f})"
)

# Random guessing reference line

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Customer Churn Prediction")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Get prediction probabilities
y_prob_lr = log_model.predict_proba(X_test)[:, 1]
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.01)


def optimize_threshold(y_true, y_prob):

    results = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append([threshold, precision, recall, f1])

    return np.array(results)


# Calculate results
lr_results = optimize_threshold(y_test, y_prob_lr)
rf_results = optimize_threshold(y_test, y_prob_rf)


# Find best thresholds
best_lr = lr_results[np.argmax(lr_results[:, 3])]
best_rf_threshold = rf_results[np.argmax(rf_results[:, 3])]


print("\n========== THRESHOLD OPTIMIZATION ==========")

print("\nLOGISTIC REGRESSION")
print("Best Threshold:", best_lr[0])
print("Precision:", best_lr[1])
print("Recall:", best_lr[2])
print("F1 Score:", best_lr[3])

print("\nRANDOM FOREST")
print("Best Threshold:", best_rf_threshold[0])
print("Precision:", best_rf_threshold[1])
print("Recall:", best_rf_threshold[2])
print("F1 Score:", best_rf_threshold[3])


# Plot F1 comparison
plt.figure(figsize=(10, 6))

plt.plot(lr_results[:, 0], lr_results[:, 3],
         label="Logistic Regression")

plt.plot(rf_results[:, 0], rf_results[:, 3],
         label="Random Forest")

plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold Optimization Comparison")

plt.legend()
plt.grid()
plt.show()
import joblib

# Save trained Random Forest model
joblib.dump(best_rf_model, "best_rf_model.pkl")

# Save test data
joblib.dump(X_test, "X_test.pkl")

print("\nModel and test data saved successfully!")