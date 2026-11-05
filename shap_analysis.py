import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load saved model and test data
best_model = joblib.load("best_rf_model.pkl")
X_test = joblib.load("X_test.pkl")

print("Model and test data loaded successfully!")

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Handle binary classification output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# SHAP summary plot
shap.summary_plot(shap_values, X_test)

print("SHAP analysis completed!")
# SHAP Feature Importance
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)

plt.title("SHAP Feature Importance")
plt.show()
# Select one customer
customer_index = 0

# Get SHAP values for that customer
customer_shap = shap_values[customer_index]

# Get base value
base_value = explainer.expected_value

if isinstance(base_value, (list, tuple, np.ndarray)):
    base_value = np.asarray(base_value).flatten()[-1]

# Create SHAP explanation
explanation = shap.Explanation(
    values=customer_shap,
    base_values=float(base_value),
    data=X_test.iloc[customer_index].values,
    feature_names=X_test.columns.tolist()
)

# Waterfall plot
shap.plots.waterfall(explanation)

plt.show()
print("\nMODEL INPUT FEATURES:")
print(X_test.columns.tolist())

print("\nNUMBER OF FEATURES:")
print(X_test.shape[1])
for i, feature in enumerate(X_test.columns):
    print(i + 1, ":", feature)