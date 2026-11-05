# Customer Churn Prediction & Customer Intelligence Dashboard

An end-to-end Machine Learning project that predicts customer churn, identifies customers at risk of leaving, and provides actionable insights to support customer retention.

## Project Overview

Customer churn is a major challenge for businesses. Retaining existing customers is often more cost-effective than acquiring new ones.

This project uses Machine Learning to predict customer churn and an interactive Streamlit dashboard to help businesses understand customer behaviour and identify customers who may need attention.

## Key Features

- Customer churn prediction using Machine Learning.
- Customer risk classification.
- Interactive customer intelligence dashboard.
- Model evaluation and performance comparison.
- SHAP-based explainable AI.
- Customer retention recommendations.
- Campaign management functionality.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SHAP
- Joblib
- SQLite

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Hyperparameter tuning using GridSearchCV

## Model Performance

Final model test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 76.72% |
| Precision | 54.79% |
| Recall | 70.32% |
| F1 Score | 61.59% |
| ROC-AUC | 84.26% |
| PR-AUC | 63.32% |

## Dataset

The project uses the Telco Customer Churn dataset containing 7,043 customer records.

## Project Structure

```text
CUSTOMER-CHURN-PREDICTION/
│
├── app.py
├── main.py
├── requirements.txt
├── shap_analysis.py
│
├── data/
├── models/
├── pages/
├── src/
│
└── .gitignore
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/kavadiyajiya-ux/Customer-Churn-Prediction.git
```

2. Navigate to the project:

```bash
cd Customer-Churn-Prediction
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the dashboard:

```bash
streamlit run app.py
```

## Project Objective

The objective is to combine predictive analytics, explainable AI, and customer intelligence to help businesses make data-driven customer retention decisions.

## Author

**Jiya Kumari**

GitHub: [@kavadiyajiya-ux](https://github.com/kavadiyajiya-ux)

---

*Built as a hands-on Machine Learning and Data Science project.*
