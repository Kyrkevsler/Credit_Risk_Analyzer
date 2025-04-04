# Loan Default Predictor

A machine learning pipeline for predicting loan defaults and calculating expected loss using either Logistic Regression or Random Forest classifiers.

## Features

- Loads and preprocesses loan data
- Splits data into training and test sets (stratified)
- Trains either Logistic Regression or Random Forest model
- Predicts probability of default for individual loans
- Calculates expected loss using:
  - Probability of Default (PD)
  - Exposure at Default (EAD)
  - Loss Given Default (LGD)

## Usage

### Basic Example

```python
from loan_default_predictor import LoanDefaultPredictor
```

# Initialize and train the model
predictor = LoanDefaultPredictor('loan_data.csv')
predictor.preprocess_data()
predictor.train_model('random_forest')  # or 'logistic_regression'

# Predict for a sample loan
example_loan = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 15000,
    'income': 60000,
    'years_employed': 3,
    'fico_score': 650
}

## Model Evaluation Results

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.92      | 0.90   | 0.95     | 1600    |
| 1     | 0.85      | 0.58   | 0.69     | 400     |

**Overall Accuracy**: 0.91 (2000 samples)

**Averages**:
- Macro Avg: Precision=0.88, Recall=0.78, F1-Score=0.82
- Weighted Avg: Precision=0.91, Recall=0.91, F1-Score=0.90

### ROC AUC Score
**0.9437**

### Risk Prediction
- **Probability of default**: 18.35%
- **Expected loss**: $825.68
