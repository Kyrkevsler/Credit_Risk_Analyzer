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
```python
example_loan = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 15000,
    'income': 60000,
    'years_employed': 3,
    'fico_score': 650
}
```
## Model Evaluation Results

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 1630    |
| 1     | 0.99      | 0.99   | 0.99     | 370     |

(2000 samples)

### ROC AUC Score
**0.9998**

### Risk Prediction
- **Probability of default**: 22.00%
- **Expected loss**: $990.00
