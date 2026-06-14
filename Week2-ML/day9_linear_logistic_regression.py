# ============================================
# DAY 9 - Linear & Logistic Regression
# Author: Prateek kumar kuntal
# Date: 13 May 2026
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────
# PART 1 - LINEAR REGRESSION
# (Predicting a NUMBER)
# ──────────────────────────────────────────

print("===== PART 1: Linear Regression =====")

print("""
LINEAR REGRESSION:
  Used when target is a NUMBER
  Examples:
    Predict house price
    Predict exam score
    Predict temperature
    Predict salary

Formula:
  y = mx + c
  y = w1x1 + w2x2 + ... + b

  y  = prediction
  w  = weights (learned by model)
  x  = features
  b  = bias/intercept
""")

# Create dataset — predict house price
np.random.seed(42)
n = 200

df_house = pd.DataFrame({
    "Size"      : np.random.uniform(500, 3000, n),
    "Bedrooms"  : np.random.randint(1, 6, n),
    "Age"       : np.random.randint(1, 30, n),
    "Distance"  : np.random.uniform(1, 20, n),
})

# Price formula
df_house["Price"] = (
    df_house["Size"]     * 3000 +
    df_house["Bedrooms"] * 200000 +
    df_house["Age"]      * -50000 +
    df_house["Distance"] * -100000 +
    np.random.normal(0, 50000, n)
).clip(500000, 15000000).round(-3)

print("House Dataset:")
print(df_house.head())
print(f"\nShape: {df_house.shape}")

# Features and target
X = df_house.drop(columns=["Price"])
y = df_house["Price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Linear Regression Results ---")
print(f"R² Score  : {r2:.4f}")
print(f"RMSE      : ₹{rmse:,.0f}")

print(f"\n--- Feature Coefficients ---")
for feature, coef in zip(X.columns, lr.coef_):
    print(f"  {feature:12} : {coef:,.2f}")

print(f"\n--- Sample Predictions ---")
print(f"{'Actual Price':<18} {'Predicted Price':<18} {'Difference'}")
print("-" * 55)
for actual, pred in zip(y_test[:8], y_pred[:8]):
    diff = abs(actual - pred)
    print(f"₹{actual:<17,.0f} ₹{pred:<17,.0f} ₹{diff:,.0f}")

# Predict new house
new_house = pd.DataFrame({
    "Size"     : [1500],
    "Bedrooms" : [3],
    "Age"      : [5],
    "Distance" : [8]
})
predicted_price = lr.predict(new_house)[0]
print(f"\n--- New House Prediction ---")
print(f"Size: 1500 sqft | Bedrooms: 3 | Age: 5yrs | Distance: 8km")
print(f"Predicted Price: ₹{predicted_price:,.0f}")


# ──────────────────────────────────────────
# PART 2 - EVALUATION METRICS (Regression)
# ──────────────────────────────────────────

print("\n===== PART 2: Regression Metrics =====")

print("""
R² SCORE (R-squared):
  Range: 0 to 1 (higher is better)
  1.0 = perfect prediction
  0.0 = model predicts nothing useful
  Rule: above 0.7 is generally good

MSE (Mean Squared Error):
  Average of squared differences
  Penalizes large errors heavily
  Lower is better

RMSE (Root Mean Squared Error):
  Square root of MSE
  Same unit as target (easy to interpret)
  Lower is better

MAE (Mean Absolute Error):
  Average of absolute differences
  Less sensitive to outliers
  Lower is better
""")

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : ₹{mae:,.0f}")
print(f"MSE      : ₹{mse:,.0f}")
print(f"RMSE     : ₹{rmse:,.0f}")


# ──────────────────────────────────────────
# PART 3 - LOGISTIC REGRESSION
# (Predicting a CATEGORY)
# ──────────────────────────────────────────

print("\n===== PART 3: Logistic Regression =====")

print("""
LOGISTIC REGRESSION:
  Used when target is a CATEGORY
  Examples:
    Pass or Fail (0 or 1)
    Spam or Not Spam (0 or 1)
    Disease or No Disease (0 or 1)

  Despite name it is CLASSIFICATION not regression!

  Uses Sigmoid function to output probability (0 to 1)
  If probability > 0.5 → Class 1
  If probability < 0.5 → Class 0
""")

# Create dataset — predict pass/fail
np.random.seed(42)
n = 300

df_student = pd.DataFrame({
    "StudyHours"    : np.random.uniform(1, 12, n),
    "Attendance"    : np.random.uniform(50, 100, n),
    "AssignmentScore": np.random.uniform(40, 100, n),
    "PrevScore"     : np.random.uniform(35, 100, n),
})

# Pass if weighted score > 60
score = (
    df_student["StudyHours"]     * 3 +
    df_student["Attendance"]     * 0.4 +
    df_student["AssignmentScore"]* 0.3 +
    df_student["PrevScore"]      * 0.3
)
df_student["Pass"] = (score > score.median()).astype(int)

print("Student Dataset:")
print(df_student.head())
print(f"\nPass/Fail distribution:")
print(df_student["Pass"].value_counts())

# Features and target
X = df_student.drop(columns=["Pass"])
y = df_student["Pass"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features (important for logistic regression!)
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predict
y_pred_log = log_reg.predict(X_test_scaled)
y_prob     = log_reg.predict_proba(X_test_scaled)


print(f"\n Logistic Regression trained!")

# ──────────────────────────────────────────
# PART 4 - EVALUATION METRICS (Classification)
# ──────────────────────────────────────────

print("\n===== PART 4: Classification Metrics =====")

print("""
ACCURACY:
  Correct predictions / Total predictions
  Good when classes are balanced

CONFUSION MATRIX:
                 Predicted 0   Predicted 1
  Actual 0    [ True Neg (TN)  False Pos (FP) ]
  Actual 1    [ False Neg (FN) True Pos (TP)  ]

PRECISION:
  Of all predicted positives how many are correct?
  TP / (TP + FP)
  Important when False Positive is costly
  Example: Spam filter (don't mark real email as spam)

RECALL (Sensitivity):
  Of all actual positives how many did we catch?
  TP / (TP + FN)
  Important when False Negative is costly
  Example: Disease detection (don't miss sick patients)

F1 SCORE:
  Balance between Precision and Recall
  2 * (Precision * Recall) / (Precision + Recall)
  Best metric when classes are imbalanced
""")

accuracy = accuracy_score(y_test, y_pred_log)
cm       = confusion_matrix(y_test, y_pred_log)
report   = classification_report(y_test, y_pred_log)

print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nTN: {cm[0][0]}  FP: {cm[0][1]}")
print(f"FN: {cm[1][0]}  TP: {cm[1][1]}")
print(f"\nClassification Report:")
print(report)

# Probability predictions
print("--- Probability Predictions (first 10) ---")
print(f"{'Actual':<10} {'Predicted':<12} {'Prob Pass':<12} {'Prob Fail'}")
print("-" * 48)
for actual, pred, prob in zip(y_test[:10],
                               y_pred_log[:10],
                               y_prob[:10]):
    print(f"{actual:<10} {pred:<12} {prob[1]:<12.4f} {prob[0]:.4f}")


# ──────────────────────────────────────────
# PART 5 - WHEN TO USE WHICH
# ──────────────────────────────────────────

print("\n===== PART 5: When to Use Which =====")

print("""
USE LINEAR REGRESSION WHEN:
   House prices, salaries, temperatures
   Target is continuous number
 You want to understand relationships

USE LOGISTIC REGRESSION WHEN:
   Target is category (yes/no, 0/1)
   Spam detection, disease prediction
   You need probability scores
   Simple and interpretable model needed

BOTH ARE:
   Fast to train
   Easy to interpret
   Good baseline models
   Work well on small datasets
""")


# ──────────────────────────────────────────
# MINI PROJECT - Loan Approval Predictor
# ──────────────────────────────────────────

print("===== MINI PROJECT: Loan Approval Predictor =====")

np.random.seed(42)
n = 500

df_loan = pd.DataFrame({
    "Age"          : np.random.randint(22, 60, n),
    "Income"       : np.random.uniform(20000, 200000, n),
    "LoanAmount"   : np.random.uniform(50000, 1000000, n),
    "CreditScore"  : np.random.randint(300, 850, n),
    "Experience"   : np.random.randint(0, 30, n),
    "ExistingLoans": np.random.randint(0, 5, n),
})

# Approval logic
approval_score = (
    df_loan["CreditScore"] * 0.4 +
    df_loan["Income"]      * 0.0003 +
    df_loan["Experience"]  * 2 +
    df_loan["Age"]         * 0.5 -
    df_loan["ExistingLoans"] * 10 -
    df_loan["LoanAmount"]  * 0.00005
)
df_loan["Approved"] = (
    approval_score > approval_score.median()
).astype(int)

print(f"Loan Dataset Shape : {df_loan.shape}")
print(f"\nApproval Distribution:")
print(df_loan["Approved"].value_counts())
print(f"\nFirst 5 rows:")
print(df_loan.head())

# Split
X = df_loan.drop(columns=["Approved"])
y = df_loan["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred   = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Results ---")
print(f"Accuracy         : {accuracy*100:.2f}%")
print(f"Confusion Matrix :\n{cm}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict new applicant
new_applicant = pd.DataFrame({
    "Age"          : [35],
    "Income"       : [75000],
    "LoanAmount"   : [300000],
    "CreditScore"  : [720],
    "Experience"   : [10],
    "ExistingLoans": [1]
})

new_scaled   = scaler.transform(new_applicant)
prediction   = model.predict(new_scaled)[0]
probability  = model.predict_proba(new_scaled)[0]

print(f"\n--- New Applicant Prediction ---")
print(f"Age          : 35")
print(f"Income       : ₹75,000")
print(f"Loan Amount  : ₹3,00,000")
print(f"Credit Score : 720")
print(f"Experience   : 10 years")
print(f"Existing Loans: 1")
print(f"\nApproval Probability : {probability[1]*100:.2f}%")
print(f"Decision             : {'✅ APPROVED' if prediction == 1 else '❌ REJECTED'}")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Linear Regression - predict numbers")
print(" Logistic Regression - predict categories")
print(" Regression metrics - R², RMSE, MAE")
print(" Classification metrics - Accuracy, F1")
print(" Confusion Matrix - TP, TN, FP, FN")
print(" Feature Scaling with StandardScaler")
print(" Mini Project - Loan Approval Predictor")
print("\n Day 9 Done! Tomorrow - Decision Trees & Random Forest!")