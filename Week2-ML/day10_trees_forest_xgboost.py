# ============================================
# DAY 10 - Decision Trees + Random Forest
#           + XGBoost
# Author: Prateek Kumar Kuntal
# Date: 14 May 2025
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────
# PART 1 - DECISION TREE
# ──────────────────────────────────────────

print("===== PART 1: Decision Tree =====")

print("""
DECISION TREE:
  Works like a flowchart of YES/NO questions
  Splits data based on best feature at each step
  Easy to understand and visualize

Example — Should I play cricket today?
  Is it raining?
    YES → Don't play
    NO  → Is it too hot?
            YES → Don't play
            NO  → PLAY!

PROS:
   Easy to understand and explain
   No feature scaling needed
   Handles both numbers and categories

CONS:
   Overfits easily
   Small change in data = very different tree
   Not great accuracy alone
""")

# Create dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Age"          : np.random.randint(18, 65, n),
    "Income"       : np.random.uniform(15000, 200000, n),
    "CreditScore"  : np.random.randint(300, 850, n),
    "LoanAmount"   : np.random.uniform(10000, 500000, n),
    "ExistingLoans": np.random.randint(0, 5, n),
    "Experience"   : np.random.randint(0, 40, n),
})

# Target
score = (
    df["CreditScore"] * 0.45 +
    df["Income"]      * 0.0003 +
    df["Experience"]  * 2.5 -
    df["ExistingLoans"] * 15 -
    df["LoanAmount"]  * 0.00004
)
df["Approved"] = (score > score.median()).astype(int)

print("Dataset:")
print(df.head())
print(f"\nShape      : {df.shape}")
print(f"Approved   : {df['Approved'].sum()}")
print(f"Rejected   : {(df['Approved']==0).sum()}")

# Split
X = df.drop(columns=["Approved"])
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5,        # limit depth to avoid overfitting
    random_state=42
)
dt.fit(X_train, y_train)

# Evaluate
y_pred_dt = dt.predict(X_test)
acc_dt    = accuracy_score(y_test, y_pred_dt)

print(f"\n--- Decision Tree Results ---")
print(f"Accuracy     : {acc_dt*100:.2f}%")
print(f"Max Depth    : {dt.max_depth}")
print(f"Tree Depth   : {dt.get_depth()}")
print(f"Leaves       : {dt.get_n_leaves()}")

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in sorted(
    zip(X.columns, dt.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15} : {bar} {imp:.4f}")

# Overfitting check
train_acc = accuracy_score(y_train, dt.predict(X_train))
print(f"\nTrain Accuracy : {train_acc*100:.2f}%")
print(f"Test  Accuracy : {acc_dt*100:.2f}%")
if train_acc - acc_dt > 0.05:
    print("  Overfitting detected!")
else:
    print(" Good fit!")


# ──────────────────────────────────────────
# PART 2 - RANDOM FOREST
# ──────────────────────────────────────────

print("\n===== PART 2: Random Forest =====")

print("""
RANDOM FOREST:
  Collection of many Decision Trees
  Each tree trained on random subset of data
  Final prediction = majority vote of all trees

  Like asking 100 doctors instead of 1!
  Each doctor sees different patient info
  Final diagnosis = what most doctors say

WHY BETTER THAN SINGLE TREE?
   Reduces overfitting dramatically
   More accurate and stable
   Handles missing values well
   Works great out of the box

PROS:
   High accuracy
   Robust to outliers
   Feature importance built in

CONS:
   Slower than single tree
   Harder to interpret
   More memory needed
""")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test)
acc_rf    = accuracy_score(y_test, y_pred_rf)

print(f"--- Random Forest Results ---")
print(f"Accuracy      : {acc_rf*100:.2f}%")
print(f"Trees used    : {rf.n_estimators}")

# Overfitting check
train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
print(f"\nTrain Accuracy : {train_acc_rf*100:.2f}%")
print(f"Test  Accuracy : {acc_rf*100:.2f}%")
if train_acc_rf - acc_rf > 0.05:
    print("  Overfitting detected!")
else:
    print(" Good fit!")

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in sorted(
    zip(X.columns, rf.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15} : {bar} {imp:.4f}")


# ──────────────────────────────────────────
# PART 3 - XGBOOST
# ──────────────────────────────────────────

print("\n===== PART 3: XGBoost =====")

print("""
XGBOOST (Extreme Gradient Boosting):
  One of the most powerful ML algorithms!
  Used in almost every Kaggle competition winner

HOW IT WORKS:
  1. Train a weak tree
  2. Look at mistakes
  3. Train next tree to fix mistakes
  4. Repeat 100s of times
  5. All trees combined = super powerful model

BOOSTING vs BAGGING:
  Random Forest = BAGGING
    Trees trained independently in parallel
    Final = majority vote

  XGBoost = BOOSTING
    Trees trained sequentially
    Each fixes previous tree mistakes
    Final = weighted sum of all trees

PROS:
   Extremely high accuracy
   Handles missing values automatically
   Built in regularization
   Industry standard for tabular data

CONS:
   Many hyperparameters to tune
   Slower to train than Random Forest
   Can overfit if not tuned properly
""")

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
xgb.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb.predict(X_test)
acc_xgb    = accuracy_score(y_test, y_pred_xgb)

print(f"--- XGBoost Results ---")
print(f"Accuracy      : {acc_xgb*100:.2f}%")

# Overfitting check
train_acc_xgb = accuracy_score(y_train, xgb.predict(X_train))
print(f"\nTrain Accuracy : {train_acc_xgb*100:.2f}%")
print(f"Test  Accuracy : {acc_xgb*100:.2f}%")
if train_acc_xgb - acc_xgb > 0.05:
    print("  Overfitting detected!")
else:
    print(" Good fit!")

# Feature importance
print(f"\nFeature Importance:")
xgb_imp = xgb.feature_importances_
for feat, imp in sorted(
    zip(X.columns, xgb_imp),
    key=lambda x: x[1], reverse=True
):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15} : {bar} {imp:.4f}")


# ──────────────────────────────────────────
# PART 4 - MODEL COMPARISON
# ──────────────────────────────────────────

print("\n===== PART 4: Model Comparison =====")

models = {
    "Decision Tree" : y_pred_dt,
    "Random Forest" : y_pred_rf,
    "XGBoost"       : y_pred_xgb,
}

print(f"{'Model':<18} {'Accuracy':>10} {'Winner'}")
print("-" * 40)

best_acc   = 0
best_model = ""

for name, pred in models.items():
    acc = accuracy_score(y_test, pred) * 100
    if acc > best_acc:
        best_acc   = acc
        best_model = name
    print(f"{name:<18} {acc:>9.2f}%")

print(f"\ Best Model: {best_model} ({best_acc:.2f}%)")

print(f"\n--- Detailed Report: {best_model} ---")
best_pred = models[best_model]
print(classification_report(y_test, best_pred))


# ──────────────────────────────────────────
# PART 5 - WHEN TO USE WHICH
# ──────────────────────────────────────────

print("===== PART 5: When to Use Which =====")

print("""
DECISION TREE:
   When you need to explain the model
   Small datasets
   Quick baseline model

RANDOM FOREST:
   When accuracy matters more than speed
   Medium to large datasets
   When you have many features
   Good default choice for most problems

XGBOOST:
   When you need maximum accuracy
   Kaggle competitions
   Tabular/structured data
   When you have time to tune parameters
   Industry standard for production
""")


# ──────────────────────────────────────────
# MINI PROJECT - Student Dropout Predictor
# ──────────────────────────────────────────

print("===== MINI PROJECT: Student Dropout Predictor =====")

np.random.seed(42)
n = 600

df_student = pd.DataFrame({
    "Age"            : np.random.randint(17, 25, n),
    "Attendance"     : np.random.uniform(30, 100, n),
    "CGPA"           : np.random.uniform(4.0, 10.0, n),
    "Backlogs"       : np.random.randint(0, 8, n),
    "FamilyIncome"   : np.random.uniform(10000, 200000, n),
    "StudyHours"     : np.random.uniform(0, 10, n),
    "ParticipationScore": np.random.uniform(0, 100, n),
})

# Dropout logic
dropout_score = (
    df_student["Attendance"]        * -0.4 +
    df_student["CGPA"]              * -5 +
    df_student["Backlogs"]          * 8 +
    df_student["FamilyIncome"]      * -0.00005 +
    df_student["StudyHours"]        * -3 +
    df_student["ParticipationScore"]* -0.2
)
df_student["Dropout"] = (
    dropout_score > dropout_score.median()
).astype(int)

print(f"Dataset Shape   : {df_student.shape}")
print(f"Dropout Count   : {df_student['Dropout'].sum()}")
print(f"Retained Count  : {(df_student['Dropout']==0).sum()}")
print(f"\nFirst 5 rows:")
print(df_student.head())

# Split
X = df_student.drop(columns=["Dropout"])
y = df_student["Dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train all 3 models
dt_m  = DecisionTreeClassifier(max_depth=5, random_state=42)
rf_m  = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_m = XGBClassifier(n_estimators=100, random_state=42,
                       eval_metric="logloss", verbosity=0)

dt_m.fit(X_train, y_train)
rf_m.fit(X_train, y_train)
xgb_m.fit(X_train, y_train)

print(f"\n--- Model Comparison ---")
print(f"{'Model':<18} {'Accuracy':>10}")
print("-" * 30)
for name, model in [("Decision Tree", dt_m),
                     ("Random Forest", rf_m),
                     ("XGBoost", xgb_m)]:
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<18} {acc*100:>9.2f}%")

# Best model prediction
print(f"\n--- Predict New Student ---")
new_student = pd.DataFrame({
    "Age"               : [19],
    "Attendance"        : [55],
    "CGPA"              : [5.5],
    "Backlogs"          : [4],
    "FamilyIncome"      : [25000],
    "StudyHours"        : [2],
    "ParticipationScore": [30],
})

prediction  = xgb_m.predict(new_student)[0]
probability = xgb_m.predict_proba(new_student)[0]

print(f"Attendance       : 55%")
print(f"CGPA             : 5.5")
print(f"Backlogs         : 4")
print(f"Study Hours      : 2/day")
print(f"\nDropout Risk     : {probability[1]*100:.2f}%")
print(f"Prediction       : {'⚠️  HIGH DROPOUT RISK' if prediction == 1 else '✅ LIKELY TO CONTINUE'}")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Decision Trees - flowchart of questions")
print(" Random Forest - 100 trees voting together")
print(" XGBoost - sequential boosting powerhouse")
print(" Model comparison - which works best")
print(" Feature importance - what matters most")
print(" Mini Project - Student Dropout Predictor")
print("\n Day 10 Done! Tomorrow - Math Basics for ML!")