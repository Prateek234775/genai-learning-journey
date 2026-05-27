# ============================================
# DAY 8 - ML Basics
# What is ML, Types, Train/Test Split,
# Overfitting & Underfitting
# Author: Prateek kumar kuntal
# Date: 12 May 2025
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# ──────────────────────────────────────────
# PART 1 - WHAT IS MACHINE LEARNING
# ──────────────────────────────────────────

print("===== PART 1: What is Machine Learning =====")

print("""
MACHINE LEARNING = Teaching computers to learn from data
                   without being explicitly programmed.

Traditional Programming:
  Input + Rules → Output

Machine Learning:
  Input + Output → Rules (Model learns the rules itself!)

3 TYPES OF ML:
──────────────
1. SUPERVISED LEARNING
   - You give data WITH correct answers
   - Model learns to predict answers for new data
   - Examples: Spam detection, House price prediction,
               Disease diagnosis

2. UNSUPERVISED LEARNING
   - You give data WITHOUT correct answers
   - Model finds hidden patterns itself
   - Examples: Customer grouping, Anomaly detection,
               Topic modelling

3. REINFORCEMENT LEARNING
   - Model learns by trial and error
   - Gets reward for good actions, penalty for bad
   - Examples: Game playing AI, Self driving cars,
               Robot navigation

TODAY we focus on SUPERVISED LEARNING!
""")


# ──────────────────────────────────────────
# PART 2 - KEY ML TERMS
# ──────────────────────────────────────────

print("===== PART 2: Key ML Terms =====")

print("""
FEATURES (X)   = Input columns used to predict
                 Example: House size, rooms, location

TARGET (y)     = Output column we want to predict
                 Example: House price

TRAINING DATA  = Data model learns from (usually 80%)

TEST DATA      = Data used to check model accuracy (20%)
                 Model has NEVER seen this data before

MODEL          = The mathematical function that maps
                 features to target after learning

PREDICTION     = Model's output for new unseen data

ACCURACY       = How correct the model's predictions are
""")


# ──────────────────────────────────────────
# PART 3 - TRAIN TEST SPLIT
# ──────────────────────────────────────────

print("===== PART 3: Train Test Split =====")

# Create sample dataset
np.random.seed(42)
n = 100

data = {
    "StudyHours"  : np.random.uniform(1, 10, n),
    "AttendancePct": np.random.uniform(60, 100, n),
    "PrevScore"   : np.random.uniform(40, 95, n),
}

df = pd.DataFrame(data)

# Target — exam score based on features
df["ExamScore"] = (
    df["StudyHours"]   * 4.5 +
    df["AttendancePct"] * 0.3 +
    df["PrevScore"]    * 0.4 +
    np.random.normal(0, 3, n)  # some noise
).clip(35, 100).round(2)

print("Dataset:")
print(df.head(10))
print(f"\nShape: {df.shape}")

# Separate features and target
X = df[["StudyHours", "AttendancePct", "PrevScore"]]
y = df["ExamScore"]

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target (y) shape   : {y.shape}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # for reproducibility
)

print(f"\nTraining set size  : {X_train.shape[0]} samples")
print(f"Testing set size   : {X_test.shape[0]} samples")
print(f"Train percentage   : {len(X_train)/len(X)*100:.0f}%")
print(f"Test percentage    : {len(X_test)/len(X)*100:.0f}%")


# ──────────────────────────────────────────
# PART 4 - TRAIN A SIMPLE MODEL
# ──────────────────────────────────────────

print("\n===== PART 4: Train a Simple Model =====")

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)   # LEARNING happens here!

print(" Model trained!")
print(f"\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:20} : {coef:.4f}")
print(f"  {'Intercept':20} : {model.intercept_:.4f}")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# Evaluate
train_r2  = r2_score(y_train, y_pred_train)
test_r2   = r2_score(y_test,  y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse  = mean_squared_error(y_test,  y_pred_test)

print(f"\n--- Model Performance ---")
print(f"Train R² Score  : {train_r2:.4f}")
print(f"Test  R² Score  : {test_r2:.4f}")
print(f"Train MSE       : {train_mse:.4f}")
print(f"Test  MSE       : {test_mse:.4f}")

print(f"\n--- Sample Predictions ---")
print(f"{'Actual':<12} {'Predicted':<12} {'Difference':<12}")
print("-" * 36)
for actual, predicted in zip(y_test[:10],
                              y_pred_test[:10]):
    diff = abs(actual - predicted)
    print(f"{actual:<12.2f} {predicted:<12.2f} {diff:<12.2f}")


# ──────────────────────────────────────────
# PART 5 - OVERFITTING & UNDERFITTING
# (Most important concept in ML!)
# ──────────────────────────────────────────

print("\n===== PART 5: Overfitting & Underfitting =====")

print("""
UNDERFITTING = Model is too simple
               Performs badly on BOTH train and test
               Like a student who didn't study at all

GOOD FIT     = Model is just right
               Performs well on BOTH train and test
               Like a student who studied properly

OVERFITTING  = Model is too complex
               Performs great on train, badly on test
               Like a student who memorized answers
               but can't solve new questions!

HOW TO DETECT:
  Train score much higher than Test score = OVERFITTING
  Both scores low                         = UNDERFITTING
  Both scores similar and high            = GOOD FIT
""")

# Demonstrate with polynomial regression
np.random.seed(42)
X_demo = np.sort(np.random.uniform(0, 10, 30))
y_demo = 2 * X_demo + np.random.normal(0, 2, 30)

X_demo_2d = X_demo.reshape(-1, 1)

results = []

for degree in [1, 3, 15]:
    poly    = PolynomialFeatures(degree=degree)
    X_poly  = poly.fit_transform(X_demo_2d)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_poly, y_demo, test_size=0.2, random_state=42)

    m = LinearRegression()
    m.fit(X_tr, y_tr)

    train_score = r2_score(y_tr, m.predict(X_tr))
    test_score  = r2_score(y_te, m.predict(X_te))

    if degree == 1:
        label = "UNDERFIT  (degree=1)"
    elif degree == 3:
        label = "GOOD FIT  (degree=3)"
    else:
        label = "OVERFIT   (degree=15)"

    results.append((label, train_score, test_score))
    print(f"{label}")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test  R²: {test_score:.4f}")
    print()


# ──────────────────────────────────────────
# PART 6 - HOW TO FIX OVERFITTING
# ──────────────────────────────────────────

print("===== PART 6: How to Fix Overfitting =====")

print("""
SOLUTIONS FOR OVERFITTING:
──────────────────────────
1. Get MORE training data
   More data = harder to memorize = better generalization

2. Reduce model complexity
   Use simpler model with fewer parameters

3. Cross Validation
   Test on multiple splits, not just one

4. Regularization (L1, L2)
   Penalizes model for being too complex
   We will learn this in coming days!

5. Dropout (for Deep Learning)
   Randomly turn off neurons during training
   We will learn this in Week 3!

SOLUTIONS FOR UNDERFITTING:
────────────────────────────
1. Use more complex model
2. Add more features
3. Train for longer
4. Reduce regularization
""")


# ──────────────────────────────────────────
# MINI PROJECT - ML Pipeline from Scratch
# ──────────────────────────────────────────

print("===== MINI PROJECT: Full ML Pipeline =====")

# Dataset — predict student exam score
np.random.seed(42)
n = 200

df_project = pd.DataFrame({
    "StudyHours"    : np.random.uniform(1, 12, n),
    "Attendance"    : np.random.uniform(50, 100, n),
    "AssignmentScore": np.random.uniform(40, 100, n),
    "SleepHours"    : np.random.uniform(4, 9, n),
    "ExamScore"     : None
})

df_project["ExamScore"] = (
    df_project["StudyHours"]     * 5.0 +
    df_project["Attendance"]     * 0.25 +
    df_project["AssignmentScore"]* 0.35 +
    df_project["SleepHours"]     * 1.5 +
    np.random.normal(0, 4, n)
).clip(35, 100).round(2)

print(f"Dataset shape    : {df_project.shape}")
print(f"\nFirst 5 rows:")
print(df_project.head())
print(f"\nStatistics:")
print(df_project.describe().round(2))

# Step 1 — Separate features and target
X = df_project.drop(columns=["ExamScore"])
y = df_project["ExamScore"]

# Step 2 — Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# Step 3 — Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model trained!")

# Step 4 — Evaluate
y_pred = model.predict(X_test)
r2     = r2_score(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)

print(f"\n--- Results ---")
print(f"R² Score  : {r2:.4f}  (1.0 = perfect)")
print(f"MSE       : {mse:.4f}")
print(f"RMSE      : {rmse:.4f}")

# Step 5 — Predict new student
new_student = pd.DataFrame({
    "StudyHours"     : [8],
    "Attendance"     : [90],
    "AssignmentScore": [85],
    "SleepHours"     : [7]
})

predicted_score = model.predict(new_student)[0]
print(f"\n--- New Student Prediction ---")
print(f"Study Hours      : 8")
print(f"Attendance       : 90%")
print(f"Assignment Score : 85")
print(f"Sleep Hours      : 7")
print(f"Predicted Score  : {predicted_score:.2f}")

# Step 6 — Feature importance
print(f"\n--- Feature Importance ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:20} : {coef:.4f}")


print("\n===== WHAT I LEARNED TODAY =====")
print(" What is ML and 3 types")
print(" Key terms - Features, Target, Model")
print(" Train Test Split - 80/20 rule")
print(" Trained first ML model!")
print(" Overfitting vs Underfitting")
print(" How to fix overfitting")
print(" Mini Project - Full ML Pipeline")
print("\n Day 8 Done! Tomorrow - Linear & Logistic Regression!")