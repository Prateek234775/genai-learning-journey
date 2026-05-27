# ============================================
# DAY 13 - Titanic ML Model
# Full End to End ML Pipeline
# Author: Prateek Kumar Kuntal
# Date: 17 May 2025
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────
# PART 1 - LOAD & EXPLORE DATA
# ──────────────────────────────────────────

print("===== PART 1: Load & Explore Data =====")

df = pd.read_csv("train.csv")

print(f"Dataset Shape    : {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nColumn Info:")
df.info()

print(f"\nStatistics:")
print(df.describe())

print(f"\nNull Values:")
print(df.isnull().sum())

print(f"\nNull Percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

print(f"\nSurvival Distribution:")
print(df["Survived"].value_counts())
print(f"Survival Rate    : {df['Survived'].mean()*100:.2f}%")


# ──────────────────────────────────────────
# PART 2 - EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────

print("\n===== PART 2: Exploratory Data Analysis =====")

# Survival by gender
print("--- Survival by Gender ---")
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(gender_survival.round(2))

# Survival by class
print("\n--- Survival by Passenger Class ---")
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(class_survival.round(2))

# Survival by age group
df["AgeGroup"] = pd.cut(df["Age"],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen",
                                 "Adult", "Middle", "Senior"])
print("\n--- Survival by Age Group ---")
age_survival = df.groupby("AgeGroup")["Survived"].mean() * 100
print(age_survival.round(2))

# Average fare by class
print("\n--- Average Fare by Class ---")
print(df.groupby("Pclass")["Fare"].mean().round(2))

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
print("\n--- Survival by Family Size ---")
print(df.groupby("FamilySize")["Survived"].mean().round(2))


# ──────────────────────────────────────────
# PART 3 - FEATURE ENGINEERING
# ──────────────────────────────────────────

print("\n===== PART 3: Feature Engineering =====")

print("""
FEATURE ENGINEERING:
  Creating new features from existing ones
  to help model learn better patterns

  This is where domain knowledge + creativity
  separates good data scientists from great ones!
""")

def engineer_features(df):
    df = df.copy()

    # 1. Fill missing Age with median by class
    df["Age"] = df.groupby("Pclass")["Age"].transform(
        lambda x: x.fillna(x.median()))

    # 2. Fill missing Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0])

    # 3. Fill missing Fare with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # 4. Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 5. Is alone?
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 6. Title from name
    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')
    title_map = {
        "Mr"      : "Mr",
        "Miss"    : "Miss",
        "Mrs"     : "Mrs",
        "Master"  : "Master",
    }
    df["Title"] = df["Title"].map(title_map).fillna("Other")

    # 7. Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # 8. Fare groups
    df["FareGroup"] = pd.qcut(
        df["Fare"],
        q=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

    # 9. Encode categorical
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    df["Title"]    = le.fit_transform(df["Title"])

    return df

df_eng = engineer_features(df)

print("Engineered Features:")
print(df_eng[["Name", "Title", "FamilySize",
              "IsAlone", "AgeGroup", "FareGroup"]].head(10))


# ──────────────────────────────────────────
# PART 4 - PREPARE DATA FOR ML
# ──────────────────────────────────────────

print("\n===== PART 4: Prepare Data =====")

# Select features
features = ["Pclass", "Sex", "Age", "Fare",
            "SibSp", "Parch", "Embarked",
            "FamilySize", "IsAlone", "Title",
            "AgeGroup", "FareGroup"]

X = df_eng[features]
y = df_eng["Survived"]

print(f"Features used    : {features}")
print(f"X shape          : {X.shape}")
print(f"y shape          : {y.shape}")
print(f"\nNull check       : {X.isnull().sum().sum()} nulls")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTrain size       : {len(X_train)}")
print(f"Test size        : {len(X_test)}")

# Scale
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ──────────────────────────────────────────
# PART 5 - TRAIN MULTIPLE MODELS
# ──────────────────────────────────────────

print("\n===== PART 5: Train Multiple Models =====")

models = {
    "Logistic Regression" : LogisticRegression(random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(
                                max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(
                                n_estimators=100, random_state=42),
    "XGBoost"             : XGBClassifier(
                                n_estimators=100, random_state=42,
                                eval_metric="logloss", verbosity=0),
}

results = {}

for name, model in models.items():
    # Use scaled data for LR, original for trees
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc          = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "pred": y_pred, "acc": acc}
    print(f" {name:<22} trained | Accuracy: {acc*100:.2f}%")


# ──────────────────────────────────────────
# PART 6 - COMPARE & EVALUATE MODELS
# ──────────────────────────────────────────

print("\n===== PART 6: Model Comparison =====")

print(f"\n{'Model':<25} {'Accuracy':>10} {'Rank'}")
print("-" * 45)

sorted_results = sorted(results.items(),
                         key=lambda x: x[1]["acc"],
                         reverse=True)

for rank, (name, result) in enumerate(sorted_results, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{medal} {name:<23} {result['acc']*100:>9.2f}%")

# Best model details
best_name, best_result = sorted_results[0]
print(f"\n🏆 Best Model: {best_name}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_result["pred"])
print(cm)
print(f"\nTN: {cm[0][0]}  FP: {cm[0][1]}")
print(f"FN: {cm[1][0]}  TP: {cm[1][1]}")
print(f"\nClassification Report:")
print(classification_report(y_test, best_result["pred"],
                             target_names=["Died", "Survived"]))


# ──────────────────────────────────────────
# PART 7 - CROSS VALIDATION
# ──────────────────────────────────────────

print("===== PART 7: Cross Validation =====")

print("""
CROSS VALIDATION:
  Test model on multiple different splits
  More reliable than single train/test split

  K-Fold CV:
  Split data into K parts (folds)
  Train on K-1 folds, test on 1 fold
  Repeat K times, average the scores

  Why use it?
  Single split might get lucky or unlucky
  CV gives more stable estimate of performance
""")

best_model = results["Random Forest"]["model"]

cv_scores = cross_val_score(
    best_model, X, y,
    cv=5,               # 5-fold cross validation
    scoring="accuracy"
)

print(f"5-Fold CV Scores  : {cv_scores.round(4)}")
print(f"Mean CV Score     : {cv_scores.mean()*100:.2f}%")
print(f"Std CV Score      : {cv_scores.std()*100:.2f}%")
print(f"Min CV Score      : {cv_scores.min()*100:.2f}%")
print(f"Max CV Score      : {cv_scores.max()*100:.2f}%")


# ──────────────────────────────────────────
# PART 8 - FEATURE IMPORTANCE
# ──────────────────────────────────────────

print("\n===== PART 8: Feature Importance =====")

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_

print("Feature Importance (Random Forest):")
print(f"{'Feature':<15} {'Importance':>12} {'Bar'}")
print("-" * 50)

for feat, imp in sorted(
    zip(features, importances),
    key=lambda x: x[1], reverse=True
):
    bar = "█" * int(imp * 60)
    print(f"{feat:<15} {imp:>12.4f}  {bar}")


# ──────────────────────────────────────────
# PART 9 - PREDICT NEW PASSENGERS
# ──────────────────────────────────────────

print("\n===== PART 9: Predict New Passengers =====")

def predict_survival(pclass, sex, age, fare,
                      sibsp, parch, embarked,
                      model, scaler_obj=None):
    family_size = sibsp + parch + 1
    is_alone    = 1 if family_size == 1 else 0
    age_group   = (0 if age <= 12 else
                   1 if age <= 18 else
                   2 if age <= 35 else
                   3 if age <= 60 else 4)
    fare_group  = (0 if fare <= 7.9 else
                   1 if fare <= 14.4 else
                   2 if fare <= 31 else 3)
    sex_enc     = 1 if sex == "male" else 0
    emb_enc     = {"C": 0, "Q": 1, "S": 2}.get(embarked, 2)
    title_enc   = (0 if sex == "male" and age >= 18 else
                   1 if sex == "female" and parch > 0 else
                   2 if sex == "female" else 3)

    features_arr = np.array([[
        pclass, sex_enc, age, fare,
        sibsp, parch, emb_enc,
        family_size, is_alone, title_enc,
        age_group, fare_group
    ]])

    pred = model.predict(features_arr)[0]
    prob = model.predict_proba(features_arr)[0]
    return pred, prob

# Test passengers
test_passengers = [
    ("Rich woman 1st class",  1, "female", 29, 211, 0, 0, "S"),
    ("Poor man 3rd class",    3, "male",   25,   7, 0, 0, "S"),
    ("Child 2nd class",       2, "female",  8,  26, 1, 1, "C"),
    ("Old man 1st class",     1, "male",   60, 100, 0, 0, "C"),
    ("Young woman 3rd class", 3, "female", 20,   8, 0, 0, "Q"),
]

rf_best = results["Random Forest"]["model"]

print(f"{'Passenger':<25} {'Survival%':>10} {'Prediction'}")
print("-" * 55)

for desc, pclass, sex, age, fare, sibsp, parch, emb in test_passengers:
    pred, prob = predict_survival(
        pclass, sex, age, fare,
        sibsp, parch, emb, rf_best)
    result = " SURVIVED" if pred == 1 else "❌ DIED"
    print(f"{desc:<25} {prob[1]*100:>9.1f}%  {result}")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Real dataset EDA - Titanic")
print(" Handling missing values strategically")
print(" Feature Engineering - create new features")
print(" Full ML pipeline end to end")
print(" Comparing 4 models together")
print(" Cross Validation - reliable evaluation")
print(" Feature Importance analysis")
print(" Predicting new passengers")
print("\n Day 13 Done! Tomorrow is REST DAY!")