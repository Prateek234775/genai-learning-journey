# ============================================
# DAY 5 - Pandas Complete
# Author: PRATEEK KUMAR KUNTAL
# Date: 9 May 2026
# ============================================

import pandas as pd
import numpy as np


# ──────────────────────────────────────────
# PART 1 - CREATING DATAFRAMES
# ──────────────────────────────────────────

print("===== PART 1: Creating DataFrames =====")

# From dictionary
data = {
    "Name"   : ["Raj", "Priya", "Amit", "Sneha", "Arjun"],
    "Branch" : ["AIML", "CSE", "ECE", "IT", "AIML"],
    "Marks"  : [85, 92, 45, 78, 88],
    "Age"    : [20, 21, 20, 22, 21]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print(f"\nShape  : {df.shape}")       # rows, cols
print(f"Columns: {list(df.columns)}")
print(f"Types  :\n{df.dtypes}")


# ──────────────────────────────────────────
# PART 2 - READING CSV FILE
# ──────────────────────────────────────────

print("\n===== PART 2: Reading CSV File =====")

# First create a CSV to read
df.to_csv("students.csv", index=False)
print("✅ CSV created!")

# Read it back
df_csv = pd.read_csv("students.csv")
print("\nRead from CSV:")
print(df_csv)

# Basic info
print(f"\nFirst 3 rows:\n{df_csv.head(3)}")
print(f"\nLast 2 rows:\n{df_csv.tail(2)}")
print(f"\nInfo:")
df_csv.info()
print(f"\nStatistics:\n{df_csv.describe()}")


# ──────────────────────────────────────────
# PART 3 - SELECTING DATA
# ──────────────────────────────────────────

print("\n===== PART 3: Selecting Data =====")

# Select single column
print("Names column:")
print(df["Name"])

# Select multiple columns
print("\nName and Marks:")
print(df[["Name", "Marks"]])

# Select by row index — iloc
print(f"\nFirst row (iloc)  : \n{df.iloc[0]}")
print(f"\nRows 1-3 (iloc)   :\n{df.iloc[1:4]}")

# Select by label — loc
print(f"\nRow label 0 (loc) :\n{df.loc[0]}")

# Specific cell
print(f"\nMarks of Priya    : {df.loc[1, 'Marks']}")


# ──────────────────────────────────────────
# PART 4 - FILTERING DATA
# ──────────────────────────────────────────

print("\n===== PART 4: Filtering Data =====")

# Single condition
passing = df[df["Marks"] >= 50]
print("Passing students:")
print(passing)

# Multiple conditions
aiml_toppers = df[(df["Branch"] == "AIML") & (df["Marks"] >= 80)]
print("\nAIML students with 80+ marks:")
print(aiml_toppers)

# isin filter
selected = df[df["Branch"].isin(["AIML", "CSE"])]
print("\nAIML and CSE students:")
print(selected)

# String filter
r_names = df[df["Name"].str.startswith("R")]
print("\nNames starting with R:")
print(r_names)


# ──────────────────────────────────────────
# PART 5 - ADDING & REMOVING COLUMNS
# ──────────────────────────────────────────

print("\n===== PART 5: Adding & Removing Columns =====")

# Add new column
df["Pass"] = df["Marks"].apply(lambda x: "Pass" if x >= 50 else "Fail")
df["Grade"] = df["Marks"].apply(lambda x:
    "A+" if x >= 90 else
    "A"  if x >= 75 else
    "B"  if x >= 60 else
    "C"  if x >= 50 else "F")

print("With Pass and Grade columns:")
print(df)

# Drop a column
df_dropped = df.drop(columns=["Age"])
print("\nAfter dropping Age:")
print(df_dropped)


# ──────────────────────────────────────────
# PART 6 - HANDLING NULL VALUES
# ──────────────────────────────────────────

print("\n===== PART 6: Handling Null Values =====")

# Create dataframe with nulls
data_with_nulls = {
    "Name"  : ["Raj", "Priya", None, "Sneha", "Arjun"],
    "Marks" : [85, None, 45, 78, None],
    "City"  : ["Delhi", "Mumbai", "Pune", None, "Chennai"]
}
df_null = pd.DataFrame(data_with_nulls)

print("DataFrame with nulls:")
print(df_null)
print(f"\nNull counts:\n{df_null.isnull().sum()}")

# Fill nulls
df_filled = df_null.copy()
df_filled["Marks"] = df_filled["Marks"].fillna(df_filled["Marks"].mean())
df_filled["Name"]  = df_filled["Name"].fillna("Unknown")
df_filled["City"]  = df_filled["City"].fillna("Not Provided")

print("\nAfter filling nulls:")
print(df_filled)

# Drop rows with nulls
df_dropped_nulls = df_null.dropna()
print("\nAfter dropping null rows:")
print(df_dropped_nulls)


# ──────────────────────────────────────────
# PART 7 - GROUPBY
# (Group data and analyse each group)
# ──────────────────────────────────────────

print("\n===== PART 7: GroupBy =====")

data2 = {
    "Name"    : ["Raj", "Priya", "Amit", "Sneha", "Arjun",
                 "Neha", "Ravi", "Pooja"],
    "Branch"  : ["AIML", "CSE", "ECE", "AIML", "CSE",
                 "ECE", "AIML", "CSE"],
    "Marks"   : [85, 92, 45, 78, 88, 55, 91, 67],
    "Gender"  : ["M", "F", "M", "F", "M", "F", "M", "F"]
}
df2 = pd.DataFrame(data2)

# Average marks by branch
print("Average marks by Branch:")
print(df2.groupby("Branch")["Marks"].mean())

# Multiple aggregations
print("\nBranch statistics:")
print(df2.groupby("Branch")["Marks"].agg(["mean", "max", "min", "count"]))

# Group by multiple columns
print("\nBranch + Gender average:")
print(df2.groupby(["Branch", "Gender"])["Marks"].mean())


# ──────────────────────────────────────────
# PART 8 - MERGING DATAFRAMES
# (Like SQL JOIN)
# ──────────────────────────────────────────

print("\n===== PART 8: Merging DataFrames =====")

# Student basic info
students = pd.DataFrame({
    "StudentID" : [1, 2, 3, 4, 5],
    "Name"      : ["Raj", "Priya", "Amit", "Sneha", "Arjun"],
    "Branch"    : ["AIML", "CSE", "ECE", "IT", "AIML"]
})

# Student marks
marks = pd.DataFrame({
    "StudentID" : [1, 2, 3, 4, 6],
    "Marks"     : [85, 92, 45, 78, 88],
    "Grade"     : ["A", "A+", "F", "A", "A+"]
})

print("Students table:")
print(students)
print("\nMarks table:")
print(marks)

# Inner join — only matching
inner = pd.merge(students, marks, on="StudentID", how="inner")
print("\nInner Join (matching only):")
print(inner)

# Left join — all students
left = pd.merge(students, marks, on="StudentID", how="left")
print("\nLeft Join (all students):")
print(left)


# ──────────────────────────────────────────
# PART 9 - SORTING
# ──────────────────────────────────────────

print("\n===== PART 9: Sorting =====")

# Sort by marks descending
sorted_df = df2.sort_values("Marks", ascending=False)
print("Sorted by Marks (high to low):")
print(sorted_df[["Name", "Branch", "Marks"]])

# Sort by multiple columns
sorted_multi = df2.sort_values(["Branch", "Marks"],
                                ascending=[True, False])
print("\nSorted by Branch then Marks:")
print(sorted_multi[["Name", "Branch", "Marks"]])


# ──────────────────────────────────────────
# MINI PROJECT - Full Student Analysis
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Full Student Analysis =====")

# Generate dataset
np.random.seed(42)
n = 50

branches = ["AIML", "CSE", "ECE", "IT", "Mechanical"]
cities   = ["Delhi", "Mumbai", "Pune", "Hyderabad", "Chennai"]

dataset = {
    "StudentID" : range(1, n+1),
    "Name"      : [f"Student_{i}" for i in range(1, n+1)],
    "Branch"    : np.random.choice(branches, n),
    "City"      : np.random.choice(cities, n),
    "Python"    : np.random.randint(35, 100, n),
    "Maths"     : np.random.randint(35, 100, n),
    "AI"        : np.random.randint(35, 100, n),
}

df_main = pd.DataFrame(dataset)

# Add calculated columns
df_main["Total"]   = df_main["Python"] + df_main["Maths"] + df_main["AI"]
df_main["Average"] = (df_main["Total"] / 3).round(2)
df_main["Result"]  = df_main["Average"].apply(
    lambda x: "Pass" if x >= 50 else "Fail")
df_main["Grade"]   = df_main["Average"].apply(lambda x:
    "A+" if x >= 90 else
    "A"  if x >= 75 else
    "B"  if x >= 60 else
    "C"  if x >= 50 else "F")

print(f"Dataset shape : {df_main.shape}")
print(f"\nFirst 5 rows:\n{df_main.head()}")

# Analysis
print("\n--- Branch Wise Analysis ---")
branch_analysis = df_main.groupby("Branch")["Average"].agg(
    ["mean", "max", "min", "count"]).round(2)
branch_analysis.columns = ["Avg Marks", "Highest", "Lowest", "Students"]
print(branch_analysis)

print("\n--- Pass/Fail Summary ---")
print(df_main["Result"].value_counts())

print("\n--- Grade Distribution ---")
print(df_main["Grade"].value_counts().sort_index())

print("\n--- Top 5 Students ---")
top5 = df_main.nlargest(5, "Total")[["Name", "Branch", "Total", "Grade"]]
print(top5)

print("\n--- City Wise Average ---")
print(df_main.groupby("City")["Average"].mean().round(2).sort_values(ascending=False))

# Save to CSV
df_main.to_csv("full_analysis.csv", index=False)
print("\n Full analysis saved to full_analysis.csv!")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Creating DataFrames")
print(" Reading CSV files")
print(" Selecting data - iloc, loc")
print(" Filtering - conditions, isin, string")
print(" Adding & Removing columns")
print(" Handling Null values")
print(" GroupBy - aggregate & analyse")
print(" Merging DataFrames - like SQL JOIN")
print(" Sorting data")
print(" Mini Project - Full 50 Student Analysis")
