# ============================================
# DAY 6 - Pandas Advanced
# Merge, Pivot, Null Values
# Author: PRATEEK KUMAR KUNTAL
# Date: 10 May 2026
# ============================================

import pandas as pd
import numpy as np


# ──────────────────────────────────────────
# PART 1 - ADVANCED MERGING
# ──────────────────────────────────────────

print("===== PART 1: Advanced Merging =====")

students = pd.DataFrame({
    "StudentID" : [1, 2, 3, 4, 5],
    "Name"      : ["Raj", "Priya", "Amit", "Sneha", "Arjun"],
    "Branch"    : ["AIML", "CSE", "ECE", "IT", "AIML"]
})

marks = pd.DataFrame({
    "StudentID" : [1, 2, 3, 6, 7],
    "Marks"     : [85, 92, 45, 78, 88],
    "Grade"     : ["A", "A+", "F", "A", "A+"]
})

attendance = pd.DataFrame({
    "StudentID"  : [1, 2, 3, 4, 5],
    "Attendance" : [95, 88, 72, 91, 65]
})

print("Students:\n", students)
print("\nMarks:\n", marks)
print("\nAttendance:\n", attendance)

# Inner join
inner = pd.merge(students, marks, on="StudentID", how="inner")
print("\nInner Join:")
print(inner)

# Left join
left = pd.merge(students, marks, on="StudentID", how="left")
print("\nLeft Join:")
print(left)

# Right join
right = pd.merge(students, marks, on="StudentID", how="right")
print("\nRight Join:")
print(right)

# Outer join
outer = pd.merge(students, marks, on="StudentID", how="outer")
print("\nOuter Join:")
print(outer)

# Merge 3 tables
full = pd.merge(students, marks, on="StudentID", how="left")
full = pd.merge(full, attendance, on="StudentID", how="left")
print("\nAll 3 tables merged:")
print(full)


# ──────────────────────────────────────────
# PART 2 - CONCATENATION
# ──────────────────────────────────────────

print("\n===== PART 2: Concatenation =====")

batch1 = pd.DataFrame({
    "Name"   : ["Raj", "Priya", "Amit"],
    "Branch" : ["AIML", "CSE", "ECE"],
    "Marks"  : [85, 92, 45]
})

batch2 = pd.DataFrame({
    "Name"   : ["Sneha", "Arjun", "Neha"],
    "Branch" : ["IT", "AIML", "CSE"],
    "Marks"  : [78, 88, 67]
})

# Stack vertically
combined = pd.concat([batch1, batch2], ignore_index=True)
print("Combined batches:")
print(combined)

# Stack horizontally
extra_info = pd.DataFrame({
    "City" : ["Delhi", "Mumbai", "Pune",
               "Chennai", "Hyderabad", "Kolkata"]
})
combined_h = pd.concat([combined, extra_info], axis=1)
print("\nWith city info:")
print(combined_h)


# ──────────────────────────────────────────
# PART 3 - HANDLING NULL VALUES (Advanced)
# ──────────────────────────────────────────

print("\n===== PART 3: Handling Null Values =====")

np.random.seed(42)
data = {
    "Name"       : ["Raj", None, "Amit", "Sneha", None,
                    "Neha", "Ravi", None, "Arjun", "Pooja"],
    "Age"        : [20, 21, None, 22, 20,
                    None, 21, 22, None, 20],
    "Marks"      : [85, 92, 45, None, 88,
                    55, None, 78, 91, None],
    "City"       : ["Delhi", "Mumbai", None, "Pune", "Delhi",
                    None, "Mumbai", "Chennai", None, "Pune"],
    "Attendance" : [95, None, 72, 88, None,
                    91, 65, None, 88, 77]
}

df = pd.DataFrame(data)
print("Original DataFrame with nulls:")
print(df)

# Check nulls
print(f"\nNull count per column:")
print(df.isnull().sum())

print(f"\nNull percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Strategy 1 — Fill with mean (numerical)
df_filled = df.copy()
df_filled["Marks"]      = df_filled["Marks"].fillna(
                           df_filled["Marks"].mean().round(2))
df_filled["Age"]        = df_filled["Age"].fillna(
                           df_filled["Age"].median())
df_filled["Attendance"] = df_filled["Attendance"].fillna(
                           df_filled["Attendance"].mean().round(2))

# Strategy 2 — Fill with mode (categorical)
df_filled["Name"] = df_filled["Name"].fillna("Unknown")
df_filled["City"] = df_filled["City"].fillna(
                    df_filled["City"].mode()[0])

print("\nAfter filling nulls:")
print(df_filled)

# Strategy 3 — Forward fill
df_ffill = df.copy()
df_ffill = df_ffill.ffill()
print("\nForward Fill:")
print(df_ffill)

# Strategy 4 — Backward fill
df_bfill = df.copy()
df_bfill = df_bfill.bfill()
print("\nBackward Fill:")
print(df_bfill)

# Strategy 5 — Drop rows with too many nulls
df_dropped = df.dropna(thresh=4)  # keep rows with 4+ non-null
print("\nAfter dropping rows with many nulls:")
print(df_dropped)


# ──────────────────────────────────────────
# PART 4 - PIVOT TABLES
# (Summarize data like Excel pivot)
# ──────────────────────────────────────────

print("\n===== PART 4: Pivot Tables =====")

data2 = {
    "Name"    : ["Raj", "Priya", "Amit", "Sneha", "Arjun",
                 "Neha", "Ravi", "Pooja", "Vikram", "Ananya"],
    "Branch"  : ["AIML", "CSE", "ECE", "AIML", "CSE",
                 "ECE", "AIML", "CSE", "ECE", "AIML"],
    "Gender"  : ["M", "F", "M", "F", "M",
                 "F", "M", "F", "M", "F"],
    "Marks"   : [85, 92, 45, 78, 88, 55, 91, 67, 72, 83],
    "Attendance":[95, 88, 72, 91, 65, 85, 90, 78, 82, 94]
}

df2 = pd.DataFrame(data2)
print("Data:")
print(df2)

# Basic pivot — average marks by branch
pivot1 = df2.pivot_table(
    values="Marks",
    index="Branch",
    aggfunc="mean"
).round(2)
print("\nAverage Marks by Branch:")
print(pivot1)

# Pivot with multiple values
pivot2 = df2.pivot_table(
    values=["Marks", "Attendance"],
    index="Branch",
    aggfunc="mean"
).round(2)
print("\nAverage Marks & Attendance by Branch:")
print(pivot2)

# Pivot with rows and columns
pivot3 = df2.pivot_table(
    values="Marks",
    index="Branch",
    columns="Gender",
    aggfunc="mean"
).round(2)
print("\nAverage Marks by Branch and Gender:")
print(pivot3)

# Pivot with multiple aggregations
pivot4 = df2.pivot_table(
    values="Marks",
    index="Branch",
    aggfunc=["mean", "max", "min", "count"]
).round(2)
print("\nFull Stats by Branch:")
print(pivot4)


# ──────────────────────────────────────────
# PART 5 - APPLY & MAP
# (Apply custom functions to columns)
# ──────────────────────────────────────────

print("\n===== PART 5: Apply & Map =====")

df3 = pd.DataFrame({
    "Name"  : ["Raj", "Priya", "Amit", "Sneha", "Arjun"],
    "Marks" : [85, 92, 45, 78, 38]
})

# Apply with lambda
df3["Grade"] = df3["Marks"].apply(lambda x:
    "A+" if x >= 90 else
    "A"  if x >= 75 else
    "B"  if x >= 60 else
    "C"  if x >= 50 else "F")

# Apply custom function
def scholarship(marks):
    if marks >= 90:
        return "Full Scholarship"
    elif marks >= 75:
        return "Half Scholarship"
    else:
        return "No Scholarship"

df3["Scholarship"] = df3["Marks"].apply(scholarship)

# Map — replace values
df3["Result"] = df3["Grade"].map({
    "A+" : "Outstanding",
    "A"  : "Excellent",
    "B"  : "Good",
    "C"  : "Average",
    "F"  : "Fail"
})

print("DataFrame with applied functions:")
print(df3)


# ──────────────────────────────────────────
# MINI PROJECT - Full EDA on Student Dataset
# (EDA = Exploratory Data Analysis)
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Full Student EDA =====")

np.random.seed(42)
n = 100

names    = [f"Student_{i}"  for i in range(1, n+1)]
branches = np.random.choice(["AIML","CSE","ECE","IT","Mech"], n)
cities   = np.random.choice(["Delhi","Mumbai","Pune","Hyd","Chennai"], n)
genders  = np.random.choice(["M", "F"], n)
python   = np.random.randint(35, 100, n).astype(float)  # make float for nulls
maths    = np.random.randint(35, 100, n).astype(float)
ai       = np.random.randint(35, 100, n) 

# Introduce some nulls
python[np.random.choice(n, 5, replace=False)] = np.nan
maths[np.random.choice(n, 5, replace=False)]  = np.nan

df_eda = pd.DataFrame({
    "Name"    : names,
    "Branch"  : branches, 
    "City"    : cities,
    "Gender"  : genders,
    "Python"  : python,
    "Maths"   : maths,
    "AI"      : ai
})

print(f"Dataset Shape: {df_eda.shape}")
print(f"\nFirst 5 rows:\n{df_eda.head()}")
print(f"\nNull values:\n{df_eda.isnull().sum()}")

# Clean nulls
df_eda["Python"] = df_eda["Python"].fillna(df_eda["Python"].mean())
df_eda["Maths"]  = df_eda["Maths"].fillna(df_eda["Maths"].mean())

# Add calculated columns
df_eda["Total"]   = (df_eda["Python"] +
                     df_eda["Maths"]  +
                     df_eda["AI"]).round(2)
df_eda["Average"] = (df_eda["Total"] / 3).round(2)
df_eda["Grade"]   = df_eda["Average"].apply(lambda x:
    "A+" if x >= 90 else
    "A"  if x >= 75 else
    "B"  if x >= 60 else
    "C"  if x >= 50 else "F")
df_eda["Result"]  = df_eda["Average"].apply(
    lambda x: "Pass" if x >= 50 else "Fail")

print("\n--- Branch Analysis ---")
print(df_eda.pivot_table(
    values="Average",
    index="Branch",
    aggfunc=["mean","max","min","count"]
).round(2))

print("\n--- Gender Analysis ---")
print(df_eda.groupby("Gender")["Average"].mean().round(2))

print("\n--- Grade Distribution ---")
print(df_eda["Grade"].value_counts())

print("\n--- City Toppers ---")
city_toppers = df_eda.loc[
    df_eda.groupby("City")["Average"].idxmax(),
    ["City","Name","Branch","Average"]
]
print(city_toppers)

print("\n--- Top 5 Students ---")
print(df_eda.nlargest(5,"Total")[
    ["Name","Branch","Total","Grade"]
])

# Save
df_eda.to_csv("eda_results.csv", index=False)
print("\n EDA saved to eda_results.csv!")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Advanced Merging - inner/left/right/outer")
print(" Concatenation - vertical and horizontal")
print(" Null Handling - mean/mode/ffill/bfill/drop")
print(" Pivot Tables - summarize like Excel")
print(" Apply & Map - custom functions on columns")
print(" Mini Project - Full EDA on 100 students")
