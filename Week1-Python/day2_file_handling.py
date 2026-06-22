# ============================================
# DAY 2 - File Handling + Exception Handling
#          + List Comprehensions
# Author: PRATEEK KUMAR KUNTAL
# Date: 6 May 2026
# ============================================


# ──────────────────────────────────────────
# PART 1 - FILE HANDLING
# ──────────────────────────────────────────

print("===== PART 1: File Handling =====")

# Write to file
with open("students.txt", "w") as f:
    f.write("Raj, AIML, 2nd Year\n")
    f.write("Priya, CSE, 3rd Year\n")
    f.write("Amit, ECE, 1st Year\n")
print(" File created and written!")

# Read full file
print("\n--- Full File Content ---")
with open("students.txt", "r") as f:
    print(f.read())

# Read line by line
print("--- Line by Line ---")
with open("students.txt", "r") as f:
    for line in f:
        print(line.strip())

# Append new data
with open("students.txt", "a") as f:
    f.write("Sneha, IT, 4th Year\n")
print("\n New student appended!")

# Read updated file
with open("students.txt", "r") as f:
    print(f.read())


# ──────────────────────────────────────────
# PART 2 - EXCEPTION HANDLING
# ──────────────────────────────────────────

print("===== PART 2: Exception Handling =====")

# Basic try except
def safe_divide(a, b):
    try:
        result = a / b
        print(f" {a} / {b} = {result}")
    except ZeroDivisionError:
        print(" Cannot divide by zero!")
    except TypeError:
        print(" Numbers only please!")
    finally:
        print("--- attempt done ---\n")

safe_divide(10, 2)
safe_divide(10, 0)
safe_divide(10, "five")

# File not found exception
print("--- File Not Found ---")
try:
    with open("missing.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print(" File not found! Program still running ✅")

# Value error
print("\n--- Value Error ---")
try:
    age = int("twenty")
except ValueError:
    print(" Cannot convert text to number!")


# ──────────────────────────────────────────
# PART 3 - LIST COMPREHENSIONS
# ──────────────────────────────────────────

print("\n===== PART 3: List Comprehensions =====")

# Normal way vs List comprehension
numbers = [1, 2, 3, 4, 5]

# Old way
squares_old = []
for n in numbers:
    squares_old.append(n ** 2)
print(f"Old way    : {squares_old}")

# List comprehension way
squares_new = [n ** 2 for n in numbers]
print(f"New way    : {squares_new}")

# With condition
evens = [n for n in numbers if n % 2 == 0]
print(f"Evens only : {evens}")

# String list comprehension
names = ["raj", "priya", "amit", "sneha"]
upper = [name.upper() for name in names]
print(f"Original   : {names}")
print(f"Uppercase  : {upper}")

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"Matrix     : {matrix}")
print(f"Flattened  : {flat}")


# ──────────────────────────────────────────
# MINI PROJECT - Student Report System
# Using all 3 concepts together
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Student Report =====")

students = [
    {"name": "Raj",   "marks": 85},
    {"name": "Priya", "marks": 92},
    {"name": "Amit",  "marks": 45},
    {"name": "Sneha", "marks": 78},
    {"name": "Jo",    "marks": 38},
]

# List comprehension to get passers and failers
passing = [s for s in students if s["marks"] >= 50]
failing = [s for s in students if s["marks"] < 50]
toppers = [s for s in students if s["marks"] >= 80]

# Save to file with exception handling
try:
    with open("report.txt", "w") as f:
        f.write("===== STUDENT REPORT =====\n\n")

        f.write("ALL STUDENTS:\n")
        for s in students:
            f.write(f"  {s['name']} - {s['marks']} marks\n")

        f.write(f"\nPASSING ({len(passing)} students):\n")
        for s in passing:
            f.write(f"  {s['name']} - {s['marks']} marks\n")

        f.write(f"\nFAILING ({len(failing)} students):\n")
        for s in failing:
            f.write(f"  {s['name']} - {s['marks']} marks\n")

        f.write(f"\nTOPPERS ({len(toppers)} students):\n")
        for s in toppers:
            f.write(f"  {s['name']} - {s['marks']} marks\n")

    print("✅ Report saved to report.txt!")

    # Read and display
    with open("report.txt", "r") as f:
        print(f.read())

except Exception as e:
    print(f" Error: {e}")
