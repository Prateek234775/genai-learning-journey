# ============================================
# DAY 4 - NumPy 
# Author: PRATEEK KUMAR KUNTAL
# Date: 8 May 2025
# ============================================

import numpy as np


# ──────────────────────────────────────────
# PART 1 - CREATING ARRAYS
# ──────────────────────────────────────────

print("===== PART 1: Creating Arrays =====")

# From list
arr1 = np.array([1, 2, 3, 4, 5])
print(f"1D Array     : {arr1}")
print(f"Type         : {type(arr1)}")
print(f"Data type    : {arr1.dtype}")
print(f"Shape        : {arr1.shape}")

# 2D Array (Matrix)
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(f"\n2D Array:\n{arr2}")
print(f"Shape  : {arr2.shape}")   # (3 rows, 3 cols)
print(f"Size   : {arr2.size}")    # total elements
print(f"Dims   : {arr2.ndim}")    # number of dimensions

# Special arrays
zeros   = np.zeros((3, 3))
ones    = np.ones((2, 4))
full    = np.full((3, 3), 7)
eye     = np.eye(4)              # identity matrix
random  = np.random.rand(3, 3)  # random 0 to 1
randint = np.random.randint(1, 100, (3, 3))  # random integers

print(f"\nZeros:\n{zeros}")
print(f"\nOnes:\n{ones}")
print(f"\nFull of 7s:\n{full}")
print(f"\nIdentity Matrix:\n{eye}")
print(f"\nRandom floats:\n{random}")
print(f"\nRandom integers:\n{randint}")

# Range arrays
range_arr  = np.arange(0, 20, 2)    # start, stop, step
linspace   = np.linspace(0, 1, 5)   # 5 evenly spaced values

print(f"\nArange (0-20 step 2) : {range_arr}")
print(f"Linspace (0 to 1)    : {linspace}")


# ──────────────────────────────────────────
# PART 2 - ARRAY SLICING & INDEXING
# ──────────────────────────────────────────

print("\n===== PART 2: Slicing & Indexing =====")

arr = np.array([10, 20, 30, 40, 50, 60, 70])

# Basic indexing
print(f"Array        : {arr}")
print(f"First item   : {arr[0]}")
print(f"Last item    : {arr[-1]}")
print(f"3rd item     : {arr[2]}")

# Slicing
print(f"\nFirst 3      : {arr[:3]}")
print(f"Last 3       : {arr[-3:]}")
print(f"Middle       : {arr[2:5]}")
print(f"Every 2nd    : {arr[::2]}")
print(f"Reversed     : {arr[::-1]}")

# 2D Array slicing
matrix = np.array([[1,  2,  3,  4],
                   [5,  6,  7,  8],
                   [9,  10, 11, 12]])

print(f"\nMatrix:\n{matrix}")
print(f"\nRow 0        : {matrix[0]}")
print(f"Row 1        : {matrix[1]}")
print(f"Col 0        : {matrix[:, 0]}")
print(f"Col 2        : {matrix[:, 2]}")
print(f"Row 0-1, Col 1-2:\n{matrix[0:2, 1:3]}")
print(f"Element [1][2]: {matrix[1][2]}")


# ──────────────────────────────────────────
# PART 3 - ARRAY OPERATIONS
# ──────────────────────────────────────────

print("\n===== PART 3: Array Operations =====")

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Arithmetic
print(f"a         : {a}")
print(f"b         : {b}")
print(f"a + b     : {a + b}")
print(f"b - a     : {b - a}")
print(f"a * b     : {a * b}")
print(f"b / a     : {b / a}")
print(f"a ** 2    : {a ** 2}")
print(f"a * 10    : {a * 10}")   # scalar operation

# Matrix multiplication
mat1 = np.array([[1, 2],
                 [3, 4]])
mat2 = np.array([[5, 6],
                 [7, 8]])

print(f"\nMatrix 1:\n{mat1}")
print(f"Matrix 2:\n{mat2}")
print(f"Dot product:\n{np.dot(mat1, mat2)}")
print(f"Element multiply:\n{mat1 * mat2}")


# ──────────────────────────────────────────
# PART 4 - USEFUL NUMPY FUNCTIONS
# ──────────────────────────────────────────

print("\n===== PART 4: Useful Functions =====")

data = np.array([15, 42, 8, 99, 23, 55, 37, 71, 4, 88])

print(f"Data       : {data}")
print(f"Min        : {np.min(data)}")
print(f"Max        : {np.max(data)}")
print(f"Sum        : {np.sum(data)}")
print(f"Mean       : {np.mean(data)}")
print(f"Median     : {np.median(data)}")
print(f"Std Dev    : {np.std(data):.2f}")
print(f"Variance   : {np.var(data):.2f}")
print(f"Sorted     : {np.sort(data)}")
print(f"Index of Max: {np.argmax(data)}")
print(f"Index of Min: {np.argmin(data)}")


# ──────────────────────────────────────────
# PART 5 - RESHAPING & STACKING
# ──────────────────────────────────────────

print("\n===== PART 5: Reshaping & Stacking =====")

arr = np.arange(1, 13)   # [1, 2, 3 ... 12]
print(f"Original     : {arr}")

# Reshape
reshaped = arr.reshape(3, 4)   # 3 rows 4 cols
print(f"\nReshaped 3x4:\n{reshaped}")

reshaped2 = arr.reshape(2, 6)
print(f"\nReshaped 2x6:\n{reshaped2}")

reshaped3 = arr.reshape(4, 3)
print(f"\nReshaped 4x3:\n{reshaped3}")

# Flatten back to 1D
flat = reshaped.flatten()
print(f"\nFlattened    : {flat}")

# Stacking arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

h_stack = np.hstack([a, b])   # horizontal
v_stack = np.vstack([a, b])   # vertical

print(f"\nHorizontal stack : {h_stack}")
print(f"Vertical stack:\n{v_stack}")


# ──────────────────────────────────────────
# PART 6 - BOOLEAN MASKING
# (Very important in ML data filtering!)
# ──────────────────────────────────────────

print("\n===== PART 6: Boolean Masking =====")

marks = np.array([85, 42, 91, 55, 38, 77, 63, 29, 88, 71])

print(f"All marks    : {marks}")

# Create boolean mask
passing_mask = marks >= 50
print(f"Passing mask : {passing_mask}")

# Apply mask
passing_marks = marks[passing_mask]
failing_marks = marks[~passing_mask]   # ~ means NOT

print(f"Passing marks: {passing_marks}")
print(f"Failing marks: {failing_marks}")

# Multiple conditions
good_marks = marks[(marks >= 60) & (marks <= 90)]
print(f"60 to 90     : {good_marks}")

# Replace values using mask
marks_copy = marks.copy()
marks_copy[marks_copy < 50] = 0   # fail = 0
print(f"Fails = 0    : {marks_copy}")


# ──────────────────────────────────────────
# MINI PROJECT - Student Marks Analyser
# Full analysis using NumPy
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Marks Analyser =====")

# Simulate marks of 20 students in 3 subjects
np.random.seed(42)   # for consistent results
marks = np.random.randint(35, 100, size=(20, 3))
subjects = ["Python", "Maths", "AI"]

print("Marks of 20 Students (3 subjects):")
print(f"{'Student':<10} {'Python':<10} {'Maths':<10} {'AI':<10} {'Total':<10} {'Avg':<10}")
print("-" * 60)

totals   = np.sum(marks, axis=1)     # row wise sum
averages = np.mean(marks, axis=1)    # row wise mean

for i in range(20):
    print(f"{'S'+str(i+1):<10} {marks[i][0]:<10} {marks[i][1]:<10} {marks[i][2]:<10} {totals[i]:<10} {averages[i]:.1f}")

print("\n--- Subject Analysis ---")
for i, subject in enumerate(subjects):
    subject_marks = marks[:, i]
    print(f"\n{subject}:")
    print(f"  Highest  : {np.max(subject_marks)}")
    print(f"  Lowest   : {np.min(subject_marks)}")
    print(f"  Average  : {np.mean(subject_marks):.2f}")
    print(f"  Std Dev  : {np.std(subject_marks):.2f}")
    print(f"  Passing  : {np.sum(subject_marks >= 50)}/20 students")

print("\n--- Overall Analysis ---")
print(f"Class topper    : Student S{np.argmax(totals)+1} with {np.max(totals)} total marks")
print(f"Lowest scorer   : Student S{np.argmin(totals)+1} with {np.min(totals)} total marks")
print(f"Class average   : {np.mean(averages):.2f}")

# Students above class average
above_avg = np.sum(averages > np.mean(averages))
print(f"Above average   : {above_avg} students")

