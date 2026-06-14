# ============================================
# DAY 11 - Math Basics for ML
# Vectors, Matrices, Dot Product
# Author: Prateek Kumar Kuntal
# Date: 15 May 2026
# ============================================

import numpy as np


# ──────────────────────────────────────────
# PART 1 - WHAT IS A VECTOR
# ──────────────────────────────────────────

print("===== PART 1: Vectors =====")

print("""
VECTOR:
  A list of numbers with direction and magnitude
  In ML — every data row is a vector!

  Example:
  Student = [85, 92, 78, 90]
             Python Maths AI English

  This student is represented as a vector
  in 4-dimensional space!
""")

# Creating vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector v1        : {v1}")
print(f"Vector v2        : {v2}")
print(f"Shape            : {v1.shape}")
print(f"Dimensions       : {v1.ndim}")

# Vector operations
print(f"\n--- Vector Operations ---")
print(f"v1 + v2          : {v1 + v2}")
print(f"v1 - v2          : {v1 - v2}")
print(f"v1 * 2           : {v1 * 2}")
print(f"v1 / 2           : {v1 / 2}")
print(f"v1 * v2          : {v1 * v2}")    # element wise

# Vector magnitude (length)
magnitude = np.linalg.norm(v1)
print(f"\nMagnitude of v1  : {magnitude:.4f}")

# Unit vector (direction only)
unit_v1 = v1 / magnitude
print(f"Unit vector v1   : {unit_v1}")
print(f"Unit magnitude   : {np.linalg.norm(unit_v1):.4f}")  # always 1

# Real ML example
print(f"\n--- Real ML Example ---")
student1 = np.array([85, 92, 78, 90])   # marks in 4 subjects
student2 = np.array([70, 65, 80, 75])

print(f"Student 1 marks  : {student1}")
print(f"Student 2 marks  : {student2}")
print(f"Average diff     : {np.mean(student1 - student2):.2f}")
print(f"Student 1 total  : {np.sum(student1)}")
print(f"Student 2 total  : {np.sum(student2)}")


# ──────────────────────────────────────────
# PART 2 - DOT PRODUCT
# ──────────────────────────────────────────

print("\n===== PART 2: Dot Product =====")

print("""
DOT PRODUCT:
  Multiply corresponding elements then add them all
  Result is a single number (scalar)

  v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ...

  In ML used EVERYWHERE:
  - Neural network layer calculations
  - Similarity between vectors
  - Linear regression predictions
  - Attention mechanism in Transformers!
""")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Manual dot product
manual_dot = (v1[0]*v2[0]) + (v1[1]*v2[1]) + (v1[2]*v2[2])
print(f"v1               : {v1}")
print(f"v2               : {v2}")
print(f"Manual dot       : {manual_dot}")

# NumPy dot product
numpy_dot = np.dot(v1, v2)
print(f"NumPy dot        : {numpy_dot}")

# Using @ operator
operator_dot = v1 @ v2
print(f"@ operator dot   : {operator_dot}")

# Cosine Similarity (used in NLP!)
print(f"\n--- Cosine Similarity ---")
print("""
Cosine Similarity measures angle between vectors
Used in NLP to find similar documents/words
Range: -1 (opposite) to 1 (identical)
""")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

doc1 = np.array([1, 1, 0, 1, 0])   # word frequencies
doc2 = np.array([1, 1, 1, 0, 0])
doc3 = np.array([0, 0, 1, 0, 1])

sim_12 = cosine_similarity(doc1, doc2)
sim_13 = cosine_similarity(doc1, doc3)
sim_23 = cosine_similarity(doc2, doc3)

print(f"Doc1 vs Doc2 similarity : {sim_12:.4f}")
print(f"Doc1 vs Doc3 similarity : {sim_13:.4f}")
print(f"Doc2 vs Doc3 similarity : {sim_23:.4f}")
print(f"Doc1 most similar to    : {'Doc2' if sim_12 > sim_13 else 'Doc3'}")

# ML prediction using dot product
print(f"\n--- Linear Regression is just Dot Product! ---")
weights = np.array([0.5, 0.3, 0.2])   # learned weights
features = np.array([8, 90, 85])       # study hours, attendance, prev score
bias = 5.0

prediction = np.dot(weights, features) + bias
print(f"Weights          : {weights}")
print(f"Features         : {features}")
print(f"Bias             : {bias}")
print(f"Prediction       : {prediction}")
print("(This is exactly what Linear Regression does!)")


# ──────────────────────────────────────────
# PART 3 - MATRICES
# ──────────────────────────────────────────

print("\n===== PART 3: Matrices =====")

print("""
MATRIX:
  A 2D grid of numbers (rows x columns)
  In ML — entire dataset is a matrix!

  Dataset with 100 students, 5 features
  = Matrix of shape (100, 5)

  Each ROW    = one data sample
  Each COLUMN = one feature
""")

# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print(f"Matrix A:\n{A}")
print(f"\nMatrix B:\n{B}")
print(f"\nShape of A       : {A.shape}")
print(f"Rows             : {A.shape[0]}")
print(f"Columns          : {A.shape[1]}")

# Matrix operations
print(f"\n--- Matrix Operations ---")
print(f"A + B:\n{A + B}")
print(f"\nA - B:\n{A - B}")
print(f"\nA * 2:\n{A * 2}")
print(f"\nElement wise A*B:\n{A * B}")

# Transpose
print(f"\n--- Transpose ---")
print("""
Transpose flips rows and columns
Shape (3,4) becomes (4,3)
Used in matrix multiplication!
""")
print(f"A:\n{A}")
print(f"\nA Transpose:\n{A.T}")
print(f"Original shape   : {A.shape}")
print(f"Transposed shape : {A.T.shape}")


# ──────────────────────────────────────────
# PART 4 - MATRIX MULTIPLICATION
# ──────────────────────────────────────────

print("\n===== PART 4: Matrix Multiplication =====")

print("""
MATRIX MULTIPLICATION:
  (m x n) @ (n x p) = (m x p)
  Inner dimensions MUST match!

  This is the core operation in:
  - Linear Regression
  - Neural Networks
  - Deep Learning
  - Transformers
  EVERY ML model uses this!
""")

# Matrix multiplication
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])   # shape (3,2)

B = np.array([[7, 8, 9],
              [10, 11, 12]])  # shape (2,3)

C = np.dot(A, B)   # result shape (3,3)
print(f"A shape          : {A.shape}")
print(f"B shape          : {B.shape}")
print(f"A @ B shape      : {C.shape}")
print(f"\nA:\n{A}")
print(f"\nB:\n{B}")
print(f"\nA @ B:\n{C}")

# Neural Network layer example
print(f"\n--- Neural Network Layer Example ---")
print("""
Input layer → Hidden layer is just Matrix Multiplication!

Input  = data matrix    shape (batch, features)
Weights= learned matrix shape (features, hidden)
Output = Input @ Weights shape (batch, hidden)
""")

batch_size = 4
features   = 3
hidden     = 2

inputs  = np.random.randn(batch_size, features)
weights = np.random.randn(features, hidden)
bias    = np.zeros(hidden)

output  = inputs @ weights + bias

print(f"Input shape      : {inputs.shape}")
print(f"Weights shape    : {weights.shape}")
print(f"Output shape     : {output.shape}")
print(f"\nInputs:\n{inputs.round(3)}")
print(f"\nWeights:\n{weights.round(3)}")
print(f"\nOutput (inputs @ weights):\n{output.round(3)}")


# ──────────────────────────────────────────
# PART 5 - USEFUL MATRIX OPERATIONS IN ML
# ──────────────────────────────────────────

print("\n===== PART 5: Useful Matrix Operations =====")

A = np.array([[4, 7],
              [2, 6]])

# Determinant
det = np.linalg.det(A)
print(f"Matrix A:\n{A}")
print(f"\nDeterminant      : {det:.4f}")

# Inverse
inv = np.linalg.inv(A)
print(f"\nInverse:\n{inv.round(4)}")
print(f"\nA @ A_inv (should be Identity):\n{(A @ inv).round(4)}")

# Eigenvalues (used in PCA!)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues      : {eigenvalues.round(4)}")
print(f"Eigenvectors:\n{eigenvectors.round(4)}")

# Rank
rank = np.linalg.matrix_rank(A)
print(f"\nMatrix Rank      : {rank}")


# ──────────────────────────────────────────
# MINI PROJECT - Marks Matrix Analysis
# Using vectors and matrices together
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Marks Matrix Analysis =====")

np.random.seed(42)

# 10 students, 5 subjects
students = [f"Student_{i}" for i in range(1, 11)]
subjects = ["Python", "Maths", "AI", "English", "Physics"]

marks = np.random.randint(40, 100, size=(10, 5))

print("Marks Matrix (10 students x 5 subjects):")
print(f"{'Student':<12}", end="")
for s in subjects:
    print(f"{s:>10}", end="")
print()
print("-" * 62)

for i, student in enumerate(students):
    print(f"{student:<12}", end="")
    for mark in marks[i]:
        print(f"{mark:>10}", end="")
    print()

# Analysis using matrix operations
totals    = np.sum(marks, axis=1)      # row wise
averages  = np.mean(marks, axis=1)    # row wise
sub_avg   = np.mean(marks, axis=0)    # column wise
sub_max   = np.max(marks, axis=0)
sub_min   = np.min(marks, axis=0)

print(f"\n--- Student Analysis ---")
print(f"{'Student':<12} {'Total':>8} {'Average':>10} {'Grade':>8}")
print("-" * 42)
for i, student in enumerate(students):
    grade = ("A+" if averages[i] >= 90 else
             "A"  if averages[i] >= 75 else
             "B"  if averages[i] >= 60 else
             "C"  if averages[i] >= 50 else "F")
    print(f"{student:<12} {totals[i]:>8} {averages[i]:>10.2f} {grade:>8}")

print(f"\n--- Subject Analysis ---")
print(f"{'Subject':<12} {'Avg':>8} {'Max':>8} {'Min':>8}")
print("-" * 38)
for i, subject in enumerate(subjects):
    print(f"{subject:<12} {sub_avg[i]:>8.2f} {sub_max[i]:>8} {sub_min[i]:>8}")

# Similarity between students using dot product
print(f"\n--- Student Similarity (Dot Product) ---")
s1_marks = marks[0] / np.linalg.norm(marks[0])
s2_marks = marks[1] / np.linalg.norm(marks[1])
s3_marks = marks[2] / np.linalg.norm(marks[2])

sim_12 = np.dot(s1_marks, s2_marks)
sim_13 = np.dot(s1_marks, s3_marks)

print(f"Student_1 vs Student_2 : {sim_12:.4f}")
print(f"Student_1 vs Student_3 : {sim_13:.4f}")
print(f"Student_1 most similar to: {'Student_2' if sim_12 > sim_13 else 'Student_3'}")

# Topper and lowest
print(f"\n--- Rankings ---")
print(f"Class Topper     : {students[np.argmax(totals)]} ({np.max(totals)} marks)")
print(f"Lowest Scorer    : {students[np.argmin(totals)]} ({np.min(totals)} marks)")
print(f"Best Subject     : {subjects[np.argmax(sub_avg)]} (avg {np.max(sub_avg):.2f})")
print(f"Hardest Subject  : {subjects[np.argmin(sub_avg)]} (avg {np.min(sub_avg):.2f})")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Vectors - what they are and operations")
print(" Dot Product - core of all ML calculations")
print(" Cosine Similarity - used in NLP")
print(" Matrices - dataset as a matrix")
print(" Matrix Multiplication - heart of Neural Networks")
print(" Matrix operations - inverse, eigenvalues")
print(" Mini Project - Marks Matrix Analysis")
print("\ Day 11 Done! Tomorrow - Derivatives & Gradients!")