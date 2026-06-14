# ============================================
# DAY 12 - Derivatives, Gradients, Chain Rule
# Author: Prateek Kumar Kuntal
# Date: 16 May 2026
# ============================================

import numpy as np


# ──────────────────────────────────────────
# PART 1 - WHAT IS A DERIVATIVE
# ──────────────────────────────────────────

print("===== PART 1: What is a Derivative =====")

print("""
DERIVATIVE:
  Rate of change of a function
  How much output changes when input changes slightly

  In simple words:
  If you increase x by tiny amount,
  how much does y change?

  Notation: dy/dx or f'(x)

WHY IT MATTERS IN ML:
  ML training = finding minimum of loss function
  Derivative tells us which direction to move
  to reduce the loss!

  Like finding the lowest point in a valley
  by always stepping downhill
""")

# Numerical derivative
def numerical_derivative(f, x, h=0.0001):
    return (f(x + h) - f(x - h)) / (2 * h)

# Simple functions and their derivatives
def f1(x): return x ** 2          # derivative = 2x
def f2(x): return x ** 3          # derivative = 3x²
def f3(x): return 2*x + 5         # derivative = 2
def f4(x): return np.sin(x)       # derivative = cos(x)

print("--- Numerical vs Analytical Derivatives ---")
x = 3.0

print(f"\nf(x) = x²  at x={x}")
print(f"  Numerical  : {numerical_derivative(f1, x):.6f}")
print(f"  Analytical : {2*x:.6f}  (formula: 2x)")

print(f"\nf(x) = x³  at x={x}")
print(f"  Numerical  : {numerical_derivative(f2, x):.6f}")
print(f"  Analytical : {3*x**2:.6f}  (formula: 3x²)")

print(f"\nf(x) = 2x+5  at x={x}")
print(f"  Numerical  : {numerical_derivative(f3, x):.6f}")
print(f"  Analytical : {2:.6f}  (formula: constant 2)")

print(f"\nf(x) = sin(x)  at x={x}")
print(f"  Numerical  : {numerical_derivative(f4, x):.6f}")
print(f"  Analytical : {np.cos(x):.6f}  (formula: cos(x))")


# ──────────────────────────────────────────
# PART 2 - GRADIENT
# ──────────────────────────────────────────

print("\n===== PART 2: Gradient =====")

print("""
GRADIENT:
  Vector of partial derivatives
  One derivative for each parameter/weight

  For a function with multiple inputs:
  f(x, y, z) → gradient = [df/dx, df/dy, df/dz]

  Gradient points in direction of STEEPEST INCREASE
  Negative gradient = direction of steepest DECREASE

  In ML:
  Loss function has millions of parameters
  Gradient tells us how to adjust EACH parameter
  to reduce loss!
""")

# Gradient of simple function
def loss_function(w, b, X, y):
    predictions = w * X + b
    loss        = np.mean((predictions - y) ** 2)
    return loss

def compute_gradients(w, b, X, y):
    n           = len(X)
    predictions = w * X + b
    errors      = predictions - y
    dw          = (2/n) * np.dot(errors, X)   # df/dw
    db          = (2/n) * np.sum(errors)       # df/db
    return dw, db

# Sample data
np.random.seed(42)
X = np.random.uniform(1, 10, 50)
y = 3 * X + 7 + np.random.normal(0, 1, 50)

# Initial weights
w, b = 0.0, 0.0

print(f"True relationship  : y = 3x + 7")
print(f"Initial w          : {w}")
print(f"Initial b          : {b}")
print(f"Initial loss       : {loss_function(w, b, X, y):.4f}")

# Compute gradient at initial point
dw, db = compute_gradients(w, b, X, y)
print(f"\nGradient dw        : {dw:.4f}")
print(f"Gradient db        : {db:.4f}")
print(f"(Positive gradient → w and b need to increase)")


# ──────────────────────────────────────────
# PART 3 - GRADIENT DESCENT
# ──────────────────────────────────────────

print("\n===== PART 3: Gradient Descent =====")

print("""
GRADIENT DESCENT:
  Algorithm to find minimum of loss function

  Steps:
  1. Start with random weights
  2. Calculate loss
  3. Calculate gradient (direction of increase)
  4. Move opposite to gradient (downhill)
  5. Repeat until loss stops decreasing

  Update rule:
  w = w - learning_rate * gradient

LEARNING RATE:
  How big each step is
  Too big  → overshoot minimum, loss explodes
  Too small → very slow convergence
  Just right → reaches minimum efficiently
  Typical values: 0.001, 0.01, 0.1
""")

def gradient_descent(X, y, lr=0.01, epochs=1000):
    w, b   = 0.0, 0.0
    losses = []

    for epoch in range(epochs):
        # Forward pass
        loss = loss_function(w, b, X, y)
        losses.append(loss)

        # Compute gradients
        dw, db = compute_gradients(w, b, X, y)

        # Update weights
        w = w - lr * dw
        b = b - lr * db

        # Print progress
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, losses

print("Training with Gradient Descent:")
print(f"{'Epoch':<10} {'Loss':<12} {'w':<10} {'b'}")
print("-" * 45)

w_final, b_final, losses = gradient_descent(X, y, lr=0.01, epochs=1000)

print(f"\n--- Results ---")
print(f"True values      : w=3.0, b=7.0")
print(f"Learned w        : {w_final:.4f}")
print(f"Learned b        : {b_final:.4f}")
print(f"Final loss       : {losses[-1]:.4f}")
print(f"Initial loss     : {losses[0]:.4f}")
print(f"Loss reduced by  : {((losses[0]-losses[-1])/losses[0]*100):.2f}%")


# ──────────────────────────────────────────
# PART 4 - TYPES OF GRADIENT DESCENT
# ──────────────────────────────────────────

print("\n===== PART 4: Types of Gradient Descent =====")

print("""
1. BATCH GRADIENT DESCENT:
   Uses entire dataset for each update
    Stable convergence
    Very slow for large datasets
    Memory intensive

2. STOCHASTIC GRADIENT DESCENT (SGD):
   Uses ONE random sample for each update
    Very fast updates
    Can escape local minima
    Noisy updates, less stable

3. MINI-BATCH GRADIENT DESCENT:
   Uses small batch (32, 64, 128 samples)
    Balance between speed and stability
    Most commonly used in Deep Learning!
    GPU friendly

BATCH SIZE in Deep Learning = Mini-batch siz
When you see batch_size=32 in PyTorch
that is Mini-batch Gradient Descent!
""")

def mini_batch_gd(X, y, lr=0.01, epochs=500, batch_size=16):
    w, b   = 0.0, 0.0
    n      = len(X)
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuf  = X[indices]
        y_shuf  = y[indices]

        epoch_loss = 0

        # Mini batch updates
        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]

            batch_loss  = loss_function(w, b, X_batch, y_batch)
            dw, db      = compute_gradients(w, b, X_batch, y_batch)

            w          -= lr * dw
            b          -= lr * db
            epoch_loss += batch_loss

        losses.append(epoch_loss / (n // batch_size))

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {losses[-1]:.4f} | w: {w:.4f} | b: {b:.4f}")

    return w, b, losses

print("Training with Mini-Batch Gradient Descent:")
print(f"{'Epoch':<10} {'Loss':<12} {'w':<10} {'b'}")
print("-" * 45)

w_mb, b_mb, losses_mb = mini_batch_gd(X, y, lr=0.01, epochs=500, batch_size=16)

print(f"\nMini-Batch Results:")
print(f"True values      : w=3.0, b=7.0")
print(f"Learned w        : {w_mb:.4f}")
print(f"Learned b        : {b_mb:.4f}")


# ──────────────────────────────────────────
# PART 5 - CHAIN RULE
# ──────────────────────────────────────────

print("\n===== PART 5: Chain Rule =====")

print("""
CHAIN RULE:
  How to differentiate composed functions

  If y = f(g(x))
  Then dy/dx = f'(g(x)) * g'(x)

  Or: dy/dx = dy/du * du/dx
  where u = g(x)

WHY CRITICAL IN ML:
  Neural Networks are deeply nested functions:
  Output = f4(f3(f2(f1(input))))

  Backpropagation uses chain rule to compute
  gradients through ALL layers!

  Without chain rule there is no Deep Learning!
""")

# Chain rule example
print("--- Chain Rule Example ---")
print("""
f(x) = (2x + 3)²

Let u = 2x + 3
Let f = u²

du/dx = 2
df/du = 2u

df/dx = df/du * du/dx
      = 2u * 2
      = 2(2x + 3) * 2
      = 4(2x + 3)
""")

def f_composed(x):
    return (2*x + 3) ** 2

def f_analytical_deriv(x):
    return 4 * (2*x + 3)   # chain rule result

x_vals = [1, 2, 3, 4, 5]
print(f"{'x':<6} {'f(x)':<12} {'Numerical':<15} {'Analytical'}")
print("-" * 45)
for x in x_vals:
    num  = numerical_derivative(f_composed, x)
    ana  = f_analytical_deriv(x)
    print(f"{x:<6} {f_composed(x):<12} {num:<15.6f} {ana:.6f}")

# Backpropagation demo
print(f"\n--- Simple Backpropagation Demo ---")
print("""
Neural Network:
  x → [Linear: z = wx + b] → [Sigmoid: a = σ(z)] → Loss

Forward pass:
  x = 2.0, w = 0.5, b = 0.1
  z = wx + b
  a = sigmoid(z)
  Loss = (a - target)²

Backward pass (chain rule):
  dL/dw = dL/da * da/dz * dz/dw
""")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

x      = 2.0
w      = 0.5
b      = 0.1
target = 1.0

# Forward pass
z    = w * x + b
a    = sigmoid(z)
loss = (a - target) ** 2

print(f"Forward Pass:")
print(f"  x       = {x}")
print(f"  w       = {w}")
print(f"  b       = {b}")
print(f"  z = wx+b= {z}")
print(f"  a = σ(z)= {a:.6f}")
print(f"  target  = {target}")
print(f"  Loss    = {loss:.6f}")

# Backward pass (chain rule)
dL_da = 2 * (a - target)                # dLoss/da
da_dz = sigmoid_derivative(z)           # da/dz
dz_dw = x                               # dz/dw
dz_db = 1                               # dz/db

dL_dw = dL_da * da_dz * dz_dw          # chain rule!
dL_db = dL_da * da_dz * dz_db          # chain rule!

print(f"\nBackward Pass (Chain Rule):")
print(f"  dL/da   = {dL_da:.6f}")
print(f"  da/dz   = {da_dz:.6f}")
print(f"  dz/dw   = {dz_dw:.6f}")
print(f"  dL/dw   = dL/da * da/dz * dz/dw = {dL_dw:.6f}")
print(f"  dL/db   = dL/da * da/dz * dz/db = {dL_db:.6f}")

# Update weights
lr    = 0.1
w_new = w - lr * dL_dw
b_new = b - lr * dL_db

print(f"\nWeight Update:")
print(f"  Old w   = {w:.6f} → New w = {w_new:.6f}")
print(f"  Old b   = {b:.6f} → New b = {b_new:.6f}")

# Verify loss decreased
z_new    = w_new * x + b_new
a_new    = sigmoid(z_new)
loss_new = (a_new - target) ** 2

print(f"\nLoss before update : {loss:.6f}")
print(f"Loss after update  : {loss_new:.6f}")
print(f"Loss reduced       : {'✅ Yes!' if loss_new < loss else '❌ No'}")


# ──────────────────────────────────────────
# MINI PROJECT - Gradient Descent from Scratch
# Train Linear Regression manually
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Linear Regression from Scratch =====")
print("(Using only Gradient Descent — no sklearn!)")

np.random.seed(42)
n = 100

# True relationship: score = 5 * study_hours + 40
study_hours  = np.random.uniform(1, 10, n)
exam_scores  = 5 * study_hours + 40 + np.random.normal(0, 3, n)

print(f"Dataset            : {n} students")
print(f"True relationship  : score = 5 * hours + 40")
print(f"Avg study hours    : {np.mean(study_hours):.2f}")
print(f"Avg exam score     : {np.mean(exam_scores):.2f}")

# Normalize features
X_norm = (study_hours - np.mean(study_hours)) / np.std(study_hours)

# Initialize
w, b   = 0.0, 0.0
lr     = 0.01
epochs = 2000
losses = []

print(f"\nTraining...")
print(f"{'Epoch':<10} {'Loss':<15} {'w':<12} {'b'}")
print("-" * 50)

for epoch in range(epochs):
    predictions = w * X_norm + b
    errors      = predictions - exam_scores
    loss        = np.mean(errors ** 2)
    losses.append(loss)

    dw = (2/n) * np.dot(errors, X_norm)
    db = (2/n) * np.sum(errors)

    w -= lr * dw
    b -= lr * db

    if epoch % 400 == 0:
        print(f"{epoch:<10} {loss:<15.4f} {w:<12.4f} {b:.4f}")

print(f"\n--- Final Results ---")
print(f"Learned w          : {w:.4f}")
print(f"Learned b          : {b:.4f}")
print(f"Final loss         : {losses[-1]:.4f}")
print(f"Loss reduction     : {((losses[0]-losses[-1])/losses[0]*100):.2f}%")

# Predict
print(f"\n--- Predictions ---")
test_hours = [2, 5, 8, 10]
print(f"{'Hours':<10} {'Predicted':>12} {'True Score':>12}")
print("-" * 36)
for h in test_hours:
    h_norm    = (h - np.mean(study_hours)) / np.std(study_hours)
    predicted = w * h_norm + b
    true_val  = 5 * h + 40
    print(f"{h:<10} {predicted:>12.2f} {true_val:>12.2f}")


print("\n===== WHAT I LEARNED TODAY =====")
print(" Derivatives - rate of change")
print(" Gradient - direction of steepest change")
print(" Gradient Descent - find minimum of loss")
print(" Learning Rate - step size importance")
print(" Batch vs SGD vs Mini-Batch GD")
print(" Chain Rule - backbone of backpropagation")
print(" Backpropagation - how Neural Nets learn")
print(" Mini Project - Linear Regression from scratch!")
print("\n Day 12 Done! Tomorrow - Build ML Model on Titanic!")