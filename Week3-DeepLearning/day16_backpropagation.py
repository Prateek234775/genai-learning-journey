# ============================================
# DAY 16 - Backpropagation & Gradient Descent
# How Neural Networks Actually Learn
# Author: Prateek Kumar Kuntal
# Date: 20 May 2025
# ============================================

import numpy as np


# ──────────────────────────────────────────
# PART 1 - WHAT IS BACKPROPAGATION
# ──────────────────────────────────────────

print("===== PART 1: What is Backpropagation =====")

print("""
BACKPROPAGATION:
  Algorithm to compute gradients in neural networks
  Uses chain rule to propagate error backwards
  From output layer → hidden layers → input layer

TRAINING LOOP:
  1. Forward Pass  → compute predictions
  2. Compute Loss  → how wrong are we?
  3. Backward Pass → compute gradients (backprop)
  4. Update Weights→ gradient descent step
  5. Repeat until loss is small enough

WHY BACKWARD?
  We know the error at output
  We need to know HOW MUCH each weight
  contributed to that error
  Chain rule lets us compute this efficiently!
""")


# ──────────────────────────────────────────
# PART 2 - FORWARD PASS IN DETAIL
# ──────────────────────────────────────────

print("===== PART 2: Forward Pass in Detail =====")

np.random.seed(42)

# Simple 2 layer network
# Input(2) → Hidden(3) → Output(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Single sample forward pass
x  = np.array([2.0, 3.0])
y  = np.array([1.0])

# Weights (random init)
W1 = np.array([[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]])
b1 = np.array([0.1, 0.1, 0.1])

W2 = np.array([[0.7],
                [0.8],
                [0.9]])
b2 = np.array([0.1])

print("--- Network Architecture ---")
print(f"Input            : {x}")
print(f"Target           : {y}")
print(f"W1 shape         : {W1.shape}  (2 inputs → 3 neurons)")
print(f"W2 shape         : {W2.shape}  (3 neurons → 1 output)")

# Forward pass step by step
z1 = x @ W1 + b1
a1 = relu(z1)

z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

loss = mse_loss(y, a2)

print(f"\n--- Forward Pass ---")
print(f"z1 = x @ W1 + b1 : {z1.round(4)}")
print(f"a1 = relu(z1)    : {a1.round(4)}")
print(f"z2 = a1 @ W2 + b2: {z2.round(4)}")
print(f"a2 = sigmoid(z2) : {a2.round(4)}")
print(f"Loss             : {loss:.6f}")


# ──────────────────────────────────────────
# PART 3 - BACKWARD PASS IN DETAIL
# ──────────────────────────────────────────

print("\n===== PART 3: Backward Pass in Detail =====")

print("""
CHAIN RULE APPLICATION:
  dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2
  dL/dW1 = dL/da2 * da2/dz2 * dz2/da1
                             * da1/dz1 * dz1/dW1

We compute gradients from OUTPUT → INPUT
""")

# Backward pass step by step
print("--- Backward Pass ---")

# Step 1 — gradient of loss w.r.t output
dL_da2 = 2 * (a2 - y) / y.size
print(f"Step 1: dL/da2   : {dL_da2.round(6)}")

# Step 2 — gradient through sigmoid
da2_dz2 = sigmoid_derivative(z2)
dL_dz2  = dL_da2 * da2_dz2
print(f"Step 2: da2/dz2  : {da2_dz2.round(6)}")
print(f"        dL/dz2   : {dL_dz2.round(6)}")

# Step 3 — gradient w.r.t W2 and b2
dL_dW2 = a1.reshape(-1, 1) @ dL_dz2.reshape(1, -1)
dL_db2 = dL_dz2
print(f"Step 3: dL/dW2   :\n{dL_dW2.round(6)}")
print(f"        dL/db2   : {dL_db2.round(6)}")

# Step 4 — gradient through layer 1
dL_da1  = dL_dz2 @ W2.T
da1_dz1 = relu_derivative(z1)
dL_dz1  = dL_da1 * da1_dz1
print(f"Step 4: dL/da1   : {dL_da1.round(6)}")
print(f"        da1/dz1  : {da1_dz1.round(6)}")
print(f"        dL/dz1   : {dL_dz1.round(6)}")

# Step 5 — gradient w.r.t W1 and b1
dL_dW1 = x.reshape(-1, 1) @ dL_dz1.reshape(1, -1)
dL_db1 = dL_dz1
print(f"Step 5: dL/dW1   :\n{dL_dW1.round(6)}")
print(f"        dL/db1   : {dL_db1.round(6)}")

# Update weights
lr     = 0.1
W1_new = W1 - lr * dL_dW1
b1_new = b1 - lr * dL_db1
W2_new = W2 - lr * dL_dW2
b2_new = b2 - lr * dL_db2

# Verify loss decreased
z1_new   = x @ W1_new + b1_new
a1_new   = relu(z1_new)
z2_new   = a1_new @ W2_new + b2_new
a2_new   = sigmoid(z2_new)
loss_new = mse_loss(y, a2_new)

print(f"\n--- Weight Update ---")
print(f"Old Loss         : {loss:.6f}")
print(f"New Loss         : {loss_new:.6f}")
print(f"Loss Reduced     : {'✅ Yes!' if loss_new < loss else '❌ No'}")


# ──────────────────────────────────────────
# PART 4 - OPTIMIZERS
# ──────────────────────────────────────────

print("\n===== PART 4: Optimizers =====")

print("""
OPTIMIZERS:
  Different strategies for gradient descent

1. SGD (Stochastic Gradient Descent)
   w = w - lr * gradient
   Simple but slow, can oscillate

2. SGD with Momentum
   Builds up velocity in consistent direction
   Faster convergence, less oscillation

3. RMSprop
   Adapts learning rate per parameter
   Good for RNNs

4. ADAM (Adaptive Moment Estimation)
   Combines momentum + RMSprop
   Most popular optimizer!
   Works well for almost everything
   Default choice in PyTorch/TensorFlow
""")

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param, grad):
        return param - self.lr * grad

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, param, grad, key):
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)
        self.velocity[key] = (self.momentum * self.velocity[key]
                               - self.lr * grad)
        return param + self.velocity[key]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9,
                 beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = {}
        self.v     = {}
        self.t     = 0

    def update(self, param, grad, key):
        self.t += 1
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Compare optimizers
def train_with_optimizer(optimizer_name, epochs=500):
    np.random.seed(42)
    w = np.array([0.0, 0.0])
    b = 0.0

    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    if optimizer_name == "sgd":
        opt = SGD(lr=0.01)
    elif optimizer_name == "momentum":
        opt = SGDMomentum(lr=0.01)
    else:
        opt = Adam(lr=0.01)

    losses = []
    for epoch in range(epochs):
        pred   = sigmoid(X @ w + b)
        loss   = binary_cross_entropy_loss(y, pred)
        losses.append(loss)

        grad_w = X.T @ (pred - y) / len(y)
        grad_b = np.mean(pred - y)

        if optimizer_name == "sgd":
            w = opt.update(w, grad_w)
            b = opt.update(b, grad_b)
        elif optimizer_name == "momentum":
            w = opt.update(w, grad_w, "w")
            b = opt.update(b, grad_b, "b")
        else:
            w = opt.update(w, grad_w, "w")
            b = opt.update(b, grad_b, "b")

    return losses

def binary_cross_entropy_loss(y_true, y_pred):
    eps    = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))

print("Comparing Optimizers (500 epochs):")
print(f"{'Optimizer':<15} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction'}")
print("-" * 60)

for opt_name in ["sgd", "momentum", "adam"]:
    losses  = train_with_optimizer(opt_name)
    initial = losses[0]
    final   = losses[-1]
    reduction = ((initial - final) / initial * 100)
    print(f"{opt_name:<15} {initial:<15.6f} {final:<15.6f} {reduction:.2f}%")

print("\n🏆 Adam usually converges fastest and most reliably!")


# ──────────────────────────────────────────
# PART 5 - LEARNING RATE EFFECT
# ──────────────────────────────────────────

print("\n===== PART 5: Learning Rate Effect =====")

print("""
LEARNING RATE MATTERS A LOT:
  Too high → loss explodes or oscillates
  Too low  → too slow to converge
  Just right → converges smoothly

COMMON VALUES:
  0.1    → often too high for deep nets
  0.01   → good starting point
  0.001  → Adam default, works well
  0.0001 → conservative, use for fine tuning
""")

def train_lr_test(lr, epochs=300):
    np.random.seed(42)
    w      = 0.5
    losses = []

    for _ in range(epochs):
        y_pred = sigmoid(np.array([w]))
        y_true = np.array([1.0])
        loss   = mse_loss(y_true, y_pred)
        losses.append(loss)

        grad   = 2 * (sigmoid(np.array([w])) - y_true) * sigmoid_derivative(np.array([w]))
        w      = w - lr * grad[0]

        if np.isnan(loss) or loss > 1e6:
            return losses, "💥 EXPLODED"

    return losses, f"Final: {losses[-1]:.6f}"

print(f"{'Learning Rate':<15} {'Result':<20} {'Status'}")
print("-" * 50)
for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
    losses, status = train_lr_test(lr)
    valid_losses   = [l for l in losses if not np.isnan(l) and l < 1e6]
    init           = losses[0]
    print(f"{lr:<15} {status:<20} {'✅' if 'Final' in status else '❌'}")


# ──────────────────────────────────────────
# MINI PROJECT - Full Neural Network
# with proper Backpropagation
# Train on student dataset
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Full Neural Network with Backprop =====")

np.random.seed(42)
n = 300

# Dataset
X_raw = np.column_stack([
    np.random.uniform(1, 10, n),    # study hours
    np.random.uniform(50, 100, n),  # attendance
    np.random.uniform(40, 100, n),  # assignment score
])
y_raw = ((X_raw[:, 0] * 5 +
          X_raw[:, 1] * 0.3 +
          X_raw[:, 2] * 0.2) > 75).astype(float)

# Normalize
X_norm = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

# Split
split        = int(0.8 * n)
X_train      = X_norm[:split]
y_train      = y_raw[:split].reshape(-1, 1)
X_test       = X_norm[split:]
y_test       = y_raw[split:].reshape(-1, 1)

print(f"Dataset          : {n} students")
print(f"Features         : Study Hours, Attendance, Assignment Score")
print(f"Train size       : {len(X_train)}")
print(f"Test size        : {len(X_test)}")

# Network: 3 → 8 → 4 → 1
np.random.seed(42)
params = {
    "W1": np.random.randn(3, 8) * 0.1,
    "b1": np.zeros((1, 8)),
    "W2": np.random.randn(8, 4) * 0.1,
    "b2": np.zeros((1, 4)),
    "W3": np.random.randn(4, 1) * 0.1,
    "b3": np.zeros((1, 1)),
}

adam   = Adam(lr=0.01)
losses = []
epochs = 1000

print(f"\nTraining 3-layer Neural Network...")
print(f"Architecture     : 3 → 8 → 4 → 1")
print(f"Optimizer        : Adam")
print(f"Epochs           : {epochs}")
print()
print(f"{'Epoch':<10} {'Loss':<15} {'Train Acc':<15} {'Test Acc'}")
print("-" * 55)

for epoch in range(epochs):
    # Forward
    z1 = X_train @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    a3 = sigmoid(z3)

    # Loss
    loss = binary_cross_entropy_loss(y_train, a3)
    losses.append(loss)

    # Backward
    m      = len(X_train)
    dL_dz3 = (a3 - y_train) / m
    dW3    = a2.T @ dL_dz3
    db3    = np.sum(dL_dz3, axis=0, keepdims=True)

    dL_da2 = dL_dz3 @ params["W3"].T
    dL_dz2 = dL_da2 * relu_derivative(z2)
    dW2    = a1.T @ dL_dz2
    db2    = np.sum(dL_dz2, axis=0, keepdims=True)

    dL_da1 = dL_dz2 @ params["W2"].T
    dL_dz1 = dL_da1 * relu_derivative(z1)
    dW1    = X_train.T @ dL_dz1
    db1    = np.sum(dL_dz1, axis=0, keepdims=True)

    # Update with Adam
    params["W3"] = adam.update(params["W3"], dW3, "W3")
    params["b3"] = adam.update(params["b3"], db3, "b3")
    params["W2"] = adam.update(params["W2"], dW2, "W2")
    params["b2"] = adam.update(params["b2"], db2, "b2")
    params["W1"] = adam.update(params["W1"], dW1, "W1")
    params["b1"] = adam.update(params["b1"], db1, "b1")

    if epoch % 200 == 0:
        # Train accuracy
        train_pred = (a3 > 0.5).astype(float)
        train_acc  = np.mean(train_pred == y_train)

        # Test accuracy
        tz1 = X_test @ params["W1"] + params["b1"]
        ta1 = relu(tz1)
        tz2 = ta1 @ params["W2"] + params["b2"]
        ta2 = relu(tz2)
        tz3 = ta2 @ params["W3"] + params["b3"]
        ta3 = sigmoid(tz3)
        test_acc = np.mean((ta3 > 0.5) == y_test)

        print(f"{epoch:<10} {loss:<15.6f} {train_acc:<15.4f} {test_acc:.4f}")

# Final test
tz1       = X_test @ params["W1"] + params["b1"]
ta1       = relu(tz1)
tz2       = ta1 @ params["W2"] + params["b2"]
ta2       = relu(tz2)
tz3       = ta2 @ params["W3"] + params["b3"]
ta3       = sigmoid(tz3)
final_acc = np.mean((ta3 > 0.5) == y_test)

print(f"\n--- Final Results ---")
print(f"Initial Loss     : {losses[0]:.6f}")
print(f"Final Loss       : {losses[-1]:.6f}")
print(f"Final Test Acc   : {final_acc*100:.2f}%")
print(f"\n✅ Full Neural Network with Adam optimizer trained!")
print(f"✅ Backpropagation computed manually step by step!")


print("\n===== WHAT I LEARNED TODAY =====")
print("✅ Backpropagation - chain rule in action")
print("✅ Forward Pass - step by step")
print("✅ Backward Pass - gradient computation")
print("✅ Optimizers - SGD, Momentum, Adam")
print("✅ Learning Rate - effect on training")
print("✅ Mini Project - Full NN with Adam + Backprop")
print("\n🚀 Day 16 Done! Tomorrow - PyTorch Basics!")