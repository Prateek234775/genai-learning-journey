# ============================================
# DAY 15 - Neural Networks
# Neurons, Layers, Activation Functions
# Author: Prateek Kumar Kuntal
# Date: 19 May 2025
# ============================================

import numpy as np


# ──────────────────────────────────────────
# PART 1 - WHAT IS A NEURAL NETWORK
# ──────────────────────────────────────────
 
print("===== PART 1: What is a Neural Network =====")

print("""
NEURAL NETWORK:
  Inspired by human brain neurons
  Layers of connected nodes (neurons)
  Each connection has a weight
  Model learns by adjusting weights

STRUCTURE:
  Input Layer   → receives raw data
  Hidden Layers → extract patterns
  Output Layer  → final prediction

Example — Image Recognition:
  Input  : 784 pixels (28x28 image)
  Hidden : 128 neurons → 64 neurons
  Output : 10 neurons (digits 0-9)

WHY BETTER THAN ML?
   Learns complex non-linear patterns
   Works with images, text, audio
   Automatically extracts features
   Scales with more data and compute
""")


# ──────────────────────────────────────────
# PART 2 - SINGLE NEURON
# ──────────────────────────────────────────

print("===== PART 2: Single Neuron =====")

print("""
SINGLE NEURON:
  1. Takes inputs
  2. Multiplies by weights
  3. Adds bias
  4. Passes through activation function
  5. Produces output

  z = w1*x1 + w2*x2 + ... + b
  output = activation(z)
""")

def single_neuron(inputs, weights, bias, activation="relu"):
    # Step 1 — weighted sum
    z = np.dot(inputs, weights) + bias

    # Step 2 — activation
    if activation == "relu":
        output = max(0, z)
    elif activation == "sigmoid":
        output = 1 / (1 + np.exp(-z))
    elif activation == "tanh":
        output = np.tanh(z)
    else:
        output = z   # linear

    return z, output

# Example neuron
inputs  = np.array([2.0, 3.0, 1.5])
weights = np.array([0.5, -0.3, 0.8])
bias    = 0.1

print("--- Single Neuron Example ---")
print(f"Inputs   : {inputs}")
print(f"Weights  : {weights}")
print(f"Bias     : {bias}")

for act in ["relu", "sigmoid", "tanh", "linear"]:
    z, out = single_neuron(inputs, weights, bias, act)
    print(f"\nActivation: {act}")
    print(f"  z (weighted sum) : {z:.4f}")
    print(f"  output           : {out:.4f}")


# ──────────────────────────────────────────
# PART 3 - ACTIVATION FUNCTIONS
# ──────────────────────────────────────────

print("\n===== PART 3: Activation Functions =====")

print("""
WHY ACTIVATION FUNCTIONS?
  Without them neural network = just linear model
  Activation adds non-linearity
  Allows learning complex patterns

TYPES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SIGMOID
   Output: 0 to 1
   Used for: Binary classification output
   Problem: Vanishing gradient for deep nets

2. TANH
   Output: -1 to 1
   Better than sigmoid (zero centered)
   Still has vanishing gradient problem

3. RELU (Rectified Linear Unit)
   Output: 0 to infinity
   Most popular for hidden layers!
   Fast, simple, works great in practice
   Problem: Dead neurons (always 0)

4. LEAKY RELU
   Fixes dead neuron problem
   Small negative slope for x < 0

5. SOFTMAX
   Output: probabilities summing to 1
   Used for multi-class output layer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

# All activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

print(f"Input x      : {x}")
print(f"\nSigmoid      : {sigmoid(x).round(4)}")
print(f"Tanh         : {tanh(x).round(4)}")
print(f"ReLU         : {relu(x).round(4)}")
print(f"Leaky ReLU   : {leaky_relu(x).round(4)}")
print(f"Softmax      : {softmax(x).round(4)}")
print(f"Softmax sum  : {softmax(x).sum():.4f}  (always 1!)")

print("""
WHICH TO USE WHERE:
  Hidden Layers → ReLU (default choice)
  Output Binary → Sigmoid
  Output Multi  → Softmax
  Output Number → Linear (no activation)
""")


# ──────────────────────────────────────────
# PART 4 - NEURAL NETWORK LAYER
# ──────────────────────────────────────────

print("===== PART 4: Neural Network Layer =====")

print("""
LAYER:
  Collection of neurons
  Each neuron connected to all inputs
  Called DENSE or FULLY CONNECTED layer

  Input  shape: (batch, features)
  Weight shape: (features, neurons)
  Output shape: (batch, neurons)
""")

np.random.seed(42)

# Layer forward pass
def dense_layer(inputs, weights, bias, activation="relu"):
    z      = inputs @ weights + bias
    if activation == "relu":
        return relu(z)
    elif activation == "sigmoid":
        return sigmoid(z)
    elif activation == "softmax":
        return softmax(z)
    return z

# 3 layer network
batch    = 4
features = 3

# Input data (4 samples, 3 features)
X = np.random.randn(batch, features)

# Layer 1: 3 → 4 neurons
W1 = np.random.randn(features, 4) * 0.1
b1 = np.zeros(4)

# Layer 2: 4 → 3 neurons
W2 = np.random.randn(4, 3) * 0.1
b2 = np.zeros(3)

# Output layer: 3 → 2 neurons
W3 = np.random.randn(3, 2) * 0.1
b3 = np.zeros(2)

# Forward pass
print("--- Forward Pass through 3 layers ---")
print(f"Input shape      : {X.shape}")

h1 = dense_layer(X, W1, b1, "relu")
print(f"After Layer 1    : {h1.shape}  (ReLU)")

h2 = dense_layer(h1, W2, b2, "relu")
print(f"After Layer 2    : {h2.shape}  (ReLU)")

output = dense_layer(h2, W3, b3, "sigmoid")
print(f"Output Layer     : {output.shape}  (Sigmoid)")

print(f"\nInput:\n{X.round(4)}")
print(f"\nHidden Layer 1:\n{h1.round(4)}")
print(f"\nHidden Layer 2:\n{h2.round(4)}")
print(f"\nOutput:\n{output.round(4)}")


# ──────────────────────────────────────────
# PART 5 - LOSS FUNCTIONS
# ──────────────────────────────────────────

print("\n===== PART 5: Loss Functions =====")

print("""
LOSS FUNCTION:
  Measures how wrong the model is
  Goal of training = minimize loss

TYPES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MSE (Mean Squared Error)
   For regression problems
   loss = mean((predicted - actual)²)

2. Binary Cross Entropy
   For binary classification
   loss = -mean(y*log(p) + (1-y)*log(1-p))

3. Categorical Cross Entropy
   For multi-class classification
   loss = -sum(y * log(p))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# MSE Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross Entropy
def binary_cross_entropy(y_true, y_pred):
    eps    = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross Entropy
def categorical_cross_entropy(y_true, y_pred):
    eps    = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.sum(y_true * np.log(y_pred))

print("--- MSE Loss ---")
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])
print(f"True     : {y_true}")
print(f"Predicted: {y_pred}")
print(f"MSE Loss : {mse_loss(y_true, y_pred):.6f}")

print("\n--- Binary Cross Entropy ---")
y_true_bin = np.array([1, 0, 1, 1, 0])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
print(f"True     : {y_true_bin}")
print(f"Predicted: {y_pred_bin}")
print(f"BCE Loss : {binary_cross_entropy(y_true_bin, y_pred_bin):.6f}")

print("\n--- Categorical Cross Entropy ---")
y_true_cat = np.array([0, 1, 0])    # one hot — class 1
y_pred_cat = np.array([0.1, 0.8, 0.1])
print(f"True     : {y_true_cat}")
print(f"Predicted: {y_pred_cat}")
print(f"CCE Loss : {categorical_cross_entropy(y_true_cat, y_pred_cat):.6f}")


# ──────────────────────────────────────────
# MINI PROJECT - Neural Network from Scratch
# Binary Classification — Pass or Fail
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: Neural Network from Scratch =====")
print("(No PyTorch, No TensorFlow — pure NumPy!)")

np.random.seed(42)
n = 200

# Dataset — predict pass/fail from study hours + attendance
X_raw = np.random.randn(n, 2)
y_raw = ((X_raw[:, 0] + X_raw[:, 1]) > 0).astype(float)

# Normalize
X_data = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

# Split
split        = int(0.8 * n)
X_train      = X_data[:split]
y_train      = y_raw[:split].reshape(-1, 1)
X_test       = X_data[split:]
y_test       = y_raw[split:].reshape(-1, 1)

print(f"Train size       : {X_train.shape}")
print(f"Test size        : {X_test.shape}")

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a1, a2

def relu(x):
    return np.maximum(0, x)

lr     = 0.1
epochs = 1000
losses = []

print(f"\nTraining Neural Network...")
print(f"{'Epoch':<10} {'Loss':<15} {'Train Acc':<15} {'Test Acc'}")
print("-" * 55)

for epoch in range(epochs):
    # Forward pass
    a1, a2 = forward(X_train)

    # Loss
    loss = binary_cross_entropy(y_train, a2)
    losses.append(loss)

    # Backward pass
    m      = X_train.shape[0]
    dL_da2 = -(y_train/a2 - (1-y_train)/(1-a2+1e-15)) / m
    da2_dz2= a2 * (1 - a2)
    dz2    = dL_da2 * da2_dz2

    dW2    = a1.T @ dz2
    db2    = np.sum(dz2, axis=0, keepdims=True)

    da1    = dz2 @ W2.T
    dz1    = da1 * (a1 > 0)

    dW1    = X_train.T @ dz1
    db1    = np.sum(dz1, axis=0, keepdims=True)

    # Update
    W2    -= lr * dW2
    b2    -= lr * db2
    W1    -= lr * dW1
    b1    -= lr * db1

    if epoch % 200 == 0:
        _, train_pred = forward(X_train)
        _, test_pred  = forward(X_test)
        train_acc = np.mean((train_pred > 0.5) == y_train)
        test_acc  = np.mean((test_pred  > 0.5) == y_test)
        print(f"{epoch:<10} {loss:<15.6f} {train_acc:<15.4f} {test_acc:.4f}")

# Final results
_, final_pred = forward(X_test)
final_acc     = np.mean((final_pred > 0.5) == y_test)

print(f"\n--- Final Results ---")
print(f"Initial Loss     : {losses[0]:.6f}")
print(f"Final Loss       : {losses[-1]:.6f}")
print(f"Test Accuracy    : {final_acc*100:.2f}%")
print(f"\n Neural Network trained from scratch using only NumPy!")


print("\n===== WHAT I LEARNED TODAY =====")
print(" What is a Neural Network")
print(" Single Neuron - weighted sum + activation")
print(" Activation Functions - ReLU, Sigmoid, Softmax")
print(" Dense Layers - forward pass")
print(" Loss Functions - MSE, BCE, CCE")
print(" Mini Project - Full NN from scratch!")
print("\n Day 15 Done! Tomorrow - Backpropagation & Gradient Descent in PyTorch!")