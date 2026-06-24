# ============================================
# DAY 17 - PyTorch Basics
# Tensors, Autograd, First Neural Network
# Author: Prateek Kumar Kuntal
# Date: 21 May 2026
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ──────────────────────────────────────────
# PART 1 - WHAT IS PYTORCH
# ──────────────────────────────────────────

print("===== PART 1: What is PyTorch =====")

print("""
PYTORCH:
  Deep Learning framework by Meta/Facebook
  Most popular for research and industry
  Used by Google, Microsoft, Tesla, OpenAI

WHY PYTORCH?
  ✅ Dynamic computation graph
  ✅ Pythonic and easy to debug
  ✅ Autograd — automatic differentiation
  ✅ GPU acceleration
  ✅ Huge community and ecosystem
  ✅ HuggingFace, LangChain all use PyTorch

PYTORCH vs TENSORFLOW:
  PyTorch    → Research + Industry (growing)
  TensorFlow → Production (older, still used)
  PyTorch is now the DEFAULT choice!
""")

print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA available   : {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device     : {device}")


# ──────────────────────────────────────────
# PART 2 - TENSORS
# ──────────────────────────────────────────

print("\n===== PART 2: Tensors =====")

print("""
TENSOR:
  Core data structure in PyTorch
  Like NumPy arrays but with superpowers:
  ✅ Can run on GPU
  ✅ Tracks gradients automatically
  ✅ Used for all computations in neural nets
""")

# Creating tensors
t1 = torch.tensor([1, 2, 3, 4, 5])
t2 = torch.tensor([1.0, 2.0, 3.0])
t3 = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])

print(f"Integer tensor   : {t1}")
print(f"Float tensor     : {t2}")
print(f"2D tensor:\n{t3}")
print(f"\nt1 dtype         : {t1.dtype}")
print(f"t2 dtype         : {t2.dtype}")
print(f"t3 shape         : {t3.shape}")
print(f"t3 size          : {t3.size()}")
print(f"t3 ndim          : {t3.ndim}")

# Special tensors
zeros   = torch.zeros(3, 4)
ones    = torch.ones(2, 3)
eye     = torch.eye(4)
rand    = torch.rand(3, 3)
randn   = torch.randn(3, 3)
arange  = torch.arange(0, 10, 2)
linspace= torch.linspace(0, 1, 5)

print(f"\nZeros:\n{zeros}")
print(f"\nOnes:\n{ones}")
print(f"\nRandom (0-1):\n{rand.round(decimals=4)}")
print(f"\nArange        : {arange}")
print(f"Linspace      : {linspace}")

# From NumPy
np_arr  = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_arr)
print(f"\nFrom NumPy    : {t_from_np}")

# To NumPy
back_to_np = t_from_np.numpy()
print(f"Back to NumPy : {back_to_np}")


# ──────────────────────────────────────────
# PART 3 - TENSOR OPERATIONS
# ──────────────────────────────────────────

print("\n===== PART 3: Tensor Operations =====")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a              : {a}")
print(f"b              : {b}")
print(f"a + b          : {a + b}")
print(f"a - b          : {a - b}")
print(f"a * b          : {a * b}")
print(f"a / b          : {a / b}")
print(f"a ** 2         : {a ** 2}")
print(f"torch.dot(a,b) : {torch.dot(a, b)}")
print(f"torch.sum(a)   : {torch.sum(a)}")
print(f"torch.mean(a)  : {torch.mean(a)}")
print(f"torch.max(a)   : {torch.max(a)}")

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 2)
C = torch.mm(A, B)       # matrix multiply

print(f"\nMatrix A shape : {A.shape}")
print(f"Matrix B shape : {B.shape}")
print(f"A @ B shape    : {C.shape}")

# Reshaping
t = torch.arange(1, 13)
print(f"\nOriginal       : {t}")
print(f"Reshape 3x4:\n{t.reshape(3, 4)}")
print(f"Reshape 2x6:\n{t.reshape(2, 6)}")
print(f"Flatten        : {t.reshape(3,4).flatten()}")

# Indexing and slicing
m = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=torch.float32)

print(f"\nMatrix:\n{m}")
print(f"Row 0          : {m[0]}")
print(f"Col 1          : {m[:, 1]}")
print(f"Element [1,2]  : {m[1, 2]}")
print(f"Sub-matrix:\n{m[0:2, 1:3]}")


# ──────────────────────────────────────────
# PART 4 - AUTOGRAD
# (Automatic Differentiation — Magic of PyTorch!)
# ──────────────────────────────────────────

print("\n===== PART 4: Autograd =====")

print("""
AUTOGRAD:
  PyTorch automatically computes gradients!
  No more manual chain rule calculations
  Just call .backward() and gradients appear

  This is why PyTorch is so powerful:
  You define the forward pass
  PyTorch handles the backward pass!
""")

# Simple autograd example
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2*x + 1   # y = x² + 2x + 1

print(f"x              : {x}")
print(f"y = x²+2x+1   : {y}")

# Compute gradient dy/dx
y.backward()
print(f"dy/dx          : {x.grad}")  # should be 2x + 2 = 8

# Verify analytically
print(f"Analytical     : {2*3 + 2} (2x+2 at x=3)")

# Multi-variable autograd
print(f"\n--- Multi-variable Autograd ---")
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
z  = x1**2 + x1*x2 + x2**2

z.backward()
print(f"z = x1² + x1*x2 + x2²")
print(f"x1             : {x1.item()}")
print(f"x2             : {x2.item()}")
print(f"z              : {z.item()}")
print(f"dz/dx1         : {x1.grad.item()}  (analytical: 2*x1+x2 = {2*2+3})")
print(f"dz/dx2         : {x2.grad.item()}  (analytical: x1+2*x2 = {2+2*3})")

# Gradient in neural network weights
print(f"\n--- Autograd for Neural Network ---")
W = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)
X = torch.randn(4, 3)
y = torch.randint(0, 2, (4,)).float()

# Forward pass
out  = X @ W + b
loss = torch.mean(out ** 2)

# Backward pass — PyTorch handles everything!
loss.backward()

print(f"W shape        : {W.shape}")
print(f"W.grad shape   : {W.grad.shape}")
print(f"b.grad         : {b.grad}")
print(f"✅ Gradients computed automatically!")


# ──────────────────────────────────────────
# PART 5 - BUILDING NEURAL NETWORK
# using nn.Module
# ──────────────────────────────────────────

print("\n===== PART 5: Building Neural Network with nn.Module =====")

print("""
nn.Module:
  Base class for all neural networks in PyTorch
  Define layers in __init__
  Define forward pass in forward()
  PyTorch handles backward automatically!
""")

class StudentPassPredictor(nn.Module):
    def __init__(self, input_size, hidden1,
                 hidden2, output_size):
        super(StudentPassPredictor, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, output_size)

        # Activation functions
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Forward pass
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x

# Create model
model = StudentPassPredictor(
    input_size=3,
    hidden1=16,
    hidden2=8,
    output_size=1
)

print("Model Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
print(f"\nTotal Parameters    : {total_params}")
print(f"Trainable Parameters: {trainable}")

# Test forward pass
sample_input = torch.randn(4, 3)
output       = model(sample_input)
print(f"\nSample input shape  : {sample_input.shape}")
print(f"Output shape        : {output.shape}")
print(f"Output values       : {output.detach().numpy().flatten().round(4)}")


# ──────────────────────────────────────────
# PART 6 - TRAINING LOOP IN PYTORCH
# ──────────────────────────────────────────

print("\n===== PART 6: Training Loop in PyTorch =====")

print("""
STANDARD PYTORCH TRAINING LOOP:
  for epoch in range(epochs):
    1. Forward pass  → model(X)
    2. Compute loss  → criterion(pred, y)
    3. Zero gradients→ optimizer.zero_grad()
    4. Backward pass → loss.backward()
    5. Update weights→ optimizer.step()
""")

# Generate dataset
torch.manual_seed(42)
np.random.seed(42)

n = 300
X_np = np.column_stack([
    np.random.uniform(1, 10, n),
    np.random.uniform(50, 100, n),
    np.random.uniform(40, 100, n),
])
y_np = ((X_np[:,0]*5 +
         X_np[:,1]*0.3 +
         X_np[:,2]*0.2) > 75).astype(float)

# Normalize
X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)

# Convert to tensors
X_tensor = torch.FloatTensor(X_np)
y_tensor = torch.FloatTensor(y_np).unsqueeze(1)

# Split
split   = int(0.8 * n)
X_train = X_tensor[:split]
y_train = y_tensor[:split]
X_test  = X_tensor[split:]
y_test  = y_tensor[split:]

# Model, loss, optimizer
model     = StudentPassPredictor(3, 16, 8, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training PyTorch Neural Network...")
print(f"{'Epoch':<10} {'Loss':<15} {'Train Acc':<15} {'Test Acc'}")
print("-" * 55)

epochs     = 1000
train_losses = []

for epoch in range(epochs):
    # Training mode
    model.train()

    # Forward pass
    y_pred = model(X_train)
    loss   = criterion(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()    # clear old gradients
    loss.backward()          # compute gradients
    optimizer.step()         # update weights

    train_losses.append(loss.item())

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred  = model(X_test)

            train_acc  = ((train_pred > 0.5) ==
                           y_train).float().mean()
            test_acc   = ((test_pred > 0.5) ==
                           y_test).float().mean()

        print(f"{epoch:<10} {loss.item():<15.6f} "
              f"{train_acc.item():<15.4f} {test_acc.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    final_pred = model(X_test)
    final_acc  = ((final_pred > 0.5) == y_test).float().mean()

print(f"\n--- Final Results ---")
print(f"Initial Loss     : {train_losses[0]:.6f}")
print(f"Final Loss       : {train_losses[-1]:.6f}")
print(f"Final Test Acc   : {final_acc.item()*100:.2f}%")


# ──────────────────────────────────────────
# MINI PROJECT - PyTorch Loan Approver
# Full training pipeline with DataLoader
# ──────────────────────────────────────────

print("\n===== MINI PROJECT: PyTorch Loan Approver =====")

from torch.utils.data import DataLoader, TensorDataset

# Dataset
np.random.seed(42)
n = 500

X_loan = np.column_stack([
    np.random.randint(22, 60, n),           # age
    np.random.uniform(20000, 200000, n),    # income
    np.random.uniform(50000, 1000000, n),   # loan amount
    np.random.randint(300, 850, n),         # credit score
    np.random.randint(0, 30, n),            # experience
    np.random.randint(0, 5, n),             # existing loans
])

approval = (
    X_loan[:,3] * 0.4 +
    X_loan[:,1] * 0.0003 +
    X_loan[:,4] * 2 -
    X_loan[:,5] * 10
)
y_loan = (approval > approval.mean()).astype(float)

# Normalize
X_loan = (X_loan - X_loan.mean(axis=0)) / X_loan.std(axis=0)

# Tensors
X_t = torch.FloatTensor(X_loan)
y_t = torch.FloatTensor(y_loan).unsqueeze(1)

# Split
split    = int(0.8 * n)
X_tr, X_te = X_t[:split], X_t[split:]
y_tr, y_te = y_t[:split], y_t[split:]

# DataLoader (batching made easy!)
train_dataset = TensorDataset(X_tr, y_tr)
test_dataset  = TensorDataset(X_te, y_te)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=32)

print(f"Dataset          : {n} loan applications")
print(f"Train batches    : {len(train_loader)}")
print(f"Test batches     : {len(test_loader)}")

# Model
class LoanApprover(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

loan_model = LoanApprover()
criterion  = nn.BCELoss()
optimizer  = optim.Adam(loan_model.parameters(), lr=0.001)

print(f"\nModel: {sum(p.numel() for p in loan_model.parameters())} parameters")
print(f"\nTraining with DataLoader (batch_size=32)...")
print(f"{'Epoch':<10} {'Loss':<15} {'Test Acc'}")
print("-" * 40)

for epoch in range(300):
    loan_model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        pred   = loan_model(X_batch)
        loss   = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 60 == 0:
        loan_model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred     = loan_model(X_batch)
                correct += ((pred > 0.5) == y_batch).sum().item()
                total   += y_batch.size(0)

        avg_loss = epoch_loss / len(train_loader)
        acc      = correct / total
        print(f"{epoch:<10} {avg_loss:<15.6f} {acc:.4f}")

# Final accuracy
loan_model.eval()
correct = total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred     = loan_model(X_batch)
        correct += ((pred > 0.5) == y_batch).sum().item()
        total   += y_batch.size(0)

print(f"\n--- Final Results ---")
print(f"Test Accuracy    : {correct/total*100:.2f}%")

# Predict new applicant
new_app   = torch.FloatTensor([[35, 75000, 300000, 720, 10, 1]])
new_norm  = (new_app - torch.FloatTensor(X_loan.mean(axis=0))) / \
             torch.FloatTensor(X_loan.std(axis=0))

loan_model.eval()
with torch.no_grad():
    prob = loan_model(new_norm).item()

print(f"\n--- New Applicant ---")
print(f"Age: 35 | Income: 75K | Credit: 720 | Exp: 10yrs")
print(f"Approval Probability : {prob*100:.2f}%")
print(f"Decision : {'✅ APPROVED' if prob > 0.5 else '❌ REJECTED'}")


print("\n===== WHAT I LEARNED TODAY =====")
print("✅ PyTorch Tensors - creation & operations")
print("✅ Autograd - automatic differentiation")
print("✅ nn.Module - building neural networks")
print("✅ Training Loop - the standard 5 steps")
print("✅ DataLoader - efficient batching")
print("✅ nn.Sequential - clean model definition")
print("✅ Mini Project - Full Loan Approver in PyTorch!")
print("\n🚀 Day 17 Done! Tomorrow - CNNs!")
