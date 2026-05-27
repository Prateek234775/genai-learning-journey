# ============================================
# DAY 19 - Image Classifier on CIFAR-10
# Data Augmentation + Better CNN Architecture
# Author: Prateek Kumar Kuntal
# Date: 23 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np


# ------------------------------------------
# PART 1 - CIFAR-10 DATASET
# ------------------------------------------

print("===== PART 1: CIFAR-10 Dataset =====")

print("""
CIFAR-10:
    60,000 color images (32x32 pixels)
    3 channels (RGB)
    10 classes:
        0 - airplane
        1 - automobile
        2 - bird
        3 - cat
        4 - deer
        5 - dog
        6 - frog
        7 - horse
        8 - ship
        9 - truck

    Much harder than MNIST because:
        Color images (3 channels vs 1)
        Real world objects
        More visual variation
        Small image size makes it harder

    Good benchmark:
        Simple CNN     - around 70-75%
        ResNet         - around 93%
        State of art   - above 99%
""")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ------------------------------------------
# PART 2 - DATA AUGMENTATION
# ------------------------------------------

print("===== PART 2: Data Augmentation =====")

print("""
DATA AUGMENTATION:
    Artificially increase training data
    by applying random transformations
    to existing images

    Makes model more robust and reduces overfitting

COMMON AUGMENTATIONS:
    Random Horizontal Flip  - mirror image left/right
    Random Crop             - crop random region
    Color Jitter            - change brightness/contrast
    Random Rotation         - rotate image slightly
    Normalization           - scale pixel values

WHY IT WORKS:
    A cat flipped horizontally is still a cat
    A rotated airplane is still an airplane
    Model learns to be invariant to these changes

WITHOUT AUGMENTATION:
    Model memorizes exact training images
    Poor performance on new images

WITH AUGMENTATION:
    Model sees more variation
    Generalizes better to new data
""")

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# Test transforms - no augmentation
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

print("Train transforms:")
print("  RandomHorizontalFlip - 50% chance")
print("  RandomCrop(32, pad=4) - crop with padding")
print("  ColorJitter - brightness, contrast, saturation")
print("  RandomRotation(15 degrees)")
print("  ToTensor + Normalize")
print("\nTest transforms:")
print("  ToTensor + Normalize only (no augmentation)")


# ------------------------------------------
# PART 3 - LOAD CIFAR-10
# ------------------------------------------

print("\n===== PART 3: Load CIFAR-10 =====")

print("Downloading CIFAR-10 dataset...")

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# Split train into train and validation
val_size   = 5000
train_size = len(train_dataset) - val_size
train_data, val_data = random_split(
    train_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_data,
    batch_size=128,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

print(f"Train samples    : {len(train_data)}")
print(f"Val samples      : {len(val_data)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Train batches    : {len(train_loader)}")
print(f"Image shape      : {train_dataset[0][0].shape}")
print(f"Classes          : {classes}")


# ------------------------------------------
# PART 4 - BUILD BETTER CNN ARCHITECTURE
# ------------------------------------------

print("\n===== PART 4: Better CNN Architecture =====")

print("""
IMPROVEMENTS OVER BASIC CNN:
    More conv layers - learn deeper features
    Residual connections - prevent vanishing gradient
    Global Average Pooling - fewer parameters
    BatchNorm everywhere - stable training
    Better activation functions

VGG STYLE ARCHITECTURE:
    Multiple conv layers before pooling
    Gradually increase channels
    64 -> 128 -> 256 -> 512
    Works very well for image classification
""")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CIFAR10Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Classifier, self).__init__()

        # Feature extraction
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CIFAR10Classifier(num_classes=10).to(device)

total_params = sum(p.numel() for p in model.parameters())
print("Model Architecture:")
print(model)
print(f"\nTotal Parameters : {total_params:,}")
print(f"Device           : {device}")

# Test forward pass
dummy = torch.randn(4, 3, 32, 32).to(device)
out   = model(dummy)
print(f"\nInput shape      : {dummy.shape}")
print(f"Output shape     : {out.shape}")


# ------------------------------------------
# PART 5 - TRAINING SETUP
# ------------------------------------------

print("\n===== PART 5: Training Setup =====")

print("""
LEARNING RATE SCHEDULER:
    Reduce learning rate during training
    Start high for fast initial learning
    Reduce when plateau reached
    Helps fine-tune in later epochs

WEIGHT DECAY:
    L2 regularization in optimizer
    Penalizes large weights
    Helps prevent overfitting

GRADIENT CLIPPING:
    Limit maximum gradient size
    Prevents exploding gradients
    Common in RNNs, also helps CNNs
""")

criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer  = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=3,
    factor=0.5,
)

print("Loss function    : CrossEntropyLoss (label smoothing=0.1)")
print("Optimizer        : Adam (lr=0.001, weight_decay=1e-4)")
print("Scheduler        : ReduceLROnPlateau (patience=3)")


# ------------------------------------------
# PART 6 - TRAINING LOOP WITH VALIDATION
# ------------------------------------------

print("\n===== PART 6: Training Loop =====")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            _, predicted   = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


print(f"{'Epoch':<8} {'Train Loss':<13} {'Train Acc':<13} {'Val Loss':<13} {'Val Acc':<13} {'LR'}")
print("-" * 75)

epochs          = 10
best_val_acc    = 0.0
best_model_state = None
history         = []

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = evaluate(
        model, val_loader, criterion, device)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch"     : epoch + 1,
        "train_loss": train_loss,
        "train_acc" : train_acc,
        "val_loss"  : val_loss,
        "val_acc"   : val_acc,
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc    = val_acc
        best_model_state = {
            k: v.clone() for k, v in model.state_dict().items()
        }

    print(f"{epoch+1:<8} {train_loss:<13.4f} {train_acc:<13.4f} "
          f"{val_loss:<13.4f} {val_acc:<13.4f} {current_lr:.6f}")

print(f"\nBest Val Accuracy : {best_val_acc*100:.2f}%")


# ------------------------------------------
# PART 7 - EVALUATE ON TEST SET
# ------------------------------------------

print("\n===== PART 7: Evaluate on Test Set =====")

# Load best model
model.load_state_dict(best_model_state)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Test Loss        : {test_loss:.4f}")
print(f"Test Accuracy    : {test_acc*100:.2f}%")

# Per class accuracy
model.eval()
class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        labels  = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            label               = labels[i].item()
            class_correct[label]+= (predicted[i] == labels[i]).item()
            class_total[label]  += 1

print(f"\nPer Class Accuracy:")
print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy'}")
print("-" * 45)
for i in range(10):
    acc = class_correct[i] / class_total[i]
    print(f"{classes[i]:<15} {class_correct[i]:<10} "
          f"{class_total[i]:<10} {acc*100:.2f}%")


# ------------------------------------------
# PART 8 - SAVE AND LOAD MODEL
# ------------------------------------------

print("\n===== PART 8: Save and Load Model =====")

print("""
SAVING MODELS IN PYTORCH:
    Two ways to save:

    1. Save full model (easy but less flexible)
       torch.save(model, "model.pth")

    2. Save state dict (recommended)
       torch.save(model.state_dict(), "model.pth")
       Load with model.load_state_dict(...)

    State dict = dictionary of all weights and biases
    Always use state dict method in production
""")

# Save model
torch.save(model.state_dict(), "cifar10_model.pth")
print("Model saved to cifar10_model.pth")

# Load model
loaded_model = CIFAR10Classifier(num_classes=10).to(device)
loaded_model.load_state_dict(
    torch.load("cifar10_model.pth", map_location=device)
)
loaded_model.eval()
print("Model loaded successfully")

# Verify loaded model works
_, loaded_acc = evaluate(loaded_model, test_loader, criterion, device)
print(f"Loaded model accuracy : {loaded_acc*100:.2f}%")
print(f"Same as before        : {abs(loaded_acc - test_acc) < 0.001}")


# ------------------------------------------
# MINI PROJECT - Predict Custom Images
# ------------------------------------------

print("\n===== MINI PROJECT: Class Prediction Analysis =====")

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion analysis
print("Most confused class pairs:")
print(f"{'True Class':<15} {'Predicted As':<15} {'Count'}")
print("-" * 45)

confusion = {}
for true, pred in zip(all_labels, all_preds):
    if true != pred:
        key = (true, pred)
        confusion[key] = confusion.get(key, 0) + 1

top_confused = sorted(
    confusion.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

for (true, pred), count in top_confused:
    print(f"{classes[true]:<15} {classes[pred]:<15} {count}")

# Training summary
print("\nTraining History Summary:")
print(f"{'Epoch':<8} {'Train Acc':<15} {'Val Acc'}")
print("-" * 35)
for h in history:
    print(f"{h['epoch']:<8} {h['train_acc']:<15.4f} {h['val_acc']:.4f}")

print(f"\nFinal Test Accuracy  : {test_acc*100:.2f}%")
print(f"Best Val Accuracy    : {best_val_acc*100:.2f}%")


print("\n===== WHAT I LEARNED TODAY =====")
print("CIFAR-10 - real world image classification")
print("Data Augmentation - increase training data variety")
print("Better CNN Architecture - deeper, stronger")
print("Learning Rate Scheduler - adaptive learning")
print("Validation Loop - monitor overfitting")
print("Per Class Accuracy - detailed evaluation")
print("Save and Load Model - production ready")
print("\nDay 19 Done! Tomorrow - Transfer Learning!")