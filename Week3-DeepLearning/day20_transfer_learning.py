# ============================================
# DAY 20 - Transfer Learning
# ResNet, VGG, Pretrained Models
# Author: Prateek Kumar Kuntal
# Date: 24 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import numpy as np


# ------------------------------------------
# PART 1 - WHAT IS TRANSFER LEARNING
# ------------------------------------------

print("===== PART 1: What is Transfer Learning =====")

print("""
TRANSFER LEARNING:
    Use a model trained on large dataset
    as starting point for your task

    Pretrained on ImageNet (1.2 million images,
    1000 classes) - model already knows:
        edges, textures, shapes
        eyes, fur, wheels, faces
        general visual concepts

    Fine tune on your smaller dataset
    Much better results with less data and time

WITHOUT TRANSFER LEARNING:
    Train from scratch on small dataset
    Model has no prior knowledge
    Needs millions of images to work well
    Slow and expensive to train

WITH TRANSFER LEARNING:
    Start with rich pretrained features
    Fine tune for your specific task
    Works well with even 1000 images
    Train in minutes instead of days

REAL WORLD USAGE:
    Almost every production image model
    uses transfer learning
    It is the standard approach in industry

TWO STRATEGIES:
    1. Feature Extraction
       Freeze all pretrained layers
       Only train new classification head
       Use when your dataset is very small

    2. Fine Tuning
       Unfreeze some or all layers
       Train entire network but with low lr
       Use when you have more data
""")


# ------------------------------------------
# PART 2 - POPULAR PRETRAINED MODELS
# ------------------------------------------

print("===== PART 2: Popular Pretrained Models =====")

print("""
RESNET (Residual Network):
    Introduced skip connections
    Solves vanishing gradient problem
    Very deep networks (18, 34, 50, 101, 152 layers)
    ResNet50 is most commonly used

VGG:
    Very deep network with small 3x3 filters
    Simple and uniform architecture
    VGG16, VGG19
    Large model size

EFFICIENTNET:
    Scales width, depth, resolution together
    Best accuracy per parameter
    Industry favorite for production

MOBILENET:
    Lightweight model for mobile devices
    Fast inference, small size
    Good for edge deployment

VISION TRANSFORMER (ViT):
    Transformer applied to images
    State of the art on most benchmarks
    We will cover this in Week 4
""")

# Load and inspect pretrained models
print("Loading pretrained ResNet18...")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print("\nResNet18 Architecture:")
print(resnet18)

total_params    = sum(p.numel() for p in resnet18.parameters())
print(f"\nTotal Parameters : {total_params:,}")
print(f"Last layer       : {resnet18.fc}")
print(f"Output classes   : 1000 (ImageNet)")


# ------------------------------------------
# PART 3 - FEATURE EXTRACTION
# ------------------------------------------

print("\n===== PART 3: Feature Extraction =====")

print("""
FEATURE EXTRACTION:
    Freeze all pretrained weights
    Replace last layer with new classifier
    Only train the new classifier

    Pretrained layers act as fixed
    feature extractor

WHEN TO USE:
    Very small dataset (less than 1000 images)
    Your data is similar to ImageNet
    Limited compute resources
    Quick prototyping
""")

def create_feature_extractor(model_name="resnet18",
                              num_classes=10):
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc   = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc   = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(
            in_features, num_classes)

    # Freeze all layers except last
    for name, param in model.named_parameters():
        if "fc" not in name and "classifier.6" not in name:
            param.requires_grad = False

    return model

model_fe = create_feature_extractor("resnet18", num_classes=10)

# Count trainable parameters
total_params     = sum(p.numel() for p in model_fe.parameters())
trainable_params = sum(p.numel() for p in model_fe.parameters()
                       if p.requires_grad)
frozen_params    = total_params - trainable_params

print(f"ResNet18 for Feature Extraction:")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Frozen Parameters    : {frozen_params:,}")
print(f"Training only {trainable_params/total_params*100:.2f}% of parameters!")
print(f"\nNew classification head:")
print(model_fe.fc)


# ------------------------------------------
# PART 4 - FINE TUNING
# ------------------------------------------

print("\n===== PART 4: Fine Tuning =====")

print("""
FINE TUNING:
    Unfreeze some or all pretrained layers
    Train with very small learning rate
    Pretrained weights are starting point
    Gradually adapt to your dataset

STRATEGY:
    Step 1 - Train only head (few epochs)
    Step 2 - Unfreeze last few layers
    Step 3 - Train with small lr (1e-4 or less)

    This prevents destroying pretrained features
    by large gradient updates early on

DIFFERENTIAL LEARNING RATES:
    Different lr for different layers
    Pretrained layers  - very small lr (1e-5)
    New layers         - larger lr (1e-3)
    Common in practice
""")

def create_fine_tuned_model(num_classes=10):
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT)

    # Replace classifier
    in_features = model.fc.in_features
    model.fc    = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model

model_ft = create_fine_tuned_model(num_classes=10)

# Differential learning rates
pretrained_params = []
new_params        = []

for name, param in model_ft.named_parameters():
    if "fc" in name:
        new_params.append(param)
    else:
        pretrained_params.append(param)

optimizer_ft = optim.Adam([
    {"params": pretrained_params, "lr": 1e-5},
    {"params": new_params,        "lr": 1e-3},
])

print(f"Fine Tuned ResNet18:")
print(f"Pretrained layers lr : 1e-5 (very small)")
print(f"New head lr          : 1e-3 (larger)")
print(f"\nNew classification head:")
print(model_ft.fc)


# ------------------------------------------
# PART 5 - DATA PREPARATION FOR TRANSFER
# ------------------------------------------

print("\n===== PART 5: Data Preparation for Transfer Learning =====")

print("""
IMPORTANT FOR TRANSFER LEARNING:
    Models pretrained on ImageNet expect:
    Input size  : 224x224 pixels minimum
    Normalization: ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Always use these exact values when
    using ImageNet pretrained models!
""")

# Transforms for transfer learning
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=8),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Train transforms for Transfer Learning:")
print("  Resize(224) - ResNet expects 224x224")
print("  RandomHorizontalFlip")
print("  RandomCrop(224, padding=8)")
print("  ColorJitter")
print("  Normalize with ImageNet stats")

print("\nTest transforms:")
print("  Resize(224)")
print("  CenterCrop(224)")
print("  Normalize with ImageNet stats")

# Load CIFAR-10 with new transforms
print("\nLoading CIFAR-10 with transfer learning transforms...")

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

val_size   = 5000
train_size = len(train_dataset) - val_size

train_data, val_data = random_split(
    train_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Train samples    : {len(train_data)}")
print(f"Val samples      : {len(val_data)}")
print(f"Test samples     : {len(test_dataset)}")


# ------------------------------------------
# PART 6 - TRAINING TRANSFER LEARNING MODEL
# ------------------------------------------

print("\n===== PART 6: Training Transfer Learning Model =====")

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device           : {device}")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Use feature extraction first (faster)
model     = create_feature_extractor("resnet18", num_classes=10)
model     = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.5)

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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


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

    return total_loss / total_samples, total_correct / total_samples


print("\nPhase 1 - Feature Extraction (frozen pretrained layers):")
print(f"{'Epoch':<8} {'Train Loss':<13} {'Train Acc':<13} {'Val Acc'}")
print("-" * 50)

epochs          = 5
best_val_acc    = 0.0
best_state      = None

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = evaluate(
        model, val_loader, criterion, device)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {
            k: v.clone() for k, v in model.state_dict().items()
        }

    print(f"{epoch+1:<8} {train_loss:<13.4f} "
          f"{train_acc:<13.4f} {val_acc:.4f}")

print(f"\nBest Val Accuracy (Feature Extraction): {best_val_acc*100:.2f}%")


# ------------------------------------------
# PART 7 - FINE TUNING PHASE
# ------------------------------------------

print("\n===== PART 7: Fine Tuning Phase =====")

print("Unfreezing all layers for fine tuning...")

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Differential learning rates
pretrained_params = []
head_params       = []

for name, param in model.named_parameters():
    if "fc" in name:
        head_params.append(param)
    else:
        pretrained_params.append(param)

optimizer_ft = optim.Adam([
    {"params": pretrained_params, "lr": 1e-5},
    {"params": head_params,       "lr": 1e-4},
])
scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=5)

trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params now : {trainable:,}")

print(f"\nPhase 2 - Fine Tuning (all layers trainable):")
print(f"{'Epoch':<8} {'Train Loss':<13} {'Train Acc':<13} {'Val Acc'}")
print("-" * 50)

for epoch in range(5):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer_ft, device)
    val_loss, val_acc     = evaluate(
        model, val_loader, criterion, device)

    scheduler_ft.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {
            k: v.clone() for k, v in model.state_dict().items()
        }

    print(f"{epoch+1:<8} {train_loss:<13.4f} "
          f"{train_acc:<13.4f} {val_acc:.4f}")

print(f"\nBest Val Accuracy (Fine Tuned): {best_val_acc*100:.2f}%")


# ------------------------------------------
# PART 8 - FINAL EVALUATION
# ------------------------------------------

print("\n===== PART 8: Final Evaluation =====")

model.load_state_dict(best_state)
test_loss, test_acc = evaluate(
    model, test_loader, criterion, device)

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
        _, preds = torch.max(outputs, 1)

        for i in range(len(labels)):
            label                = labels[i].item()
            class_correct[label] += (preds[i] == labels[i]).item()
            class_total[label]   += 1

print(f"\nPer Class Accuracy:")
print(f"{'Class':<15} {'Accuracy'}")
print("-" * 30)
for i in range(10):
    acc = class_correct[i] / class_total[i]
    print(f"{classes[i]:<15} {acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "transfer_learning_model.pth")
print(f"\nModel saved to transfer_learning_model.pth")


# ------------------------------------------
# MINI PROJECT - Compare From Scratch vs Transfer
# ------------------------------------------

print("\n===== MINI PROJECT: Scratch vs Transfer Learning =====")

print("""
Comparison Summary:
    Training from scratch on CIFAR-10 (Day 19):
        10 epochs, custom CNN
        Typical accuracy: 70-78%

    Transfer Learning with ResNet18 (Today):
        Feature extraction + fine tuning
        Typical accuracy: 85-92%

    WHY SUCH A BIG DIFFERENCE?
        ResNet18 pretrained on 1.2M ImageNet images
        Already knows rich visual features
        Fine tuning just adapts these features
        Your CNN from scratch had zero prior knowledge

    This is why transfer learning is the
    default approach in industry.
    Nobody trains image models from scratch
    unless they have millions of images.
""")

print("Comparison Table:")
print(f"{'Method':<30} {'Accuracy':<15} {'Train Time'}")
print("-" * 60)
print(f"{'CNN from scratch (Day 19)':<30} {'~70-78%':<15} {'Longer'}")
print(f"{'Transfer Learning (Today)':<30} {'~85-92%':<15} {'Shorter'}")
print(f"\nTransfer Learning wins in both accuracy and speed!")

# Model inspection
print(f"\nResNet18 Layer Groups:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:<15} : {params:>10,} parameters")


print("\n===== WHAT I LEARNED TODAY =====")
print("Transfer Learning - use pretrained model knowledge")
print("Feature Extraction - freeze and train only head")
print("Fine Tuning - unfreeze and train all layers")
print("Differential Learning Rates - different lr per layer")
print("ImageNet Normalization - correct preprocessing")
print("Two phase training - extract then fine tune")
print("Save best model checkpoint")
print("\nDay 20 Done! Tomorrow - Transfer Learning Project!")