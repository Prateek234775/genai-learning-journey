# ============================================
# DAY 18 - Convolutional Neural Networks
# How Computers See Images
# Author: Prateek Kumar Kuntal
# Date: 22 May 2025
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


# ------------------------------------------
# PART 1 - WHAT IS A CNN
# ------------------------------------------

print("===== PART 1: What is a CNN =====")

print("""
CNN (Convolutional Neural Network):
    Designed specifically for image data
    Learns spatial features automatically
    Used in image classification, detection,
    segmentation, face recognition

WHY NOT REGULAR NEURAL NETWORK FOR IMAGES?
    Image 224x224x3 = 150,528 pixels
    Regular NN would need millions of weights
    for just the first layer
    Too many parameters, too slow, overfits

HOW CNN SOLVES THIS:
    Uses small filters (3x3, 5x5)
    Slides filter across image
    Shares weights across positions
    Learns local patterns like edges, shapes

KEY LAYERS:
    Convolutional Layer  - detect features
    Pooling Layer        - reduce size
    Flatten Layer        - convert to 1D
    Fully Connected Layer- final prediction
""")


# ------------------------------------------
# PART 2 - CONVOLUTION OPERATION
# ------------------------------------------

print("===== PART 2: Convolution Operation =====")

print("""
CONVOLUTION:
    Slide a small filter over the image
    Multiply filter values with image values
    Sum them up to get one output number
    Repeat for every position

FILTER / KERNEL:
    Small matrix (3x3, 5x5)
    Contains learnable weights
    Different filters detect different features
    Edge detector, blur, sharpen etc.

STRIDE:
    How many pixels to move filter each step
    Stride 1 = move one pixel at a time
    Stride 2 = move two pixels, smaller output

PADDING:
    Add zeros around image border
    Keeps output same size as input
    same padding  = output size = input size
    valid padding = output size smaller
""")

# Manual convolution to understand it
def manual_conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode="constant")

    img_h, img_w   = image.shape
    ker_h, ker_w   = kernel.shape
    out_h = (img_h - ker_h) // stride + 1
    out_w = (img_w - ker_w) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(0, out_h):
        for j in range(0, out_w):
            region         = image[i*stride:i*stride+ker_h,
                                   j*stride:j*stride+ker_w]
            output[i, j]   = np.sum(region * kernel)

    return output

# Sample 5x5 image
image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 0],
    [2, 1, 0, 1, 1],
    [1, 2, 1, 0, 2]
], dtype=float)

# Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=float)

# Blur kernel
blur_kernel = np.ones((3, 3)) / 9

output_edge = manual_conv2d(image, edge_kernel)
output_blur = manual_conv2d(image, blur_kernel)

print("Original Image:")
print(image)
print("\nEdge Detection Kernel:")
print(edge_kernel)
print("\nAfter Edge Detection:")
print(output_edge)
print("\nAfter Blur:")
print(output_blur.round(3))
print(f"\nOriginal shape   : {image.shape}")
print(f"After conv shape : {output_edge.shape}")


# ------------------------------------------
# PART 3 - POOLING
# ------------------------------------------

print("\n===== PART 3: Pooling =====")

print("""
POOLING:
    Reduces spatial dimensions of feature map
    Makes model smaller and faster
    Adds translation invariance

MAX POOLING (most common):
    Take maximum value in each region
    Keeps strongest feature

AVERAGE POOLING:
    Take average value in each region
    Smoother but less sharp features

Why reduce size?
    Reduce computation
    Reduce memory
    Prevent overfitting
    Make features more robust
""")

def max_pool2d(feature_map, pool_size=2, stride=2):
    h, w     = feature_map.shape
    out_h    = (h - pool_size) // stride + 1
    out_w    = (w - pool_size) // stride + 1
    output   = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region       = feature_map[
                i*stride:i*stride+pool_size,
                j*stride:j*stride+pool_size
            ]
            output[i, j] = np.max(region)

    return output

feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 1, 4, 2],
    [7, 2, 3, 1]
], dtype=float)

pooled = max_pool2d(feature_map, pool_size=2, stride=2)

print("Feature Map (4x4):")
print(feature_map)
print("\nAfter Max Pooling (2x2):")
print(pooled)
print(f"\nBefore pooling : {feature_map.shape}")
print(f"After pooling  : {pooled.shape}")
print("Size reduced by 4x!")


# ------------------------------------------
# PART 4 - BUILDING CNN IN PYTORCH
# ------------------------------------------

print("\n===== PART 4: Building CNN in PyTorch =====")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.fc_layers(x)
        return x

model = SimpleCNN(num_classes=10)
print("CNN Architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal Parameters : {total_params:,}")

# Test with dummy input
dummy = torch.randn(4, 1, 28, 28)
output = model(dummy)
print(f"\nInput shape      : {dummy.shape}")
print(f"Output shape     : {output.shape}")


# ------------------------------------------
# PART 5 - UNDERSTANDING FEATURE MAPS
# ------------------------------------------

print("\n===== PART 5: Understanding Feature Maps =====")

print("""
FEATURE MAPS:
    Output of convolutional layer
    Each filter produces one feature map
    32 filters = 32 feature maps

CHANNELS:
    Input image: 3 channels (R, G, B)
    Grayscale  : 1 channel
    After conv : as many channels as filters

RECEPTIVE FIELD:
    How much of original image each neuron sees
    Deeper layers see larger regions
    Last layer sees full image context
""")

# Show how shapes change through CNN
print("Shape changes through CNN layers:")
print(f"{'Layer':<30} {'Input':<20} {'Output'}")
print("-" * 70)

x = torch.randn(1, 1, 28, 28)
print(f"{'Input':<30} {'-':<20} {str(tuple(x.shape))}")

conv1 = nn.Conv2d(1, 32, 3, padding=1)
x1    = conv1(x)
print(f"{'Conv2d(1,32,3,pad=1)':<30} {str(tuple(x.shape)):<20} {str(tuple(x1.shape))}")

pool1 = nn.MaxPool2d(2, 2)
x2    = pool1(x1)
print(f"{'MaxPool2d(2,2)':<30} {str(tuple(x1.shape)):<20} {str(tuple(x2.shape))}")

conv2 = nn.Conv2d(32, 64, 3, padding=1)
x3    = conv2(x2)
print(f"{'Conv2d(32,64,3,pad=1)':<30} {str(tuple(x2.shape)):<20} {str(tuple(x3.shape))}")

pool2 = nn.MaxPool2d(2, 2)
x4    = pool2(x3)
print(f"{'MaxPool2d(2,2)':<30} {str(tuple(x3.shape)):<20} {str(tuple(x4.shape))}")

x5    = x4.view(x4.size(0), -1)
print(f"{'Flatten':<30} {str(tuple(x4.shape)):<20} {str(tuple(x5.shape))}")


# ------------------------------------------
# PART 6 - BATCH NORMALIZATION & DROPOUT
# ------------------------------------------

print("\n===== PART 6: Batch Normalization and Dropout =====")

print("""
BATCH NORMALIZATION:
    Normalizes activations in each layer
    Makes training faster and more stable
    Reduces need for careful weight init
    Acts as mild regularizer

DROPOUT:
    Randomly turns off neurons during training
    Prevents overfitting
    Common values: 0.2, 0.3, 0.5
    Only active during training, off at test

WHY BOTH?
    BatchNorm  - stable training
    Dropout    - prevent overfitting
    Together   - faster convergence, better accuracy
""")

# Demonstrate BatchNorm
bn     = nn.BatchNorm2d(32)
x_demo = torch.randn(4, 32, 14, 14)
x_bn   = bn(x_demo)

print(f"Input mean  : {x_demo.mean().item():.4f}")
print(f"Input std   : {x_demo.std().item():.4f}")
print(f"After BN mean: {x_bn.mean().item():.4f}")
print(f"After BN std : {x_bn.std().item():.4f}")

# Demonstrate Dropout
drop   = nn.Dropout(p=0.5)
x_drop = torch.ones(1, 10)

drop.train()
x_after_drop = drop(x_drop)
print(f"\nBefore Dropout : {x_drop}")
print(f"After Dropout  : {x_after_drop}")
print(f"Zeros (dropped): {(x_after_drop == 0).sum().item()} neurons")


# ------------------------------------------
# MINI PROJECT - Train CNN on MNIST
# ------------------------------------------

print("\n===== MINI PROJECT: Train CNN on MNIST =====")

print("""
MNIST Dataset:
    70,000 handwritten digit images
    28x28 grayscale images
    10 classes (digits 0-9)
    Classic benchmark for image classification
""")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Downloading MNIST dataset...")

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64)

print(f"Train samples    : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Train batches    : {len(train_loader)}")
print(f"Image shape      : {train_dataset[0][0].shape}")

# Build model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_model = MNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

total_params = sum(p.numel() for p in mnist_model.parameters())
print(f"\nModel Parameters : {total_params:,}")
print(f"Device           : {device}")

print(f"\nTraining MNIST Classifier...")
print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Test Acc'}")
print("-" * 55)

epochs = 5

for epoch in range(epochs):
    # Training
    mnist_model.train()
    train_loss    = 0
    train_correct = 0
    train_total   = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = mnist_model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Testing
    mnist_model.eval()
    test_correct = 0
    test_total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images    = images.to(device)
            labels    = labels.to(device)
            outputs   = mnist_model(images)
            _, predicted = torch.max(outputs, 1)
            test_total   += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    avg_loss   = train_loss / len(train_loader)
    train_acc  = train_correct / train_total
    test_acc   = test_correct / test_total

    scheduler.step()

    print(f"{epoch+1:<8} {avg_loss:<15.4f} {train_acc:<15.4f} {test_acc:.4f}")

print(f"\nFinal Test Accuracy : {test_acc*100:.2f}%")
print(f"(A good CNN should reach 98-99% on MNIST)")

# Show some predictions
mnist_model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs    = mnist_model(images)
    _, predicted = torch.max(outputs, 1)

print(f"\nSample Predictions (first 10):")
print(f"{'True':<10} {'Predicted':<12} {'Correct'}")
print("-" * 35)
for i in range(10):
    true_label = labels[i].item()
    pred_label = predicted[i].item()
    correct    = "Correct" if true_label == pred_label else "Wrong"
    print(f"{true_label:<10} {pred_label:<12} {correct}")


print("\n===== WHAT I LEARNED TODAY =====")
print("Convolution Operation - sliding filter over image")
print("Pooling - reducing spatial dimensions")
print("CNN Architecture - conv, pool, flatten, fc")
print("BatchNorm - stable training")
print("Dropout - prevent overfitting")
print("Trained CNN on MNIST - real image dataset")
print("\nDay 18 Done! Tomorrow - Transfer Learning!")