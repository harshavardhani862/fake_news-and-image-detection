import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# Label Remapper
# ─────────────────────────────────────────
class RemappedDataset(Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, self.label_map[label]

# ─────────────────────────────────────────
# CBAM Attention Module
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ─────────────────────────────────────────
# ResNet18 + CBAM
# ─────────────────────────────────────────
class ResNetCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep all layers except the final FC
        self.features = nn.Sequential(*list(base.children())[:-2])  # output: (B, 512, 4, 4)
        self.cbam     = CBAM(512)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────
# EfficientNet-B0 + CBAM
# ─────────────────────────────────────────
class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features        # output: (B, 1280, 4, 4)
        self.cbam     = CBAM(1280)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────
# Ensemble: ResNetCBAM + EfficientNetCBAM
# ─────────────────────────────────────────
class EnsembleModel(nn.Module):
    """
    Soft-voting ensemble.
    Averages the softmax probabilities from ResNetCBAM and EfficientNetCBAM.
    A small learnable fusion head re-weights the two streams.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet_cbam     = ResNetCBAM(num_classes)
        self.efficientnet_cbam = EfficientNetCBAM(num_classes)
        # Learnable fusion: concatenate logits (4-dim) → 2-dim
        self.fusion = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        out1 = self.resnet_cbam(x)          # (B, 2)
        out2 = self.efficientnet_cbam(x)    # (B, 2)
        combined = torch.cat([out1, out2], dim=1)  # (B, 4)
        return self.fusion(combined)        # (B, 2)


# ─────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────
# Folder label mapping convention used throughout:
#   ImageFolder sorts subfolders alphabetically →
#   FAKE=0, REAL=1  (F < R)
#   We remap to our standard: REAL=0, FAKE=1
#
# CIFAKE folder layout (Kaggle standard):
#   cifake/
#     train/
#       FAKE/   ← class 0 (alphabetical)
#       REAL/   ← class 1
#     test/
#       FAKE/   ← class 0
#       REAL/   ← class 1
# ─────────────────────────────────────────

ai_gen_dataset = RemappedDataset(
    ImageFolder(r"C:\project\data\images\ai generated", transform=train_transform),
    label_map={0: 1, 1: 0}   # ai_art=0→FAKE=1, real_art=1→REAL=0
)

deepfake_train = RemappedDataset(
    ImageFolder(r"C:\project\data\images\deepfake\Dataset\Train", transform=train_transform),
    label_map={0: 1, 1: 0}   # fake=0→FAKE=1, real=1→REAL=0
)
deepfake_val = RemappedDataset(
    ImageFolder(r"C:\project\data\images\deepfake\Dataset\Validation", transform=val_transform),
    label_map={0: 1, 1: 0}
)
deepfake_test = RemappedDataset(
    ImageFolder(r"C:\project\data\images\deepfake\Dataset\Test", transform=val_transform),
    label_map={0: 1, 1: 0}
)

morphed_dataset = RemappedDataset(
    ImageFolder(r"C:\project\data\images\morphed", transform=train_transform),
    label_map={0: 1, 1: 0}   # fake=0→FAKE=1, real=1→REAL=0
)

# CIFAKE — Train split (used for training)
# Subfolder order: FAKE=0, REAL=1  →  remap to FAKE=1, REAL=0
cifake_train = RemappedDataset(
    ImageFolder(r"C:\project\data\images\cifake\train", transform=train_transform),
    label_map={0: 1, 1: 0}   # FAKE=0→1, REAL=1→0
)

# CIFAKE — Test split (used for final evaluation alongside deepfake_test)
cifake_test = RemappedDataset(
    ImageFolder(r"C:\project\data\images\cifake\test", transform=val_transform),
    label_map={0: 1, 1: 0}
)

# ─────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────
train_loader = DataLoader(
    ConcatDataset([ai_gen_dataset, deepfake_train, morphed_dataset, cifake_train]),
    batch_size=16, shuffle=True, num_workers=2
)
val_loader  = DataLoader(deepfake_val,  batch_size=16, shuffle=False, num_workers=2)

# Combined test loader: deepfake test + CIFAKE test
test_loader = DataLoader(
    ConcatDataset([deepfake_test, cifake_test]),
    batch_size=16, shuffle=False, num_workers=2
)

# ─────────────────────────────────────────
# Model, Optimizer, Loss
# ─────────────────────────────────────────
model     = EnsembleModel(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
best_val_acc = 0.0

for epoch in range(5):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = 100 * correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/image_model_best.pth")

# ─────────────────────────────────────────
# Test Evaluation
# ─────────────────────────────────────────
model.load_state_dict(torch.load("models/image_model_best.pth", map_location=device))
model.eval()

correct = total = 0
fake_correct = fake_total = 0
real_correct = real_total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = torch.argmax(model(imgs), dim=1)

        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        fake_mask = (labels == 1)
        real_mask = (labels == 0)
        fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
        fake_total   += fake_mask.sum().item()
        real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
        real_total   += real_mask.sum().item()

print(f"\nTest Results:")
print(f"  Overall : {100*correct/total:.2f}%")
print(f"  REAL    : {100*real_correct/real_total:.2f}%")
print(f"  FAKE    : {100*fake_correct/fake_total:.2f}%")

torch.save(model.state_dict(), "models/image_model.pth")
print("\n✅ models/image_model.pth saved (EnsembleModel: ResNetCBAM + EfficientNetCBAM)")