import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_LABELS = {0: "REAL", 1: "FAKE"}


# ─────────────────────────────────────────
# CBAM BLOCK
# ─────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) +
            self.fc(self.max_pool(x))
        ).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
class ResNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.cbam = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class EfficientNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────
# FIX 1: EnsembleModel attribute names now
# match train_image.py EXACTLY
#
# train_image.py saved weights under:
#   self.resnet_cbam        ← was self.resnet    (WRONG)
#   self.efficientnet_cbam  ← was self.efficient (WRONG)
#   self.fusion             ← was self.fc        (WRONG)
#
# Mismatch = weights never loaded = random predictions always
# ─────────────────────────────────────────
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet_cbam       = ResNetCBAM()
        self.efficientnet_cbam = EfficientNetCBAM()
        self.fusion            = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        out1 = self.resnet_cbam(x)
        out2 = self.efficientnet_cbam(x)
        return self.fusion(torch.cat([out1, out2], dim=1))


# ─────────────────────────────────────────
# SAFE MODEL LOADING
# ─────────────────────────────────────────
def load_model():
    model = EnsembleModel()

    try:
        state_dict = torch.load("models/image_model.pth", map_location=device)

        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=True)  # strict=True catches mismatches
        else:
            model = state_dict

        print("✅ Image model loaded successfully")

    except RuntimeError as e:
        print("❌ Weight mismatch error:", e)
        print("   Check that EnsembleModel attribute names match train_image.py")
    except Exception as e:
        print("⚠️ Model loading failed:", e)
        print("⚠️ Using untrained model (predictions will be random)")

    model.to(device)
    model.eval()
    return model


model = load_model()


# ─────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────
# FIX 2: Load calibrated threshold from file
# Falls back to 0.40 if not found
# ─────────────────────────────────────────
def _load_image_threshold() -> float:
    try:
        with open("models/best_threshold.txt", "r") as f:
            t = float(f.read().strip())
            print(f"[image_model] Loaded calibrated threshold: {t}")
            return t
    except Exception:
        print("[image_model] No calibrated threshold found — using default 0.40")
        return 0.40

_IMAGE_THRESHOLD = _load_image_threshold()


# ─────────────────────────────────────────
# PREDICTION FUNCTION
#
# train_image.py label mapping (all datasets):
#   REAL = 0,  FAKE = 1
# So softmax output → [prob_REAL, prob_FAKE]
#   probs[0] = REAL probability
#   probs[1] = FAKE probability ← compared against threshold
# ─────────────────────────────────────────
def predict_image(img, threshold=None):

    # Use calibrated threshold if main.py slider didn't pass one
    if threshold is None:
        threshold = _IMAGE_THRESHOLD

    img = img.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs  = torch.softmax(output, dim=1).cpu().numpy()[0]

    real_prob = float(probs[0])   # REAL (label 0 in training)
    fake_prob = float(probs[1])   # FAKE (label 1 in training)

    if fake_prob > threshold:
        return "FAKE", fake_prob * 100, [real_prob, fake_prob]
    else:
        return "REAL", real_prob * 100, [real_prob, fake_prob]