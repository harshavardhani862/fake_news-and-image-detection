"""
calibrate_threshold.py (FINAL STABLE VERSION)
"""
import os
import re
import time
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {device}")
print(f"[INFO] Estimated time : 2–4 minutes\n")

start_time = time.time()

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class BERT_CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.conv    = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.gru     = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask).last_hidden_state
        x = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)


# ─────────────────────────────────────────
# SAMPLES
# ─────────────────────────────────────────
FAKE_SAMPLES = [
    "SHOCKING: Scientists confirm drinking lemon juice cures cancer in 7 days.",
    "NASA whistleblower reveals Moon landing was fake.",
    "Bill Gates spreading COVID through 5G towers.",
    "Government adding mind-control chemicals in water.",
    "Aliens living among humans since 1947.",
    "President replaced by body double.",
    "COVID vaccine reduces fertility.",
    "Humans use only 10 percent brain.",
    "Microwave food causes DNA damage.",
    "5G tracks citizens in real time.",
    "WHO knew pandemic earlier.",
    "Chemtrails reduce population.",
]

REAL_SAMPLES = [
    "Federal Reserve raised interest rates by 25 basis points.",
    "NASA James Webb captured deep space image.",
    "UN Assembly met to discuss climate policy.",
    "Apple reported strong quarterly revenue.",
    "Study links processed food to heart disease.",
    "EU passed AI regulations.",
    "CERN ran collider experiment.",
    "WHO confirmed new flu strain.",
    "Oil prices increased after OPEC decision.",
    "Supreme Court ruled on EPA authority.",
    "Global climate agreement signed.",
    "Study links sedentary lifestyle to diabetes.",
]


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print("Loading model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BERT_CNN_GRU()

try:
    state_dict = torch.load("models/text_model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully\n")
except Exception as e:
    print("⚠️ Model loading failed:", e)
    print("⚠️ Using untrained model (results may be inaccurate)\n")

model.to(device).eval()


# ─────────────────────────────────────────
# GET PROBABILITIES
# Model trained with: REAL=0, FAKE=1
# So softmax output: [prob_REAL, prob_FAKE]
# → prob[1] is the FAKE probability (correct)
# ─────────────────────────────────────────
def get_fake_prob(texts):
    probs = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
        with torch.no_grad():
            out  = model(enc["input_ids"].to(device),
                         enc["attention_mask"].to(device))
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        probs.append(float(prob[1]))   # index 1 = FAKE (matches train_text.py)
    return np.array(probs)


print("Running inference...")
fake_probs_on_fake = get_fake_prob(FAKE_SAMPLES)
fake_probs_on_real = get_fake_prob(REAL_SAMPLES)

# ─────────────────────────────────────────
# SANITY CHECK — catch inverted probabilities
# ─────────────────────────────────────────
avg_on_fake = fake_probs_on_fake.mean()
avg_on_real = fake_probs_on_real.mean()
print(f"\n[Sanity] Avg FAKE-prob on FAKE samples: {avg_on_fake:.3f}  (should be > 0.5)")
print(f"[Sanity] Avg FAKE-prob on REAL samples: {avg_on_real:.3f}  (should be < 0.5)")

if avg_on_fake < avg_on_real:
    print("\n⚠️  WARNING: Probabilities are INVERTED!")
    print("   Your model's class 0 = FAKE and class 1 = REAL.")
    print("   Fix: change prob[1] → prob[0] in get_fake_prob() and re-run.\n")
else:
    print("✅ Probabilities look correct. Proceeding...\n")


# ─────────────────────────────────────────
# THRESHOLD SEARCH (FIXED + SAFE)
# ─────────────────────────────────────────
print("Finding best threshold...")

MARGIN = 0.05   # stability margin (0.03–0.08)

all_probs  = np.concatenate([fake_probs_on_fake, fake_probs_on_real])
all_labels = np.array([1]*len(FAKE_SAMPLES) + [0]*len(REAL_SAMPLES))

best_threshold = 0.45
best_score     = -1.0

for t in np.arange(0.10, 0.91, 0.01):

    preds = []

    for p in all_probs:
        if p > t + MARGIN:
            preds.append(1)
        elif p < t - MARGIN:
            preds.append(0)
        else:
            preds.append(-1)

    preds = np.array(preds)

    valid_idx = preds != -1
    if valid_idx.sum() == 0:
        continue

    preds_valid  = preds[valid_idx]
    labels_valid = all_labels[valid_idx]

    fake_mask = labels_valid == 1
    real_mask = labels_valid == 0

    fake_recall = (
        (preds_valid[fake_mask] == 1).mean()
        if fake_mask.sum() > 0 else 0
    )

    real_recall = (
        (preds_valid[real_mask] == 0).mean()
        if real_mask.sum() > 0 else 0
    )

    score = 0.6 * fake_recall + 0.4 * real_recall

    if score > best_score:
        best_score     = score
        best_threshold = round(t, 2)


# ─────────────────────────────────────────
# SAVE THRESHOLD
# ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
try:
    with open("models/best_threshold.txt", "w") as f:
        f.write(str(best_threshold))
    # Verify write succeeded
    with open("models/best_threshold.txt", "r") as f:
        assert f.read().strip() == str(best_threshold)
    print(f"\n✅ Best Threshold: {best_threshold} saved to models/best_threshold.txt")
except Exception as e:
    print(f"\n❌ Failed to save threshold: {e}")


# ─────────────────────────────────────────
# PATCH text_model.py
# Uses multiline regex so only the _CALIBRATED_THRESHOLD
# assignment line is updated — nothing else in the file
# ─────────────────────────────────────────
PATCH_PATH = "text_model.py"

try:
    with open(PATCH_PATH, "r") as f:
        content = f.read()

    # Only patches the exact line:  _CALIBRATED_THRESHOLD = _load_threshold()
    # OR a hardcoded fallback line if present — scoped with ^ and MULTILINE
    patched = re.sub(
        r"^(\s*_CALIBRATED_THRESHOLD\s*=\s*)[\d.]+",
        rf"\g<1>{best_threshold}",
        content,
        flags=re.MULTILINE
    )

    if patched == content:
        # File uses _load_threshold() dynamically — no hardcoded value to patch
        # That's fine: the saved .txt file will be loaded automatically at startup
        print("✅ text_model.py uses dynamic loading — threshold will be read from best_threshold.txt at startup")
    else:
        with open(PATCH_PATH, "w") as f:
            f.write(patched)
        print(f"✅ text_model.py patched → _CALIBRATED_THRESHOLD = {best_threshold}")

except FileNotFoundError:
    print(f"⚠️ {PATCH_PATH} not found — skipping patch step")
except Exception as e:
    print("⚠️ Could not patch file:", e)


# ─────────────────────────────────────────
# FINAL RESULT
# ─────────────────────────────────────────
print("\n🎯 Calibration Complete!")
print(f"Stable Threshold = {best_threshold}")
print(f"Margin Used      = {MARGIN}")
print(f"Time Taken       = {time.time()-start_time:.1f}s")