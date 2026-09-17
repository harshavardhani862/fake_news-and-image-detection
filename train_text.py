"""
train_text.py
─────────────────────────────────────────────────────────────────────────────
Fake-News Text Classifier  –  BERT + CNN + GRU
Datasets used
  1. Fake.csv / real.csv           (local, original)
  2. FakeNewsNet  (GossipCop + PolitiFact via pygooglenews / direct CSV)
  3. Wikipedia-based REAL facts    (via wikipedia-api)

Install extras before running:
    pip install wikipedia-api fakenewsnet pandas scikit-learn transformers torch
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from sklearn.utils.class_weight import compute_class_weight

# ── optional imports (graceful fallback if not installed) ─────────────────
try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False
    print("[WARN] wikipedia-api not installed. Wikipedia facts will be skipped.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────
# 1.  Wikipedia Real-Fact Scraper
# ─────────────────────────────────────────────────────────────────────────
WIKI_TOPICS = [
    # Science & Nature
    "Photosynthesis", "Theory of relativity", "Evolution", "DNA",
    "Climate change", "Quantum mechanics", "Big Bang", "Black hole",
    "Vaccine", "Antibiotic", "Solar system", "Plate tectonics",

    # History & Politics
    "World War II", "United Nations", "French Revolution",
    "American Civil War", "Cold War", "Moon landing",
    "Nelson Mandela", "Mahatma Gandhi", "Abraham Lincoln",

    # Geography & Society
    "Amazon rainforest", "Sahara Desert", "Himalaya",
    "European Union", "World Health Organization",
    "International Monetary Fund", "Nobel Prize",

    # Technology
    "Internet", "Artificial intelligence", "Blockchain",
    "CRISPR", "Renewable energy", "Electric vehicle",
]


def fetch_wikipedia_facts(topics: list[str], max_chars: int = 1500) -> pd.DataFrame:
    """
    Download the opening summary of each Wikipedia topic.
    Returns a DataFrame with columns [text, label=0 (REAL)].
    """
    if not WIKI_AVAILABLE:
        return pd.DataFrame(columns=["text", "label"])

    wiki = wikipedaapi.Wikipedia(
        language="en",
        user_agent="FakeNewsDetector/1.0 (research project)"
    )

    records = []
    for topic in topics:
        try:
            page = wiki.page(topic)
            if not page.exists():
                print(f"  [WIKI] Page not found: {topic}")
                continue

            summary = page.summary[:max_chars].strip()
            if len(summary) < 100:
                continue

            # Split into ~3-sentence chunks so each sample is shorter
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            chunk, chunks = [], []
            for sent in sentences:
                chunk.append(sent)
                if len(chunk) >= 3:
                    chunks.append(" ".join(chunk))
                    chunk = []
            if chunk:
                chunks.append(" ".join(chunk))

            for c in chunks:
                records.append({"text": c, "label": 0})   # REAL = 0

            time.sleep(0.3)   # be polite to the API
        except Exception as exc:
            print(f"  [WIKI] Error on '{topic}': {exc}")

    df = pd.DataFrame(records)
    print(f"[Wikipedia] Collected {len(df)} real-fact samples from {len(topics)} topics.")
    return df


# ─────────────────────────────────────────────────────────────────────────
# 2.  FakeNewsNet Loader
#     Expects the standard CSV layout from the FakeNewsNet repo:
#       gossipcop_fake.csv / gossipcop_real.csv
#       politifact_fake.csv / politifact_real.csv
#     Each CSV must have a 'title' and/or 'text' column.
# ─────────────────────────────────────────────────────────────────────────
FAKENEWSNET_FILES = {
    # path → label (1=FAKE, 0=REAL)
    "/kaggle/input/fakenewsnet/gossipcop_fake.csv":    1,
    "/kaggle/input/fakenewsnet/gossipcop_real.csv":    0,
    "/kaggle/input/fakenewsnet/PolitiFact_fake_news_content.csv":   1,
    "/kaggle/input/fakenewsnet/PolitiFact_real_news_content.csv":   0,
}


def load_fakenewsnet(file_map: dict) -> pd.DataFrame:
    """
    Load FakeNewsNet CSVs.  Combines 'title' and 'text' columns when both
    exist so we give the model richer input.
    """
    frames = []
    for path, label in file_map.items():
        if not os.path.exists(path):
            print(f"  [FakeNewsNet] File not found, skipping: {path}")
            continue

        df = pd.read_csv(path)

        # Prefer full article text; fall back to title
        if "text" in df.columns and "title" in df.columns:
            df["combined"] = (
                df["title"].fillna("") + " " + df["text"].fillna("")
            ).str.strip()
        elif "text" in df.columns:
            df["combined"] = df["text"].fillna("")
        elif "title" in df.columns:
            df["combined"] = df["title"].fillna("")
        else:
            print(f"  [FakeNewsNet] No usable text column in {path}, skipping.")
            continue

        df = df[df["combined"].str.len() > 30].copy()
        df["label"] = label
        frames.append(df[["combined", "label"]].rename(columns={"combined": "text"}))
        print(f"  [FakeNewsNet] Loaded {len(df)} rows from {os.path.basename(path)}")

    if not frames:
        return pd.DataFrame(columns=["text", "label"])

    result = pd.concat(frames, ignore_index=True)
    print(f"[FakeNewsNet] Total: {len(result)} rows")
    return result


# ─────────────────────────────────────────────────────────────────────────
# 3.  Dataset class
# ─────────────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────
# 4.  Model: BERT + CNN + GRU
# ─────────────────────────────────────────────────────────────────────────
class BERT_CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.conv    = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.gru     = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(128, 2)      # 64 * 2 directions
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask).last_hidden_state   # (B, T, 768)
        x = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)  # (B, T, 128)
        x, _ = self.gru(x)                                            # (B, T, 128)
        x = self.dropout(x[:, -1, :])                                 # last hidden
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────
# 5.  Main training loop
# ─────────────────────────────────────────────────────────────────────────
def main():
    # ── 5a. Load local CSVs ───────────────────────────────────────────────
    local_frames = []
    for path, label in [
        ("/content/Fake.csv", 1),
        ("/content/real.csv", 0),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["label"] = label
            local_frames.append(df[["text", "label"]])
            print(f"[Local] Loaded {len(df)} rows from {os.path.basename(path)}")
        else:
            print(f"[Local] File not found: {path}")

    # ── 5b. Load FakeNewsNet ──────────────────────────────────────────────
    fnn_df = load_fakenewsnet(FAKENEWSNET_FILES)

    # ── 5c. Fetch Wikipedia real facts ────────────────────────────────────
    wiki_df = fetch_wikipedia_facts(WIKI_TOPICS, max_chars=1500)

    # ── 5d. Combine all sources ───────────────────────────────────────────
    all_frames = [f for f in local_frames + [fnn_df, wiki_df] if not f.empty]
    if not all_frames:
        raise RuntimeError("No training data found. Check your file paths.")

    df = (
        pd.concat(all_frames, ignore_index=True)
        .dropna(subset=["text"])
        .query("text.str.strip().str.len() > 20")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    df["text"] = df["text"].astype(str)

    print(f"\n[DATA] Total samples : {len(df)}")
    print(f"[DATA] Label distribution:\n{df['label'].value_counts()}\n")

    # ── 5e. Tokeniser & DataLoaders ───────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset   = NewsDataset(df["text"], df["label"], tokenizer)

    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

    # ── 5f. Class weights to handle imbalance ─────────────────────────────
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=df["label"].values),
        dtype=torch.float,
    ).to(device)

    # ── 5g. Model, optimiser, loss ────────────────────────────────────────
    model     = BERT_CNN_GRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0

    # ── 5h. Training ──────────────────────────────────────────────────────
    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        fake_correct = fake_total = 0
        real_correct = real_total = 0

        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                preds = torch.argmax(model(ids, mask), dim=1)

                correct += (preds == labels).sum().item()
                total   += labels.size(0)

                fake_mask = (labels == 1)
                real_mask = (labels == 0)
                fake_correct += (preds[fake_mask] == labels[fake_mask]).sum().item()
                fake_total   += fake_mask.sum().item()
                real_correct += (preds[real_mask] == labels[real_mask]).sum().item()
                real_total   += real_mask.sum().item()

        val_acc  = 100 * correct / total
        fake_acc = 100 * fake_correct / fake_total if fake_total else 0
        real_acc = 100 * real_correct / real_total if real_total else 0

        scheduler.step()

        print(
            f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}% | REAL: {real_acc:.2f}% | FAKE: {fake_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/text_model_best.pth")
            print("  → Best model saved.")

    # Copy best → final
    import shutil
    shutil.copy("models/text_model_best.pth", "models/text_model.pth")
    print("\n✅ Text model saved to models/text_model.pth")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()