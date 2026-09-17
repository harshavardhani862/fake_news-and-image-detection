
"""
text_model.py
─────────────────────────────────────────────────────────────────────────────
Inference wrapper for the BERT + CNN + GRU fake-news text classifier.
Loaded once at startup; call predict_text() for every request.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_LABELS = {0: "REAL", 1: "FAKE"}


# ─────────────────────────────────────────────────────────────────────────
# Model — must match train_text.py exactly
# ─────────────────────────────────────────────────────────────────────────
class BERT_CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.conv    = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.gru     = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask).last_hidden_state  # (B,T,768)
        x = torch.relu(
            self.conv(x.permute(0, 2, 1))
        ).permute(0, 2, 1)                                          # (B,T,128)
        x, _ = self.gru(x)                                          # (B,T,128)
        x = self.dropout(x[:, -1, :])                               # (B,128)
        return self.fc(x)                                            # (B,2)


# ─────────────────────────────────────────────────────────────────────────
# Load tokeniser and model once at startup
# ─────────────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

_model = BERT_CNN_GRU()
_model.load_state_dict(torch.load("models/text_model.pth", map_location=device))
_model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────
# Text Cleaner — fixes wrong predictions caused by bad input formatting
# ─────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.strip()
    if text.isupper():              # ALL CAPS → normalize
        text = text.capitalize()
    text = re.sub(r'[!?]{2,}', '!', text)        # !!! → !
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'\s+', ' ', text).strip()     # collapse whitespace
    return text


# ─────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────
def predict_text(text: str, threshold: float = 0.45):
    """
    Predict whether `text` is REAL or FAKE.

    Parameters
    ----------
    text      : raw news article / headline string
    threshold : minimum fake_prob to classify as FAKE (default 0.45)
                Lower = catches more FAKE  |  Higher = more conservative

    Returns
    -------
    label      : "REAL", "FAKE", or "UNCERTAIN" (if text too short)
    confidence : percentage confidence in the predicted label
    probs      : numpy array [real_prob, fake_prob]
    """
    text = clean_text(text)

    # Too short — model will guess randomly on very short inputs
    if len(text.split()) < 5:
        return "UNCERTAIN", 50.0, [0.5, 0.5]

    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        out   = _model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device)
        )
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    if fake_prob > threshold:
        label, confidence = "FAKE", fake_prob * 100
    else:
        label, confidence = "REAL", real_prob * 100

    return label, confidence, probs