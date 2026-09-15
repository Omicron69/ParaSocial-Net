import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return ((1 - pt) ** self.gamma * bce).mean()


class Posts(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts = texts
        self.labels = labels
        self.tok = tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, max_length=192,
                     padding="max_length", return_tensors="pt")
        return (e.input_ids.squeeze(0), e.attention_mask.squeeze(0),
                torch.tensor(self.labels[i], dtype=torch.float))


class HarmClassifier(nn.Module):
    def __init__(self, encoder, n_labels=3, extra_dim=0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size + extra_dim, n_labels)

    def forward(self, ids, mask, extra=None):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=1)
        return self.head(pooled)


def score(y_true, y_prob, thresh=0.5):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    pred = (y_prob >= thresh).astype(int)
    out = {}
    for i, name in enumerate(["parasocial", "bullying", "financial"]):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], pred[:, i], average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float("nan")
        out[name] = {"precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f, 3), "auc": round(auc, 3),
                     "support": int(y_true[:, i].sum())}
    return out
