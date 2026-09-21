"""
models.py

Two-tier prediction models sharing a common interface so the Systems/OS lead's
router can call either one identically:

    model.predict(features) -> {"pit": 0/1, "pit_prob": float, "compound": str or None}

Fast model:  scikit-learn LogisticRegression over a single-lap snapshot.
Smart model: PyTorch GRU over a sliding window of the last WINDOW_SIZE laps.

Both predict pit_next_3 (binary) first, then compound_next (multiclass,
only meaningful when pit=1) via a second small head/classifier.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

WINDOW_SIZE = 8  # laps of history the GRU looks at
COMPOUND_CLASSES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

NUMERIC_FEATURES = [
    "tyre_life", "gap_ahead", "gap_behind", "gap_ahead_delta",
    "air_temp", "track_temp", "rainfall", "laps_since_field_pit",
    "is_caution",
]
# compound is categorical -> one-hot at feature-build time (see prepare_features)
# track_status -> collapsed into a single is_caution flag (see prepare_features)


def prepare_features(df: pd.DataFrame):
    """Turn the raw dataframe into a clean numeric feature matrix + labels."""
    df = df.copy()
    df["rainfall"] = df["rainfall"].astype(float)

    # track_status is FastF1's per-lap flag code as a string, e.g. "1" (green/
    # clear track), "2" (yellow), "4" (safety car), "5" (red flag), "6"/"7"
    # (VSC). A lap can show multiple codes concatenated (e.g. "24") if track
    # conditions changed mid-lap. Rather than one-hot every rare code (mostly
    # noise/sparse), collapse to a single binary "is anything other than
    # green flag happening" signal — this is one of the strongest real-world
    # pit-stop triggers (teams often pit under caution to minimize time
    # lost), and it was previously being silently dropped entirely.
    df["track_status"] = df["track_status"].astype(str).fillna("1")
    df["is_caution"] = df["track_status"].apply(
        lambda s: 0.0 if set(s) <= {"1"} else 1.0
    )

    df = pd.get_dummies(df, columns=["compound"], prefix="cmp")
    cmp_cols = [c for c in df.columns if c.startswith("cmp_")]
    feature_cols = NUMERIC_FEATURES + cmp_cols

    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values.astype(np.float32)
    y_pit = df["pit_next_3"].values.astype(np.int64)
    y_compound = df["compound_next"].values  # string or NaN, only used where y_pit==1
    return df, X, y_pit, y_compound, feature_cols


class FastModel:
    """Lightweight single-lap-snapshot classifier. Target: sub-millisecond inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf_pit = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf_compound = LogisticRegression(max_iter=1000)
        self.feature_cols = None

    def fit(self, X: np.ndarray, y_pit: np.ndarray, y_compound: np.ndarray, feature_cols):
        self.feature_cols = feature_cols
        Xs = self.scaler.fit_transform(X)
        self.clf_pit.fit(Xs, y_pit)

        mask = y_pit == 1
        if mask.sum() > 0:
            self.clf_compound.fit(Xs[mask], y_compound[mask])
        return self

    def predict(self, x_row: np.ndarray):
        """x_row: shape (n_features,) — a single lap snapshot."""
        xs = self.scaler.transform(x_row.reshape(1, -1))
        pit_prob = self.clf_pit.predict_proba(xs)[0, 1]
        pit = int(pit_prob >= 0.5)
        compound = self.clf_compound.predict(xs)[0] if pit else None
        return {"pit": pit, "pit_prob": float(pit_prob), "compound": compound}

    def predict_batch(self, X: np.ndarray):
        xs = self.scaler.transform(X)
        pit_probs = self.clf_pit.predict_proba(xs)[:, 1]
        pits = (pit_probs >= 0.5).astype(int)
        return pits, pit_probs


class GRUNet(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, n_compound=len(COMPOUND_CLASSES)):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers, batch_first=True)
        self.pit_head = nn.Linear(hidden_size, 1)
        self.compound_head = nn.Linear(hidden_size, n_compound)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        pit_logit = self.pit_head(last_hidden).squeeze(-1)
        compound_logits = self.compound_head(last_hidden)
        return pit_logit, compound_logits


def make_sequences(X: np.ndarray, groups: np.ndarray, window: int = WINDOW_SIZE):
    """
    Build sliding windows PER GROUP (i.e. per driver-race), so a window never
    crosses from one driver/race into another. groups: array of group ids
    aligned to X's rows, already sorted by lap within each group.
    """
    seqs, idx_map = [], []
    start = 0
    n = len(groups)
    while start < n:
        end = start
        while end < n and groups[end] == groups[start]:
            end += 1
        block = X[start:end]
        for i in range(len(block)):
            lo = max(0, i - window + 1)
            window_arr = block[lo:i + 1]
            if len(window_arr) < window:
                pad = np.zeros((window - len(window_arr), X.shape[1]), dtype=np.float32)
                window_arr = np.vstack([pad, window_arr])
            seqs.append(window_arr)
            idx_map.append(start + i)
        start = end
    return np.stack(seqs), np.array(idx_map)


class SmartModel:
    """GRU over a WINDOW_SIZE-lap history. Target: higher accuracy, slower inference."""

    def __init__(self, n_features, hidden_size=64, lr=5e-4, epochs=60, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GRUNet(n_features, hidden_size).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, X_seq: np.ndarray, y_pit: np.ndarray, y_compound_idx: np.ndarray, feature_cols):
        """
        X_seq: (n_samples, WINDOW_SIZE, n_features) already windowed.
        y_compound_idx: int index into COMPOUND_CLASSES, -1 where not applicable.
        """
        self.feature_cols = feature_cols
        n, w, f = X_seq.shape
        flat = X_seq.reshape(-1, f)
        flat = self.scaler.fit_transform(flat)
        X_seq = flat.reshape(n, w, f)

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_pit_t = torch.tensor(y_pit, dtype=torch.float32).to(self.device)
        y_cmp_t = torch.tensor(np.where(y_compound_idx < 0, 0, y_compound_idx),
                                dtype=torch.long).to(self.device)
        cmp_mask = torch.tensor(y_compound_idx >= 0).to(self.device)

        # pit_next_3 is heavily imbalanced (~90% negative laps) — without
        # reweighting, BCE loss is minimized by just predicting "no pit"
        # every time, which gives high accuracy but zero recall/precision
        # (exactly the collapse this was hitting). pos_weight counteracts
        # that the same way class_weight="balanced" does for FastModel.
        n_pos = y_pit.sum()
        n_neg = len(y_pit) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pit_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        cmp_loss_fn = nn.CrossEntropyLoss()

        # Full-batch training with a reweighted loss oscillates around the
        # decision boundary — recall can swing from 0.9 down to 0.6 between
        # epochs even while loss decreases smoothly. Whatever epoch training
        # happens to stop on is not reliably the best one, so track the
        # best-scoring epoch (by training F1) and restore those weights at
        # the end instead of just keeping the final epoch's.
        best_f1 = -1.0
        best_state = None

        self.net.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            pit_logit, cmp_logits = self.net(X_t)
            loss = pit_loss_fn(pit_logit, y_pit_t)
            if cmp_mask.sum() > 0:
                loss = loss + cmp_loss_fn(cmp_logits[cmp_mask], y_cmp_t[cmp_mask])
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = (torch.sigmoid(pit_logit) >= 0.5).float().cpu().numpy()
            metrics = evaluate_pit_predictions(y_pit, preds)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"  [GRU] epoch {epoch}: loss={loss.item():.4f}  "
                      f"train_recall={metrics['recall']:.3f}  train_f1={metrics['f1']:.3f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"  [GRU] restored best epoch (train_f1={best_f1:.3f})")
        return self


    def predict(self, x_window: np.ndarray):
        """x_window: shape (WINDOW_SIZE, n_features) — last N laps for one driver."""
        self.net.eval()
        w, f = x_window.shape
        xs = self.scaler.transform(x_window).reshape(1, w, f)
        xt = torch.tensor(xs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pit_logit, cmp_logits = self.net(xt)
            pit_prob = torch.sigmoid(pit_logit).item()
            pit = int(pit_prob >= 0.5)
            compound = COMPOUND_CLASSES[cmp_logits.argmax(dim=-1).item()] if pit else None
        return {"pit": pit, "pit_prob": pit_prob, "compound": compound}


def benchmark_speed(model, sample_input, n_runs=200):
    """Common speed benchmark for both model types — feeds the OS lead's routing table."""
    # warmup
    for _ in range(5):
        model.predict(sample_input)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample_input)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per prediction


def evaluate_pit_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }