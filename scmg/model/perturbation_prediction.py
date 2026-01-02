from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# ----------------------------
# Utilities
# ----------------------------
def build_vocab(categories: Iterable) -> Dict:
    """Map each unique category to a contiguous integer id."""
    uniq = sorted({c for c in categories})
    return {c: i for i, c in enumerate(uniq)}


def encode_categories(categories: Iterable, vocab: Dict) -> np.ndarray:
    """Convert a list of categories to integer ids using the vocab."""
    return np.array([vocab[c] for c in categories], dtype=np.int64)


# ----------------------------
# Dataset
# ----------------------------
class CatRealDataset(Dataset):
    """
    Expects:
      cats: list/array of length N (strings/ints), one categorical value per row
      X: np.ndarray of shape (N, D_in)    - real-valued features
      Y: np.ndarray of shape (N, D_out)   - real-valued targets
    """
    def __init__(self, cats: Sequence, X: np.ndarray, Y: np.ndarray, 
                Y_mask: np.ndarray,
                vocab: Dict = None):
        assert len(cats) == len(X) == len(Y), "cats, X, and Y must have the same length"
        assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"

        self.vocab = build_vocab(cats) if vocab is None else vocab
        self.cats = encode_categories(cats, self.vocab)            # (N,)
        self.X = X.astype(np.float32)                               # (N, D_in)
        self.Y = Y.astype(np.float32)                               # (N, D_out)
        self.Y_mask = Y_mask.astype(np.float32)                     # (N, D_out)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return tensors ready for the model
        return (
            torch.tensor(self.cats[idx], dtype=torch.long),         # categorical id
            torch.tensor(self.X[idx], dtype=torch.float32),         # real input vector
            torch.tensor(self.Y[idx], dtype=torch.float32),         # real target vector
            torch.tensor(self.Y_mask[idx], dtype=torch.float32),   # real target mask
        )


# ----------------------------
# Model
# ----------------------------
class CatRealRegressor(nn.Module):
    """
    Mixture-of-experts over categorical input:
      1) Shared embedding for the category.
      2) K linear heads map the embedding -> output vector (experts).
      3) A gating MLP consumes the real-valued input and outputs a softmax over heads.
      4) Final prediction = weighted sum of the heads with the gate weights.

    Args:
        num_categories: number of unique categories
        real_dim: dimension of real-valued input vector
        out_dim: dimension of prediction vector
        emb_dim: dimension of category embedding
        num_heads: number of expert heads
        gate_hidden: hidden size of gating MLP (single hidden layer by default)
        dropout: dropout applied on the embedding before experts (optional)
        softmax_temp: temperature for the gate softmax (lower -> sharper)
    """
    def __init__(
        self,
        num_categories: int,
        real_dim: int,
        out_dim: int,
        emb_dim: int = 16,
        num_heads: int = 8,
        gate_hidden: int = 32,
        dropout: float = 0.0,
        softmax_temp: float = 1.0,
    ):
        super().__init__()
        assert num_heads >= 2, "Use at least 2 heads to benefit from gating."

        # 1) Category embedding
        self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=emb_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # 2) Expert linear heads (embedding -> out_dim)
        self.num_heads = num_heads
        self.experts = nn.ModuleList([nn.Linear(emb_dim, out_dim) for _ in range(num_heads)])

        # 3) Gating network (real input -> weights over heads)
        #    Simple 1-hidden-layer MLP; feel free to deepen if needed.
        self.gate = nn.Sequential(
            nn.Linear(real_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_heads),
        )
        self.softmax_temp = softmax_temp

        # Optional: Initialize experts a bit conservatively
        for lin in self.experts:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, cat_idx: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            cat_idx: (B,)  long tensor of category ids
            x_real:  (B, real_dim) float tensor
        Output:
            y_hat:   (B, out_dim) float tensor
        """
        B = cat_idx.size(0)

        # Embed category and compute each expert head output
        emb = self.dropout(self.embedding(cat_idx))             # (B, emb_dim)
        head_outs = []
        for k in range(self.num_heads):
            yk = self.experts[k](emb)                           # (B, out_dim)
            head_outs.append(yk)
        H = torch.stack(head_outs, dim=1)                       # (B, num_heads, out_dim)

        # Gate weights from real input
        gate_logits = self.gate(x_real)                         # (B, num_heads)
        if self.softmax_temp != 1.0:
            gate_logits = gate_logits / self.softmax_temp
        w = torch.softmax(gate_logits, dim=1)                   # (B, num_heads)

        # Weighted sum of head outputs
        y_hat = torch.sum(H * w.unsqueeze(-1), dim=1)           # (B, out_dim)
        return y_hat

def _mean_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Column-wise Pearson r between y_true and y_pred, then mean across columns.
    Handles constant columns by treating their correlation as 0.0.
    y_true, y_pred: (N, D) arrays
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # center
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)

    # std (avoid divide by zero)
    syt = np.sqrt((yt ** 2).sum(axis=0) + eps)
    syp = np.sqrt((yp ** 2).sum(axis=0) + eps)

    # covariance per column
    cov = (yt * yp).sum(axis=0)

    r = cov / (syt * syp)           # shape (D,)
    r = np.clip(r, -1.0, 1.0)       # numeric safety
    # If a column is (near) constant in y_true or y_pred, correlation ~0 (already handled via eps)
    return float(np.nanmean(r))

# ----------------------------
# Training / Evaluation
# ----------------------------
@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-2
    weight_decay: float = 1e-3
    val_split: float = 0.0
    seed: int = 42
    num_workers: int = 0   # set >0 if you want background workers in DataLoader
    max_grad_norm: float = 1.0


def train_model(
    cats: Sequence,
    X: np.ndarray,
    Y: np.ndarray,
    Y_mask: np.ndarray,
    config: TrainConfig = TrainConfig(),
    emb_dim: int = 16,
) -> Dict:
    """
    Trains the model and returns a dict with: model, vocab, history, and device.
    """
    # Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Dataset & split
    full_ds = CatRealDataset(cats, X, Y, Y_mask)
    n_total = len(full_ds)
    n_val = int(n_total * config.val_split)
    n_train = n_total - n_val

    if n_val > 0:
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(config.seed))
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    else:
        train_loader = DataLoader(full_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = None

    # Model
    num_categories = len(full_ds.vocab)
    real_dim = X.shape[1]
    out_dim = Y.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatRealRegressor(
        num_categories=num_categories,
        real_dim=real_dim,
        out_dim=out_dim,
        emb_dim=emb_dim,
    ).to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    
    history = {
        "train_loss": [], "val_loss": [],
        "val_mae": [],
        "train_corr": [], "val_corr": []   # <-- new metrics
    }

    # Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0

        # for correlation we need full-epoch preds/targets
        train_preds_np = []
        train_targets_np = []

        for cat_ids, x_real, y, y_mask in train_loader:
            cat_ids = cat_ids.to(device)
            x_real = x_real.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            optimizer.zero_grad()
            preds = model(cat_ids, x_real)
            loss = criterion(preds * y_mask, y * y_mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            running += loss.item() * len(x_real)

            # collect for corr (detach -> cpu -> numpy)
            with torch.no_grad():
                train_preds_np.append((preds * y_mask).detach().cpu().numpy())
                train_targets_np.append((y * y_mask).detach().cpu().numpy())

        train_loss = running / n_train
        train_corr = _mean_pearson_corr(np.vstack(train_targets_np), np.vstack(train_preds_np))
        history["train_loss"].append(train_loss)
        history["train_corr"].append(train_corr)

        # Validation
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_mae_running = 0.0
            val_preds_np = []
            val_targets_np = []

            with torch.no_grad():
                for cat_ids, x_real, y, y_mask in val_loader:
                    cat_ids = cat_ids.to(device)
                    x_real = x_real.to(device)
                    y = y.to(device)
                    y_mask = y_mask.to(device)

                    preds = model(cat_ids, x_real)
                    loss = criterion(preds * y_mask, y * y_mask)
                    mae = torch.mean(torch.abs(preds - y))

                    val_running += loss.item() * len(x_real)
                    val_mae_running += mae.item() * len(x_real)

                    val_preds_np.append((preds * y_mask).detach().cpu().numpy())
                    val_targets_np.append((y * y_mask).detach().cpu().numpy())

            val_loss = val_running / max(1, n_val)
            val_mae = val_mae_running / max(1, n_val)
            val_corr = _mean_pearson_corr(np.vstack(val_targets_np), np.vstack(val_preds_np)) if n_val > 0 else float("nan")

            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["val_corr"].append(val_corr)       # <-- saved

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_MAE={val_mae:.4f} | train_corr={train_corr:.4f} | val_corr={val_corr:.4f}"
            )
        
        else:
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_corr={train_corr:.4f} | "
            )

    return {
        "model": model,
        "vocab": full_ds.vocab,  # category -> id mapping (save this!)
        "history": history,
        "device": device,
    }


def plot_history(history: dict):
    """
    Plot training/validation loss, MAE, and correlation curves.
    Expects keys:
      - 'train_loss', 'val_loss'
      - 'val_mae'
      - 'train_corr', 'val_corr'
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 4))

    # ---- Loss subplot ----
    plt.subplot(1, 2, 1)

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if len(history['train_loss']) == len(history['val_loss']):
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss")
    plt.legend()

    # ---- Correlation subplot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_corr"], label="Train Corr", color="green")
    if len(history['train_loss']) == len(history['val_loss']):
        plt.plot(epochs, history["val_corr"], label="Val Corr", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Pearson r")
    plt.title("Correlation")
    #plt.ylim(-1.0, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.show()

     # ---- MAE subplot ----
    if len(history['train_loss']) == len(history['val_loss']):
        plt.plot(epochs, history["val_mae"], marker="o", label="Val MAE", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Error")
        plt.title("Validation MAE")
        plt.legend()
    plt.show()


def get_category_embeddings(model, vocab: dict) -> pd.DataFrame:
    """
    Extract category embeddings as a DataFrame.
    
    Args:
        model: trained model that has `embedding` layer
        vocab: dict mapping category_name -> id (from dataset)
    
    Returns:
        DataFrame of shape (num_categories, emb_dim), 
        with category names as index.
    """
    # Get weight matrix (num_categories x emb_dim)
    emb_matrix = model.embedding.weight.detach().cpu().numpy()
    
    # Reverse vocab: id -> category
    id_to_cat = {i: c for c, i in vocab.items()}
    
    # Build dataframe
    df = pd.DataFrame(
        emb_matrix,
        index=[id_to_cat[i] for i in range(len(id_to_cat))],
        columns=[f"dim_{j}" for j in range(emb_matrix.shape[1])]
    )
    return df