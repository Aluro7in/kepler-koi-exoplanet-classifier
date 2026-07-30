"""
train_nn.py
Trains the MLP and the TabTransformer on the preprocessed Kepler TCE
features, with LIVE visualization of how the network learns:

  1. A live-updating loss/accuracy curve (train vs val), redrawn every
     epoch. In an interactive session (`python -i` / Jupyter / a normal
     desktop with a display) this pops up and animates in real time via
     matplotlib's interactive mode (plt.ion()).
  2. A PCA projection of the network's learned penultimate-layer /
     [CLS]-token embedding space, snapshotted every few epochs, so you can
     literally watch the two classes pull apart as training progresses.
     All snapshots are stitched into an animated GIF at the end
     (reports/figures/<model>_embedding_evolution.gif) so the pattern is
     visible even when running headless (e.g. this sandbox, CI, a server).

Run standalone:  python src/train_nn.py --model mlp
                  python src/train_nn.py --model transformer
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import joblib

from data_loader import load_labeled_tce
from preprocessing import build_feature_matrix, build_preprocessor
from split import grouped_train_val_test_split
from models_pytorch import MLPClassifier, TabTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1}


def get_data():
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
                           koi_cache_path="data/koi_cumulative.csv")
    X = build_feature_matrix(df)
    y = df["label"].map(LABEL_MAP).values
    groups = df["kepid"]

    train_idx, val_idx, test_idx = grouped_train_val_test_split(df, groups, "label")

    pre = build_preprocessor()
    X_train = pre.fit_transform(X.iloc[train_idx])
    X_val = pre.transform(X.iloc[val_idx])
    X_test = pre.transform(X.iloc[test_idx])

    joblib.dump(pre, "models/dl/preprocessor.joblib")
    joblib.dump(list(X.columns), "models/dl/feature_columns.joblib")

    return (X_train, y[train_idx]), (X_val, y[val_idx]), (X_test, y[test_idx]), X.shape[1]


def make_loader(X, y, batch_size=512, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_model(model_name: str, epochs: int = 30, live: bool = False,
                     batch_size: int = 512, d_model: int = 32, n_heads: int = 4,
                     n_layers: int = 2, patience: int = 10):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features = get_data()

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32).to(DEVICE)
    print(f"Class weights (NOT_PLANET, PLANET): {class_weights.tolist()}")

    if model_name == "mlp":
        model = MLPClassifier(n_features=n_features, n_classes=2).to(DEVICE)
        arch_config = {"n_features": n_features, "n_classes": 2}
    elif model_name == "transformer":
        model = TabTransformer(n_features=n_features, n_classes=2,
                                d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(DEVICE)
        arch_config = {"n_features": n_features, "n_classes": 2,
                        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers}
    else:
        raise ValueError(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    joblib.dump(arch_config, f"models/dl/{model_name}_config.joblib")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    embedding_snapshots = []   # (epoch, 2D-PCA embedding, labels) for the gif

    if live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))

    best_val_loss = float("inf")
    patience_ctr = 0
    snapshot_every = max(1, epochs // 12)  # ~12 frames across training

    print(f"\n=== Training {model_name.upper()} on {DEVICE} | {n_features} features "
          f"| {len(y_train)} train rows ===")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)
        tr_loss /= tr_n
        tr_acc = tr_correct / tr_n

        model.eval()
        va_loss, va_correct, va_n = 0.0, 0, 0
        all_emb, all_lab = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, emb = model(xb, return_embedding=True)
                loss = loss_fn(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_n += xb.size(0)
                if epoch % snapshot_every == 0 or epoch == epochs:
                    all_emb.append(emb.cpu().numpy())
                    all_lab.append(yb.cpu().numpy())
        va_loss /= va_n
        va_acc = va_correct / va_n
        sched.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if all_emb:
            emb_2d = PCA(n_components=2, random_state=42).fit_transform(np.concatenate(all_emb))
            embedding_snapshots.append((epoch, emb_2d, np.concatenate(all_lab)))

        print(f"epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"| val_loss={va_loss:.4f} acc={va_acc:.3f}")

        if live:
            ax.clear()
            ax.plot(history["train_loss"], label="train loss")
            ax.plot(history["val_loss"], label="val loss")
            ax.set_title(f"{model_name} — live training curve (epoch {epoch})")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
            plt.pause(0.01)

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            patience_ctr = 0
            torch.save(model.state_dict(), f"models/dl/{model_name}_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    print(f"Training finished in {time.time() - t0:.1f}s. Best val_loss={best_val_loss:.4f}")

    # reload best weights before final save/eval
    model.load_state_dict(torch.load(f"models/dl/{model_name}_best.pt"))
    torch.save(model.state_dict(), f"models/dl/{model_name}_final.pt")

    _plot_history(history, model_name)
    _plot_embedding_gif(embedding_snapshots, model_name)
    _plot_confusion_and_roc(model, X_test, y_test, model_name)

    # Log training history to database
    try:
        from store_and_ensemble import get_db_connection, store_training_history
        conn, db_type = get_db_connection()
        import datetime
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        store_training_history(conn, db_type, run_id, model_name, history)
        conn.close()
    except Exception as db_err:
        print(f"[WARNING] Failed to log training history to database: {db_err}")

    return model, history


def _plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name}: loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{model_name}: accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"reports/figures/{model_name}_training_curves.png", dpi=130)
    plt.close(fig)


def _plot_embedding_gif(snapshots, model_name):
    """Stitch PCA-projected embedding snapshots into an animated GIF so you
    can watch the network's internal representation separate the two
    classes over the course of training."""
    if not snapshots:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=2)
    gif_path = f"reports/figures/{model_name}_embedding_evolution.gif"

    # fix consistent axis limits across frames using the final snapshot
    all_xy = np.concatenate([s[1] for s in snapshots])
    xlim = (all_xy[:, 0].min() - 1, all_xy[:, 0].max() + 1)
    ylim = (all_xy[:, 1].min() - 1, all_xy[:, 1].max() + 1)

    with writer.saving(fig, gif_path, dpi=110):
        for epoch, emb_2d, labels in snapshots:
            ax.clear()
            for cls_val, cls_name, color in [(0, "NOT_PLANET", "tab:red"), (1, "PLANET", "tab:green")]:
                mask = labels == cls_val
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=8, alpha=0.5, label=cls_name, color=color)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(f"{model_name} learned representation (val set) — epoch {epoch}")
            ax.legend(loc="upper right")
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved embedding-evolution animation -> {gif_path}")


def _plot_confusion_and_roc(model, X_test, y_test, model_name):
    from sklearn.metrics import confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["NOT_PLANET", "PLANET"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"{model_name}: confusion matrix (test)")
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[1])
    axes[1].set_title(f"{model_name}: ROC curve (test)")
    fig.tight_layout()
    fig.savefig(f"reports/figures/{model_name}_test_diagnostics.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--live", action="store_true", help="pop up a live matplotlib window (needs a display)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=32, help="transformer token embedding dim")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train_one_model(args.model, epochs=args.epochs, live=args.live,
                     batch_size=args.batch_size, d_model=args.d_model,
                     n_heads=args.n_heads, n_layers=args.n_layers, patience=args.patience)
