import argparse
from pathlib import Path

from dependency_bootstrap import ensure_dependencies

ensure_dependencies(
    {
        "numpy": "numpy>=1.24",
        "pandas": "pandas>=2.0",
        "sklearn": "scikit-learn>=1.3",
        "matplotlib": "matplotlib>=3.7",
        "torch": "torch>=2.2",
    }
)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLPRegressor
from utils import ensure_dirs, plot_loss_curves, plot_predictions, regression_metrics, set_global_seed


FEATURE_COLUMNS = ["voltage", "temperature", "humidity"]
TARGET_COLUMN = "moisture"


def build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    y_pred = np.vstack(preds_list).squeeze()
    y_true = np.vstack(targets_list).squeeze()
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train MLP baseline for synthetic moisture regression.")
    parser.add_argument("--data-path", type=str, default="data/synthetic_moisture.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    ensure_dirs("data", args.output_dir)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please run generate_data.py first."
        )

    # Device selection for GPU/CPU
    print("=== Device Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    df = pd.read_csv(data_path)
    print("=== Data Loaded ===")
    print(f"Data path: {data_path.resolve()}")
    print(f"Shape: {df.shape}")

    missing_count = df.isnull().sum()
    print("\nMissing value check:")
    print(missing_count)

    if missing_count.sum() > 0:
        raise ValueError("Dataset contains missing values. Please clean data before training.")

    x = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # Split: train 70%, val 15%, test 15%
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=args.seed
    )

    print("\n=== Data Split ===")
    print(f"Train: {x_train.shape[0]} samples")
    print(f"Val:   {x_val.shape[0]} samples")
    print(f"Test:  {x_test.shape[0]} samples")

    # Fit scaler ONLY on training data to avoid leakage
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    train_loader = build_dataloader(x_train_scaled, y_train, args.batch_size, shuffle=True)
    val_loader = build_dataloader(x_val_scaled, y_val, args.batch_size, shuffle=False)
    test_loader = build_dataloader(x_test_scaled, y_test, args.batch_size, shuffle=False)

    model = MLPRegressor(input_dim=len(FEATURE_COLUMNS), hidden_dims=(64, 32), dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0
    train_losses = []
    val_losses = []

    checkpoint_path = Path(args.output_dir) / "best_model.pt"

    print("\n=== Start Training ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "feature_columns": FEATURE_COLUMNS,
                },
                checkpoint_path,
            )
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | "
                f"Best Val MSE: {best_val_loss:.5f} (epoch {best_epoch})"
            )

        if wait >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={args.patience}).")
            break

    # Load best checkpoint with map_location so CPU/GPU portability is preserved
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true_test, y_pred_test = predict(model, test_loader, device)
    metrics = regression_metrics(y_true_test, y_pred_test)

    print("\n=== Test Metrics ===")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2:  {metrics['r2']:.4f}")

    loss_plot_path = Path(args.output_dir) / "loss_curve.png"
    scatter_plot_path = Path(args.output_dir) / "pred_vs_true.png"
    plot_loss_curves(train_losses, val_losses, str(loss_plot_path))
    plot_predictions(y_true_test, y_pred_test, str(scatter_plot_path))

    print("\n=== Artifacts Saved ===")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    print(f"Loss curve:      {loss_plot_path.resolve()}")
    print(f"Pred scatter:    {scatter_plot_path.resolve()}")

    print("\n=== Post-Training Summary ===")
    print("1) Convergence: Training finished and best validation checkpoint was found.")
    gap = abs(train_losses[-1] - val_losses[-1])
    if gap < 1.0:
        fit_comment = "No obvious severe overfitting/underfitting in final epoch gap."
    else:
        fit_comment = "Noticeable train/val gap; tuning may be needed with real data."
    print(f"2) Loss curve interpretation: {fit_comment}")
    print(
        "3) Pipeline validation: SUCCESS. Synthetic-data generation, loading, splitting, "
        "scaling, training, evaluation, plotting, and checkpoint saving all ran end-to-end."
    )


if __name__ == "__main__":
    main()
