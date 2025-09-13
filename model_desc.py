"""
bliqnet.py

A self-contained implementation of BLiqNet (Bidirectional Liquid Neural Network)
for response spectra prediction and inverse inference, including interpretability tools.

Usage:
    python bliqnet.py --mode train --data /path/to/ngawest2.csv --out_dir ./results
    python bliqnet.py --mode eval --model_path ./results/best_model.pth --scalers ./results
    python bliqnet.py --mode plots --model_path ./results/best_model.pth --scalers ./results

Note: edit DATA_COLUMNS and OUTPUT_COLUMNS to match your CSV file column names.
"""

import os
import argparse
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torchdiffeq import odeint_adjoint

# -------------------------
# Configuration / Defaults
# -------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default column names - change if your CSV uses different names
INPUT_COLUMNS = ['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']
OUTPUT_COLUMNS = [
    'pga', 'T0.010S', 'T0.020S', 'T0.030S', 'T0.040S', 'T0.050S', 'T0.060S',
    'T0.070S', 'T0.080S', 'T0.090S', 'T0.150S', 'T0.200S', 'T0.300S',
    'T0.500S', 'T0.600S', 'T0.700S', 'T0.800S', 'T0.900S', 'T1.000S',
    'T1.200S', 'T1.500S', 'T2.000S', 'T3.000S', 'T4.000S'
]

# Model & training hyperparams (explicit values for reproducibility)
HIDDEN_SIZE = 32
FORWARD_INTERMEDIATE = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
RTOL = 1e-4
ATOL = 1e-6
TIME_SPAN_POINTS = 50  # number of time points to sample the ODE trajectory if needed
TSPAN = torch.linspace(0.0, 1.0, TIME_SPAN_POINTS)

# Reproducibility
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Model definitions
# -------------------------
class LiquidNeuralLayer(nn.Module):
    """
    A simple Liquid Neural Layer representing dh/dt = x @ W_x + h @ W_h + b
    The layer stores the current input (self.x) before integration.
    """
    def __init__(self, input_size: int, hidden_size: int, decay: float = 0.0):
        super(LiquidNeuralLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_input = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hidden = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.decay = decay
        # temporary holder for input vector; assigned before ODE solve
        self.x = None

    def forward(self, t, h):
        # t: scalar time, h: [B, H]
        # self.x expected shape: [B, input_size]
        # Return dh/dt: [B, H]
        # Use broadcasting: compute x@W_input -> [B, H]
        if self.x is None:
            raise RuntimeError("LiquidNeuralLayer.x (input) must be set before calling the ODE solver.")
        # linear combinations
        x_term = torch.matmul(self.x, self.W_input)      # [B, H]
        h_term = torch.matmul(h, self.W_hidden)          # [B, H]
        dh = x_term + h_term + self.b
        if self.decay > 0.0:
            dh = -self.decay * h + dh
        return dh

class BiLiquidModel(nn.Module):
    """
    Bidirectional Liquid Neural Network.
    Forward head: hidden -> response spectra
    Backward head: hidden -> reconstructed input
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 forward_intermediate: int = FORWARD_INTERMEDIATE):
        super(BiLiquidModel, self).__init__()
        self.liquid_layer = LiquidNeuralLayer(input_size, hidden_size, decay=0.0)
        self.hidden_size = hidden_size

        # Forward prediction head
        self.fc_forward = nn.Sequential(
            nn.Linear(hidden_size, forward_intermediate),
            nn.ReLU(),
            nn.Linear(forward_intermediate, output_size)
        )
        # Backward reconstruction head
        self.fc_backward = nn.Linear(hidden_size, input_size)

    def forward(self, x, time_span=TSPAN.to(DEVICE), solver_rtol=RTOL, solver_atol=ATOL):
        """
        x: [B, d_in]
        returns: forward_pred [B, out_dim], backward_pred [B, in_dim]
        """
        batch_size = x.size(0)
        h0 = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # store input for the liquid layer
        self.liquid_layer.x = x

        # integrate ODE using adjoint method, returns [T, B, H]
        h_t = odeint_adjoint(self.liquid_layer, h0, time_span, rtol=solver_rtol, atol=solver_atol, method='dopri5')
        # take final hidden state
        h_final = h_t[-1]  # [B, H]

        forward_pred = self.fc_forward(h_final)
        backward_pred = self.fc_backward(h_final)
        return forward_pred, backward_pred, h_t  # return full traj if needed

# -------------------------
# Utilities
# -------------------------
def load_data(csv_path: str,
              input_cols=INPUT_COLUMNS,
              output_cols=OUTPUT_COLUMNS) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    # Ensure columns exist
    missing_in = [c for c in input_cols if c not in df.columns]
    missing_out = [c for c in output_cols if c not in df.columns]
    if missing_in or missing_out:
        raise ValueError(f"Missing columns in CSV. Missing inputs: {missing_in}, missing outputs: {missing_out}")
    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)
    return X, y, df

def split_and_dataload(X, y, batch_size=BATCH_SIZE, train_frac=0.9):
    N = X.shape[0]
    train_size = int(train_frac * N)
    val_size = N - train_size
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# -------------------------
# Training and evaluation
# -------------------------
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = EPOCHS,
                lr: float = LR,
                weight_decay: float = WEIGHT_DECAY,
                out_dir: str = "./results"):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    best_val_loss = float('inf')
    best_epoch = -1
    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            xb, yb = batch
            xb = xb[0].to(DEVICE) if isinstance(xb, tuple) else xb[0].to(DEVICE) if isinstance(xb, list) else xb.to(DEVICE)
            # but random_split dataset yields tuple tensors, so adjust
            if isinstance(batch[0], tuple) or isinstance(batch[0], list):
                xb, yb = batch[0]
            xb = batch[0].to(DEVICE) if isinstance(batch[0], torch.Tensor) else torch.stack(batch[0]).to(DEVICE)
            # fixed: dataset returns tensors, so simpler:
            xb = batch[0].to(DEVICE)
            yb = batch[1].to(DEVICE)

            optimizer.zero_grad()
            forward_pred, backward_pred, _ = model(xb)
            loss_fwd = criterion(forward_pred, yb)
            loss_bwd = criterion(backward_pred, xb)
            loss = loss_fwd + loss_bwd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg_train_loss = total_loss / batches
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(DEVICE)
                yb = batch[1].to(DEVICE)
                forward_pred, backward_pred, _ = model(xb)
                val_loss += (criterion(forward_pred, yb).item() + criterion(backward_pred, xb).item())
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (epoch {best_epoch})")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds. Best epoch: {best_epoch} val_loss: {best_val_loss:.6f}")
    return history

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, scaler_y=None):
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_pred, _, _ = model(X_t)
        y_pred = y_pred.cpu().numpy()
    if scaler_y is not None:
        # invert scaling
        y_pred = scaler_y.inverse_transform(y_pred)
        y_true = scaler_y.inverse_transform(y)
    else:
        y_true = y
    mse, mae, r2 = compute_metrics(y_true, y_pred)
    return mse, mae, r2, y_true, y_pred

# -------------------------
# Interpretability plots
# -------------------------
def plot_hidden_dynamics(model: nn.Module, X_scaled: np.ndarray, sample_indices: list, out_file: str = None):
    """
    Figure A: plot hidden states h(t) for a few sample inputs.
    """
    model.eval()
    tspan = torch.linspace(0.0, 1.0, TIME_SPAN_POINTS).to(DEVICE)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(8, 3 * len(sample_indices)), sharex=True)
    if len(sample_indices) == 1:
        axes = [axes]

    for idx, si in enumerate(sample_indices):
        x_np = X_scaled[si:si+1].astype(np.float32)
        x = torch.tensor(x_np).to(DEVICE)
        model.liquid_layer.x = x
        h0 = torch.zeros(1, model.hidden_size).to(DEVICE)
        h_t = odeint_adjoint(model.liquid_layer, h0, tspan, rtol=RTOL, atol=ATOL, method='dopri5')
        h_t = h_t.squeeze(1).cpu().numpy()  # [T, H]
        # plot first few hidden units
        n_show = min(6, h_t.shape[1])
        for j in range(n_show):
            axes[idx].plot(np.linspace(0, 1, h_t.shape[0]), h_t[:, j], label=f'h{j+1}')
        axes[idx].set_ylabel('Hidden-state activation (dimensionless)')
        axes[idx].set_title(f'Sample index {si}')
        axes[idx].legend(loc='upper right')
    axes[-1].set_xlabel('Normalized time t ∈ [0,1]')
    plt.suptitle('Hidden State Dynamics (Figure A)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_file:
        plt.savefig(out_file, dpi=200)
    plt.show()

def compute_jacobian(model: nn.Module, x_sample: np.ndarray):
    """
    Compute Jacobian dy/dx for a single sample using autograd.
    Returns numpy array of shape [output_dim, input_dim].
    """
    model.eval()
    x = torch.tensor(x_sample.astype(np.float32), requires_grad=True).unsqueeze(0).to(DEVICE)
    forward_pred, _, _ = model(x)
    out_dim = forward_pred.shape[1]
    jac = []
    for i in range(out_dim):
        grad_outputs = torch.zeros_like(forward_pred)
        grad_outputs[0, i] = 1.0
        grads = torch.autograd.grad(forward_pred, x, grad_outputs=grad_outputs, retain_graph=True)[0]  # [1, in_dim]
        jac.append(grads.detach().cpu().numpy().squeeze(0))
    jac = np.stack(jac, axis=0)  # [out_dim, in_dim]
    return jac

def plot_jacobian_heatmap(jacobian: np.ndarray, input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, out_file: str = None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(jacobian, cmap='coolwarm', center=0, xticklabels=input_cols, yticklabels=output_cols)
    plt.xlabel('Input features')
    plt.ylabel('Output spectral periods')
    plt.title('Input Contribution / Sensitivity Map (Figure B): ∂y/∂x')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=200)
    plt.show()

# -------------------------
# K-fold cross-validation helper
# -------------------------
from sklearn.model_selection import KFold

def kfold_evaluate(X_scaled, y_scaled, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
        train_loader, val_loader = split_and_dataload(X_train, y_train)
        model = BiLiquidModel(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE, output_size=y_train.shape[1])
        model = model.to(DEVICE)
        _ = train_model(model, train_loader, val_loader, epochs=EPOCHS)
        # evaluate on test set
        mse, mae, r2, ytrue, ypred = evaluate_model(model, X_test, y_test)
        print(f"Fold {fold+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        fold_metrics.append((mse, mae, r2))
    fold_metrics = np.array(fold_metrics)
    mean_metrics = fold_metrics.mean(axis=0)
    std_metrics = fold_metrics.std(axis=0)
    return mean_metrics, std_metrics

# -------------------------
# Command-line interface
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train | eval | plots | kfold')
    parser.add_argument('--data', type=str, default='ngawest2.csv', help='Path to CSV dataset')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model_path', type=str, help='Path to saved model state_dict (.pth)')
    parser.add_argument('--scalers', type=str, help='Directory path where scalers were saved')
    parser.add_argument('--sample_indices', type=str, default='0,100,500', help='Indices for hidden dynamics plots (comma-separated)')
    args = parser.parse_args()

    seed_everything(SEED)

    if args.mode == 'train':
        # Load data
        X, y, df = load_data(args.data)
        # Scaling to [-1,1]
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        os.makedirs(args.out_dir, exist_ok=True)
        joblib.dump(scaler_X, os.path.join(args.out_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_y, os.path.join(args.out_dir, 'scaler_y.pkl'))

        train_loader, val_loader = split_and_dataload(X_scaled, y_scaled)
        model = BiLiquidModel(input_size=X_scaled.shape[1], hidden_size=HIDDEN_SIZE, output_size=y_scaled.shape[1])
        history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, out_dir=args.out_dir)

        # Evaluate on full dataset (after training best checkpoint)
        best_model_path = os.path.join(args.out_dir, 'best_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        mse, mae, r2, ytrue, ypred = evaluate_model(model, X_scaled, y_scaled, scaler_y=scaler_y)
        print("Final evaluation on all data (after inverse transform):")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save final predictions for inspection
        np.save(os.path.join(args.out_dir, 'y_true.npy'), ytrue)
        np.save(os.path.join(args.out_dir, 'y_pred.npy'), ypred)
        # Save model state dict path printed in train_model

    elif args.mode == 'eval':
        if args.model_path is None or args.scalers is None:
            raise ValueError("For eval mode, --model_path and --scalers must be provided.")
        # load scalers
        scaler_X = joblib.load(os.path.join(args.scalers, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(args.scalers, 'scaler_y.pkl'))
        X, y, df = load_data(args.data)
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
        model = BiLiquidModel(input_size=X_scaled.shape[1], hidden_size=HIDDEN_SIZE, output_size=y_scaled.shape[1])
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        mse, mae, r2, ytrue, ypred = evaluate_model(model, X_scaled, y_scaled, scaler_y=scaler_y)
        print("Evaluation:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    elif args.mode == 'plots':
        if args.model_path is None or args.scalers is None:
            raise ValueError("For plots mode, provide --model_path and --scalers.")
        scaler_X = joblib.load(os.path.join(args.scalers, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(args.scalers, 'scaler_y.pkl'))
        X, y, df = load_data(args.data)
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
        model = BiLiquidModel(input_size=X_scaled.shape[1], hidden_size=HIDDEN_SIZE, output_size=y_scaled.shape[1])
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model = model.to(DEVICE)

        # Hidden dynamics
        sample_indices = [int(i) for i in args.sample_indices.split(',')]
        plot_hidden_dynamics(model, X_scaled, sample_indices, out_file=os.path.join(args.out_dir, 'hidden_dynamics.png'))

        # Jacobian sensitivity heatmap at mean input
        x_mean = X_scaled.mean(axis=0)
        jac = compute_jacobian(model, x_mean)
        plot_jacobian_heatmap(jac, input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, out_file=os.path.join(args.out_dir, 'jacobian_heatmap.png'))

    elif args.mode == 'kfold':
        X, y, df = load_data(args.data)
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        mean_metrics, std_metrics = kfold_evaluate(X_scaled, y_scaled, n_splits=5)
        print("5-fold CV mean metrics (MSE, MAE, R2):", mean_metrics)
        print("5-fold CV std metrics (MSE, MAE, R2):", std_metrics)
    else:
        raise ValueError("Unknown mode. Use train | eval | plots | kfold")

if __name__ == '__main__':
    main()
