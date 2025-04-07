
import numpy as np
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/SDA Codes/Complete_GAN/final_data.csv')
df.head()
df = df.iloc[:,2:]

df.columns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# For CUDA
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure reproducibility when splitting data
torch.manual_seed(seed)

pip install torch-optimizer

pip install torchdiffeq

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, r2_score
from torchdiffeq import odeint_adjoint
import joblib  # For saving scalers as .pkl
import pandas as pd


# Define input and output features
input_features = ['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']
output_features = [
    'pga', 'T0.010S', 'T0.020S', 'T0.030S', 'T0.040S', 'T0.050S', 'T0.060S',
    'T0.070S', 'T0.080S', 'T0.090S', 'T0.150S', 'T0.200S', 'T0.300S',
    'T0.500S', 'T0.600S', 'T0.700S', 'T0.800S', 'T0.900S', 'T1.000S',
    'T1.200S', 'T1.500S', 'T2.000S', 'T3.000S', 'T4.000S'
]

# Extract inputs and outputs
X = df[input_features]
y = df[output_features]

# Apply Min-Max Scaling (-1 to 1)
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(scaler_X, '/content/drive/MyDrive/LiquidRespSpec/final_models/scaler_X.pkl')
joblib.dump(scaler_y, '/content/drive/MyDrive/LiquidRespSpec/final_models/scaler_y.pkl')

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split into train and validation sets (90%-10%)
train_size = int(0.9 * len(X_tensor))
val_size = len(X_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define Liquid Neural Layer
class LiquidNeuralLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidNeuralLayer, self).__init__()
        self.hidden_size = hidden_size

        # Parameters for the ODE system
        self.W_input = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hidden = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, t, h):
        # Dynamical equation: dh/dt = f(h, x)
        dh = torch.matmul(self.x, self.W_input) + torch.matmul(h, self.W_hidden) + self.b
        return dh

# Define Bidirectional Liquid Network
class BiLiquidModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLiquidModel, self).__init__()

        # Liquid Neural Layer
        self.liquid_layer = LiquidNeuralLayer(input_size, hidden_size)

        # Forward Prediction (Input → Output)
        self.fc_forward = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        # Backward Prediction (Output → Input)
        self.fc_backward = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.liquid_layer.hidden_size).to(x.device)
        time_span = torch.tensor([0.0, 1.0]).float().to(x.device)

        # Store input in layer
        self.liquid_layer.x = x

        # Solve ODE
        h_t = odeint_adjoint(self.liquid_layer, h0, time_span)
        h_final = h_t[-1]

        # Forward and Backward Predictions
        forward_pred = self.fc_forward(h_final)
        backward_pred = self.fc_backward(h_final)

        return forward_pred, backward_pred

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLiquidModel(input_size=len(input_features), hidden_size=32, output_size=len(output_features)).to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 50
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        forward_pred, backward_pred = model(batch_x)

        loss_fwd = criterion(forward_pred, batch_y)
        loss_bwd = criterion(backward_pred, batch_x)

        total_loss = loss_fwd + loss_bwd
        total_loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {total_loss.item():.4f}')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            forward_pred, backward_pred = model(batch_x)

            val_loss += criterion(forward_pred, batch_y).item() + criterion(backward_pred, batch_x).item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model, '/content/drive/MyDrive/LiquidRespSpec/final_models/best_liquid_model.pth')
        print(f'Model saved at epoch {epoch+1}')

# Load Model for Predictions
model = torch.load('/content/drive/MyDrive/LiquidRespSpec/final_models/best_liquid_model.pth')
model.eval()

# Predictions (Forward and Backward)
sample_input = torch.tensor(X_scaled[:5], dtype=torch.float32).to(device)
pred_forward, pred_backward = model(sample_input)

print("Predicted Forward:", pred_forward.cpu().detach().numpy())
print("Predicted Backward:", pred_backward.cpu().detach().numpy())

from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Reset index (optional)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Extract input & output features
X_test = df_test[input_features]
y_test = df_test[output_features]

# Scale using the previously saved scalers
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# Select the first 25 samples to match the size of predicted_values
true_values = true_values[:25, :]  # Select first 25 samples from true_values

# Now you can safely generate the plots
plt.figure(figsize=(18, 12))

for i, feature in enumerate(selected_outputs):
    plt.subplot(3, 3, i + 1)  # Create a 3x3 grid of plots
    plt.scatter(true_values[:, i], predicted_values[:, i], c='blue', alpha=0.5)
    plt.plot([np.min(true_values[:, i]), np.max(true_values[:, i])],
             [np.min(true_values[:, i]), np.max(true_values[:, i])], color='red', linestyle='--')

    # Dynamically adjust x and y axis limits based on the data
    min_val = min(np.min(true_values[:, i]), np.min(predicted_values[:, i]))
    max_val = max(np.max(true_values[:, i]), np.max(predicted_values[:, i]))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.title(f"True vs Predicted: {feature}")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True)

# Adjust layout and show plots
plt.tight_layout()
plt.show()

