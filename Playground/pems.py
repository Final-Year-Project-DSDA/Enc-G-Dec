
import os
import pickle
import h5py
import gdown
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# --------------------------
# 1. Download METR-LA Dataset
# --------------------------
if not os.path.exists('dataset'):
    os.makedirs('dataset/METR-LA', exist_ok=True)

    # Download from Google Drive
    url = 'https://drive.google.com/uc?id=1q4FEjQg1EaT8Zqjh8CN_LV0ZMOLhpdg6'
    output = 'dataset/metr-la.zip'
    gdown.download(url, output, quiet=False)

    # Unzip files
    import zipfile
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('dataset/')

# --------------------------
# 2. Load and Preprocess Data
# --------------------------
# Load speed data
with h5py.File('dataset/METR-LA/metr-la.h5', 'r') as f:
    speed_data = f['df']['block0_values'][:]

# Load adjacency matrix
with open('dataset/METR-LA/adj_mx.pkl', 'rb') as f:
    adj_mx = pickle.load(f)
adjacency_matrix = adj_mx[2]  # Use distance-based adjacency matrix

# Normalize data
scaler = StandardScaler()
speed_data = scaler.fit_transform(speed_data)

# Train-test split
train_size = int(len(speed_data) * 0.8)
train_data = speed_data[:train_size]
test_data = speed_data[train_size:]

# Convert to tensors
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)

# --------------------------
# 3. Create Graph Structure
# --------------------------
# Convert adjacency matrix to edge index
adj_sparse = sp.coo_matrix(adjacency_matrix)
edge_index = torch.LongTensor(np.vstack([adj_sparse.row, adj_sparse.col]))
edge_attr = torch.FloatTensor(adj_sparse.data).unsqueeze(1)

# Create PyG graph data
graph = Data(edge_index=edge_index, edge_attr=edge_attr)

# --------------------------
# 4. Model Definition
# --------------------------
# (Use the same model classes from original code)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels):
        super().__init__()
        layers = []
        prev_layer = in_channels

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_layer, hidden_dim))
            layers.append(nn.ReLU())
            prev_layer = hidden_dim

        layers.append(nn.Linear(prev_layer, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GRUDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_gru=64):
        super().__init__()
        hidden_layers = [128, 64, 32]

        self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
        self.mlp = MLP(hidden_gru, hidden_layers, out_channels)

    def forward(self, z):
        gru_out, _ = self.gru(z)
        output = self.mlp(gru_out.squeeze())
        return output


class GNNEncoder1(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index,edge_attr=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GNNEncoder2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GNNEncoder3(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels, in_edge_channels=-1)
        self.conv2 = GeneralConv((-1, -1), out_channels, in_edge_channels=-1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GNNEncoder4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, edge_dim=None):
        super().__init__()
        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            beta=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True
        )
        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=heads,
            concat=True,
            beta=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None, encoder_type='GNNEncoder4'):
        super().__init__()
        if encoder_type == 'GNNEncoder1':
            self.encoder = GNNEncoder1(hidden_channels, hidden_channels)
        elif encoder_type == 'GNNEncoder2':
            self.encoder = GNNEncoder2(hidden_channels, hidden_channels)
        elif encoder_type == 'GNNEncoder3':
            self.encoder = GNNEncoder3(hidden_channels, hidden_channels)
        elif encoder_type == 'GNNEncoder4':
            self.encoder = GNNEncoder4(input_size, hidden_channels, hidden_channels, edge_dim=edge_attr.size(1) if edge_attr is not None else None)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.decoder = GRUDecoder(hidden_channels, out_channels)
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, x):
        z = self.encoder(x, self.edge_index, self.edge_attr)
        z = z.unsqueeze(1)
        return self.decoder(z)
# --------------------------
# 5. Initialize Model
# --------------------------
input_size = speed_data.shape[1]  # 207 sensors
hidden_channels = 64
out_channels = input_size

model = Model(
    input_size=input_size,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    edge_index=graph.edge_index,
    edge_attr=graph.edge_attr,
    encoder_type='GNNEncoder4'
)

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# --------------------------
# 6. Training Loop
# --------------------------
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Use sequence length of 12 for time series (adjust as needed)
    seq_len = 12
    inputs = train_data[:-seq_len]
    targets = train_data[seq_len:]

    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# --------------------------
# 7. Evaluation
# --------------------------
model.eval()
with torch.no_grad():
    test_inputs = test_data[:-12]
    test_targets = test_data[12:]

    predictions = model(test_inputs)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions.numpy())
    test_targets = scaler.inverse_transform(test_targets.numpy())

    # Calculate metrics
    mae = mean_absolute_error(test_targets, predictions)
    mse = mean_squared_error(test_targets, predictions)
    rmse = np.sqrt(mse)

    print(f'\nTest Metrics:')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')