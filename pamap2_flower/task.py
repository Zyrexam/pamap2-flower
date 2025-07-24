"""task.py — Models and training utilities for PAMAP2-FLOWER project."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# -------------------------------------------------
# Small CNN block used for each sensor stream
# -------------------------------------------------
class CNNBlock(nn.Module):
    def __init__(self, input_dim, conv_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU()
        )
    def forward(self, x):
        # x: (batch, time, features) → transpose to (batch, features, time)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)  # back to (batch, time, conv_channels)
        return x

# -------------------------------------------------
# Gated fusion layer
# -------------------------------------------------
class GatedFusion(nn.Module):
    """
    Fuse 3 sensor streams using learned gates.
    """
    def __init__(self, conv_channels):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(3*conv_channels, 3*conv_channels),
            nn.ReLU(),
            nn.Linear(3*conv_channels, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_hand, x_chest, x_ankle):
        # concat features at each timestep
        concat = torch.cat([x_hand, x_chest, x_ankle], dim=-1)  # shape (batch, time, 3*channels)
        weights = self.gate_fc(concat)  # shape (batch, time, 3)
        w1, w2, w3 = weights.chunk(3, dim=-1)  # each (batch, time, 1)
        fused = w1 * x_hand + w2 * x_chest + w3 * x_ankle
        return fused

# -------------------------------------------------
# Full model: 3 CNN streams → gated fusion → BiLSTM → classifier
# -------------------------------------------------
class CNNBiLSTMModel(nn.Module):
    def __init__(self, num_hand_features, num_chest_features, num_ankle_features,
                 num_classes, conv_channels=64, hidden_dim=64, lstm_layers=1, dropout=0.5):
        super().__init__()
        self.cnn_hand = CNNBlock(num_hand_features, conv_channels)
        self.cnn_chest = CNNBlock(num_chest_features, conv_channels)
        self.cnn_ankle = CNNBlock(num_ankle_features, conv_channels)

        self.gate = GatedFusion(conv_channels)

        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_dim,
                            num_layers=lstm_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, time, total_features=18)
        x_hand = x[:, :, :6]
        x_chest = x[:, :, 6:12]
        x_ankle = x[:, :, 12:]

        out_hand = self.cnn_hand(x_hand)
        out_chest = self.cnn_chest(x_chest)
        out_ankle = self.cnn_ankle(x_ankle)

        fused = self.gate(out_hand, out_chest, out_ankle)

        lstm_out, _ = self.lstm(fused)
        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

# -------------------------------------------------
# Split dataset and create DataLoaders
# -------------------------------------------------
def create_data_loaders(dataset, train_ratio=0.8, batch_size=32):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )


# -------------------------------------------------
# Training function
# -------------------------------------------------
def train_model(model, train_loader, optimizer, criterion, device='cpu', epochs=10):
    """
    Standard PyTorch training loop.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

# -------------------------------------------------
# Evaluation function
# -------------------------------------------------
def evaluate_model(model, test_loader, device='cpu'):
    """
    Standard PyTorch evaluation loop.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc
