"""task.py — Models and training utilities for PAMAP2-FLOWER project."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# -------------------------------------------------
# Gated Sensor Fusion Layer
# -------------------------------------------------
class GatedFusion(nn.Module):
    """
    Learns per-time-step gates to fuse features.
    Input: (batch, time, features)
    Output: gated features, same shape.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super(GatedFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.fc(x)  # (batch, time, features)
        return x * gates

# -------------------------------------------------
# CNN + BiLSTM Model with Gated Fusion and Dropout
# -------------------------------------------------
class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, lstm_layers=1, dropout=0.5):
        super(CNNBiLSTMModel, self).__init__()

        # CNN over time dimension
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Gated fusion after CNN
        self.gate = GatedFusion(64)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=lstm_layers,
                            bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        x: (batch, time, features)
        """
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.conv(x)       # (batch, 64, time)
        x = x.transpose(1, 2)  # (batch, time, 64)

        x = self.gate(x)       # gated fusion

        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim*2)

        # mean pooling over time
        x_pooled = lstm_out.mean(dim=1)

        x_pooled = self.dropout(x_pooled)

        logits = self.fc(x_pooled)

        return logits

# -------------------------------------------------
# Split dataset and create DataLoaders
# -------------------------------------------------
def create_data_loaders(dataset, train_ratio=0.7, batch_size=32):
    """
    Splits dataset into train/test and creates DataLoaders.
    """
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

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
