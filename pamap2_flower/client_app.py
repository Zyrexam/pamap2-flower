"""client_app.py — Flower client (ClientApp) for PAMAP2 federated learning."""

import os
import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays,GetParametersRes, Status,EvaluateRes
from pamap2_flower.dataset import data_cleaning, data_preprocessing, create_fixed_windows, IMUDataset, columns
from pamap2_flower.task import CNNBiLSTMModel, create_data_loaders

# ----------------------------
# Flower Client (implements flwr.client.Client)
# ----------------------------
class PAMAP2Client(Client):
    def __init__(self, client_id, train_loader, test_loader, input_dim, num_classes=12, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = CNNBiLSTMModel(input_dim=input_dim, num_classes=num_classes, dropout=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            parameters=ndarrays_to_parameters(params),
            status=Status(code=fl.common.Code.OK, message="Success")
        )

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, np_val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(np_val, dtype=state_dict[k].dtype)
        self.model.load_state_dict(state_dict)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        print(f"[Client {self.client_id}] Starting local training for {self.local_epochs} epochs...")
        params = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(params)

        self.model.train()
        self.model.to(self.device)

        for epoch in range(self.local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = total_loss / len(self.train_loader)
            acc = correct / total
            print(f"[Client {self.client_id}] Epoch [{epoch+1}/{self.local_epochs}] | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return fl.common.FitRes(
            parameters=ndarrays_to_parameters(new_params),
            num_examples=len(self.train_loader.dataset),
            metrics={},
            status=Status(code=fl.common.Code.OK, message="Success")
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        print(f"[Client {self.client_id}] Evaluating global model...")
        params = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(params)

        self.model.eval()
        self.model.to(self.device)

        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                loss_total += loss.item()

        avg_loss = loss_total / len(self.test_loader)
        acc = correct / total
        print(f"[Client {self.client_id}] Test Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

        return EvaluateRes(
            loss=float(avg_loss),
            num_examples=total,
            metrics={"accuracy": float(acc)},
            status=Status(code=fl.common.Code.OK, message="Success")
            
        )

# ----------------------------
# client_fn: get epochs & partition_id from context
# ----------------------------

def client_fn(context: Context) -> Client:
    partition_id = int(context.node_config.get("partition_id", 0))
    local_epochs = int(context.run_config.get("local_epochs", 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    client_id = partition_id + 1
    print(f"[client_fn] Loading raw data for client_{client_id}...")

    raw_path = f"pamap2 dataset/subject10{client_id}.dat"
    df = pd.read_csv(raw_path, delim_whitespace=True, header=None)
    df.columns = columns

    df = data_cleaning(df)
    df = data_preprocessing(df)
    windows, labels = create_fixed_windows(df, window_size=50, shift=25)


    unique_labels = sorted(set(labels))
    label_mapping = {original: idx for idx, original in enumerate(unique_labels)}
    labels = [label_mapping[l] for l in labels]
    num_classes = len(unique_labels)
    print(f"[client_fn] Detected num_classes={num_classes}, label mapping: {label_mapping}")


    dataset = IMUDataset(windows, labels)
    train_loader, test_loader = create_data_loaders(dataset, train_ratio=0.8, batch_size=32)

    input_dim = windows[0].shape[1]

    return PAMAP2Client(
        client_id=client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        local_epochs=local_epochs
    )

# ----------------------------
# Flower ClientApp
# ----------------------------
app = ClientApp(client_fn=client_fn)
