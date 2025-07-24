import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import Client, ClientApp
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, GetParametersRes, Status, EvaluateRes
from pamap2_flower.dataset import data_cleaning, data_preprocessing, create_fixed_windows, IMUDataset, columns
from pamap2_flower.task import CNNBiLSTMModel, create_data_loaders

class PAMAP2Client(Client):
    def __init__(self, client_id, train_loader, test_loader,
                 num_hand_features, num_chest_features, num_ankle_features,
                 num_classes, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = CNNBiLSTMModel(
            num_hand_features=num_hand_features,
            num_chest_features=num_chest_features,
            num_ankle_features=num_ankle_features,
            num_classes=num_classes
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, ins):
        params = [v.cpu().numpy() for v in self.model.state_dict().values()]
        return GetParametersRes(parameters=ndarrays_to_parameters(params), status=Status(code=fl.common.Code.OK, message="Success"))

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, np_val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(np_val, dtype=state_dict[k].dtype)
        self.model.load_state_dict(state_dict)

    def fit(self, ins):
        print(f"[Client {self.client_id}] Training for {self.local_epochs} epochs...")
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        self.model.train().to(self.device)

        for epoch in range(self.local_epochs):
            total_loss, correct, total = 0, 0, 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            print(f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} | Loss: {total_loss/len(self.train_loader):.4f} | Acc: {correct/total*100:.2f}%")

        new_params = [v.cpu().numpy() for v in self.model.state_dict().values()]
        return fl.common.FitRes(parameters=ndarrays_to_parameters(new_params),
                                num_examples=len(self.train_loader.dataset), metrics={}, status=Status(code=fl.common.Code.OK, message="Success"))

    def evaluate(self, ins):
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        self.model.eval().to(self.device)

        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_total += self.criterion(logits, y).item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        acc = correct/total
        return EvaluateRes(loss=loss_total/len(self.test_loader), num_examples=total,
                           metrics={"accuracy": acc}, status=Status(code=fl.common.Code.OK, message="Success"))


# -----------------------
# client_fn
# -----------------------
def client_fn(context: Context):
    partition_id = int(context.node_config.get("partition_id", 0))
    local_epochs = int(context.run_config.get("local_epochs", 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client_id = partition_id + 1

    print(f"[client_fn] Loading data for client_{client_id}")
    df = pd.read_csv(f"pamap2 dataset/subject10{client_id}.dat", delim_whitespace=True, header=None)
    df.columns = columns

    df = data_cleaning(df)
    df = data_preprocessing(df)
    windows, labels = create_fixed_windows(df)

    # map labels to consecutive numbers
    label_map = {orig: i for i, orig in enumerate(sorted(set(labels)))}
    labels = [label_map[l] for l in labels]
    num_classes = len(label_map)

    dataset = IMUDataset(windows, labels)
    train_loader, test_loader = create_data_loaders(dataset)

    # known from dataset: hand 6, chest 6, ankle 6 features
    return PAMAP2Client(
        client_id=client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        num_hand_features=6, num_chest_features=6, num_ankle_features=6,
        num_classes=num_classes,
        device=device,
        local_epochs=local_epochs
    )

# ----------------------------
# Flower ClientApp
# ----------------------------
app = ClientApp(client_fn=client_fn)










            