## 📄 **README.md**

````markdown
# pamap2-flower: Federated Learning on PAMAP2 with PyTorch and Flower

A research-grade Flower + PyTorch project that implements **CNN + BiLSTM + Gated Fusion**  
for human activity recognition on the **PAMAP2** dataset, using **federated learning**.

- Each client trains on data from one subject (simulated federated clients)
- Uses real sensor fusion: hand, chest, and ankle IMUs → fused via a learned gating network

````
## 📦 Installation

Clone this repo and install dependencies in **editable** mode:

```bash
pip install -e .
````

Or (recommended) use conda:

```bash
conda env create -f environment.yml
conda activate pamap2-flower
```

---

## 📥 Download the PAMAP2 dataset

1. Download from [UCI PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
2. Place the `subject10X.dat` files in:

```plaintext
pamap2 dataset/
```

Make sure you have files like:

```plaintext
pamap2 dataset/subject101.dat
pamap2 dataset/subject102.dat
...
```

---

## 🚀 Run federated learning simulation

In the **project root** directory:

```bash
flwr run .
```

This will:

* start Flower Simulation Engine
* load data per client
* run federated averaging for multiple rounds

Logs will show:
✅ local training on each client
✅ global aggregation
✅ test accuracy and loss after each round

---

## 📡 Run with the Deployment Engine

> **Note**: for real distributed clients (not simulation)
> see Flower docs:
> 📚 [https://flower.dev/docs](https://flower.dev/docs)

---

## 🧠 **Highlights**

* Dynamic number of classes detected from data
* Real multimodal gated fusion over hand, chest, ankle streams
* CNN + BiLSTM classifier
* Server aggregates accuracy and loss using weighted average

---

## 🛠 Project structure

```plaintext
pamap2_flower/
 ├── dataset.py        # Preprocessing, sliding windows, PyTorch Dataset
 ├── task.py          # Model: CNN + BiLSTM + gated fusion
 ├── client_app.py    # Flower client logic
 └── server_app.py    # Flower server logic
```
