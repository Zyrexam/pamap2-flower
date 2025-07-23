from flwr.common import Context
from flwr.common.typing import Metrics, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple

# ----------------------------
# Aggregate client metrics: weighted average accuracy and loss
# ----------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average accuracy and loss across clients and print nicely."""
    total_examples = 0 
    weighted_accuracies = 0.0
    weighted_losses = 0.0

    for num_examples, m in metrics:
        acc = float(m.get("accuracy", 0.0))
        loss = float(m.get("loss", 0.0))
        weighted_accuracies += acc * num_examples
        weighted_losses += loss * num_examples
        total_examples += num_examples

    avg_accuracy = weighted_accuracies / total_examples if total_examples > 0 else 0.0
    avg_loss = weighted_losses / total_examples if total_examples > 0 else 0.0

    return {"accuracy": avg_accuracy, "loss": avg_loss}


# ----------------------------
# Server function
# ----------------------------
def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num_server_rounds", 5))
    local_epochs = int(context.run_config.get("local_epochs", 1))

    print(f"[Server] num_rounds: {num_rounds}")
    print(f"[Server] local_epochs for clients: {local_epochs}")

    # Send local_epochs to clients each round
    def on_fit_config_fn(rnd: int) -> dict:
        return {"local_epochs": local_epochs}

    # Define strategy with our custom metrics aggregation
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=9,
        min_evaluate_clients=9,
        min_available_clients=9,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# ----------------------------
# Create ServerApp
# ----------------------------
app = ServerApp(server_fn=server_fn)
