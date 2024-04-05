from tools import generate_data, generate_model_datas, get_matrix_level, generate_data_interval, augmented_matrix
from torch.utils.data import TensorDataset, DataLoader
from model import CNNModel, ResNet
import matplotlib.pyplot as plt
from mcts import get_mcts_cost
from munkres import Munkres
from enum import Enum, auto
import seaborn as sns
import torch.nn as nn
import numpy as np
import pickle
import torch
import tqdm
import os


class ModelManager(Enum):
    """Enum for the model manager"""
    CNN = auto()
    RESNET = auto()


def make_cnn_model(state_size: int, device: torch.device) -> CNNModel:
    """Create a CNN model"""
    return CNNModel(state_size=state_size, input_size=2, num_hidden=64).to(
        device
    )


def make_resnet_model(state_size: int, device: torch.device) -> ResNet:
    """Create a ResNet model"""
    return ResNet(
        state_size=state_size, input_size=2, num_resBlocks=16, num_hidden=128
    ).to(device)


CURRENT_MODEL = ModelManager.RESNET


def manage_graphs_generation() -> None:
    """Generate graphs and save them to a file"""
    print("\n---Generating graphs---")
    pairs = int(input("Number of pairs of graphs: "))
    size = int(input("Size of the graphs: "))
    n = int((1 + np.sqrt(1 + 8 * pairs)) / 2)
    graphs = generate_data(n, size)
    if not os.path.exists("processed"):
        os.makedirs("processed")
    with open(f"processed/{n}_graphs.pkl", "wb") as file:
        pickle.dump(graphs, file)


def manage_model_training(model_name: str) -> None:
    """Train the model"""
    print("\n---Training the model---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_size = int(input("Maximum graph size model can handle: "))
    max_graph_size = matrix_size // 2

    if CURRENT_MODEL == ModelManager.CNN:
        model = make_cnn_model(matrix_size**2, device)
    else:
        model = make_resnet_model(matrix_size**2, device)

    dataset_size = int(input("Number of graphs to train on: "))

    # Data generation
    training_graphs = generate_data_interval(
        dataset_size, max_graph_size, verbose=False, max_ged_size=0
    )

    # Data processing
    model_datas = generate_model_datas(training_graphs, matrix_size)
    test_percentage = 0.1
    test_size = int(test_percentage * len(model_datas[0]))
    print(f"Taille du dataset : {len(model_datas[0])}")
    print(f"Taille du dataset de test : {test_size}")

    train_states = model_datas[0][:-test_size]
    train_values = model_datas[1][:-test_size]

    test_states = model_datas[0][-test_size:]
    test_values = model_datas[1][-test_size:]

    dataset = TensorDataset(
        torch.tensor(train_states, dtype=torch.float32),
        torch.tensor(train_values, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = int(input("Number of epochs: "))
    losses = {
        "train": [],
        "test": []
    }
    for epoch in range(epochs):
        running_loss = 0
        for states, values in dataloader:
            optimizer.zero_grad()
            states = states.to(device)
            outputs = model(states)
            values = values.unsqueeze(1)
            values = torch.cat([1 - values, values], dim=1)
            values = values.to(device)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses["train"].append(running_loss / len(dataloader) * 10)

        if (epoch + 1) % 1 == 0:
            print(
                f"-----Epoch {epoch + 1}, loss: {running_loss / len(dataloader) * 10:.2f}-----"
            )
            # Test the model
            random_index = [np.random.randint(
                0, len(test_states)) for _ in range(100)]
            random_test_states = test_states[random_index]
            random_test_values = test_values[random_index]
            random_test_states = torch.tensor(
                random_test_states, dtype=torch.float32)
            random_test_values = torch.tensor(
                random_test_values, dtype=torch.float32)
            random_test_states = random_test_states.to(device)
            random_test_values = random_test_values.to(device)
            random_outputs = model(random_test_states)
            random_values = random_test_values.unsqueeze(1)
            random_values = torch.cat(
                [1 - random_values, random_values], dim=1)
            random_loss = criterion(random_outputs, random_values)
            print(f"Test loss: {random_loss.item() * 100:.2f}%")
            losses["test"].append(random_loss.item() * 100)

    # Save the losses
    _, ax = plt.subplots()
    ax.plot(losses["train"], label="Train loss")
    ax.plot(losses["test"], label="Test loss")
    ax.set_title("Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(f"images/{model_name}_losses.png")

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), f"models/{model_name}_{matrix_size}.pth")


def benchmark_model(model_name: str) -> None:
    """Benchmark the model"""
    print("\n---Benchmarking the model---")
    matrix_size = int(model_name.split("_")[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CURRENT_MODEL == ModelManager.CNN:
        model = make_cnn_model(matrix_size**2, device)
    else:
        model = make_resnet_model(matrix_size**2, device)
    model.load_state_dict(torch.load(
        f"models/{model_name}.pth", map_location=device)
    )

    max_graph_size = matrix_size // 2
    dataset_size = int(input("Number of graphs to test on: "))

    # Data generation
    test_graphs = generate_data_interval(
        dataset_size, max_graph_size, verbose=False, max_ged_size=0
    )

    # Data processing
    model_datas = generate_model_datas(test_graphs, matrix_size)
    print(f"Taille du dataset de test : {len(model_datas[0])}")

    test_states = model_datas[0]
    test_values = model_datas[1]

    level_matrix = {
        level: [] for level in range(matrix_size)
    }
    for i in range(len(test_states)):
        level = get_matrix_level(test_states[i][1])
        level_matrix[level].append(i)

    min_level = min(len(level_matrix[level]) for level in level_matrix)
    max_level = max(len(level_matrix[level]) for level in level_matrix)

    # show model accuracy for each level
    accuracies = []
    for level in level_matrix:
        correct = 0
        if len(level_matrix[level]) == 0:
            accuracies.append(0)
            continue
        for i in level_matrix[level]:
            outputs = model(torch.tensor(
                test_states[i].reshape(1, 2, matrix_size, matrix_size), dtype=torch.float32).to(device))
            if torch.argmax(outputs).item() == test_values[i]:
                correct += 1
        accuracies.append(correct / len(level_matrix[level]))

    print(f"Accuracy by level: {np.round(accuracies, 3)}")
    _, ax = plt.subplots()
    sns.set_style()
    sns.barplot(
        x=np.array(list(level_matrix.keys())),
        y=np.array(accuracies), ax=ax, palette="viridis"
    )
    ax.set_title("Accuracy du modèle par niveau")
    ax.set_xlabel("Niveau")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"images/{model_name}_accuracy_by_level.png")


def benchmark_mcts(model_name: str) -> None:
    print("\n---Benchmarking MCTS---")
    matrix_size = int(model_name.split("_")[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CURRENT_MODEL == ModelManager.CNN:
        model = make_cnn_model(matrix_size**2, device)
    else:
        print("ok")
        model = make_resnet_model(matrix_size**2, device)
    model.load_state_dict(torch.load(
        f"models/{model_name}.pth", map_location=device)
    )

    max_graph_size = matrix_size // 2
    dataset_size = int(input("Number of graphs to test on: "))

    # Data generation
    test_graphs = generate_data_interval(
        dataset_size, max_graph_size, verbose=False, max_ged_size=0
    )

    print(f"Taille du dataset de test : {len(test_graphs)}")

    config = {
        "exploration_weight": 2,
        "num_simulations": 1000,
        "device": device,
        "temperature": 2,
        "stable_iterations": 100,
        "rollout_policy": "random",
        "expansion_policy": "model"
    }

    mcts_values = []
    munkres_values = []
    for i in tqdm.tqdm(range(len(test_graphs))):
        matrix = np.array(test_graphs.loc[i, "cost_matrix"])
        g1 = test_graphs.loc[i, "g1"]
        g2 = test_graphs.loc[i, "g2"]
        max_depth = matrix.shape[0]
        new_matrix = augmented_matrix(matrix, matrix_size, g1, g2)
        _, value = get_mcts_cost(new_matrix, model, config, max_depth)
        mcts_values.append(value)

        munkres = Munkres()
        indexes = munkres.compute(matrix.tolist())
        munkres_values.append(sum(matrix[i][j] for i, j in indexes))

    # Plot the results
    error = np.sum(np.abs(np.array(munkres_values) - np.array(mcts_values))) / len(munkres_values)
    sns.set_theme()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=munkres_values, y=mcts_values)
    plt.plot([0, max(munkres_values)], [0, max(munkres_values)], color='red')

    plt.xlabel("Munkres values")
    plt.ylabel("MCTS values")
    plt.title(f"Munkres vs MCTS values, error = {str(error)}")
    plt.savefig(f"images/{model_name}_mcts_vs_munkres.png")


def benchmark_mcts_ged(model_name: str) -> None:
    print("\n---Benchmarking MCTS GED---")
    dataset_name = input("Dataset name: ")
    matrix_size = int(model_name.split("_")[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CURRENT_MODEL == ModelManager.CNN:
        model = make_cnn_model(matrix_size**2, device)
    else:
        model = make_resnet_model(matrix_size**2, device)
    model.load_state_dict(torch.load(
        f"models/{model_name}.pth", map_location=device)
    )

    dataset_size = int(input("Number of graphs to test on: "))

    # Data generation
    with open(f"processed/{dataset_name}_train.pkl", "rb") as file:
        test_graphs = pickle.load(file)

    print(f"Taille du dataset de test : {len(test_graphs)}")

    config = {
        "exploration_weight": 2,
        "num_simulations": 1000,
        "device": device,
        "temperature": 2,
        "stable_iterations": 100,
        "rollout_policy": "random",
        "expansion_policy": "model"
    }

    mcts_values = []
    munkres_values = []
    for i in tqdm.tqdm(range(dataset_size)):
        matrix = np.array(test_graphs.loc[i, "cost_matrix"])
        g1 = test_graphs.loc[i, "g1"]
        g2 = test_graphs.loc[i, "g2"]
        max_depth = matrix.shape[0]
        new_matrix = augmented_matrix(matrix, matrix_size, g1, g2)
        _, value = get_mcts_cost(new_matrix, model, config, max_depth)
        mcts_values.append(value / 2)

        munkres = Munkres()
        indexes = munkres.compute(matrix.tolist())
        munkres_values.append(sum(matrix[i][j] for i, j in indexes) / 2)

    ged_values = [test_graphs.loc[i, "ged"] for i in range(dataset_size)]

    mcts_mse = np.mean((np.array(ged_values) - np.array(mcts_values)) ** 2)
    munkres_mse = np.mean((np.array(ged_values) - np.array(munkres_values)) ** 2)

    sns.set_theme()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(ged_values, munkres_values, c=(np.array(munkres_values) - np.array(ged_values)) ** 2, cmap="viridis", alpha=0.5)
    axs[0].plot([0, max(munkres_values)], [0, max(munkres_values)], color="red")
    axs[0].set_title(f"Munkres vs GED, MSE = {str(munkres_mse)}")
    axs[0].set_xlabel("Ground truth")
    axs[0].set_ylabel("Predicted")

    axs[1].scatter(ged_values, mcts_values, c=(np.array(mcts_values) - np.array(ged_values)) ** 2, cmap="viridis", alpha=0.5)
    axs[1].plot([0, max(mcts_values)], [0, max(mcts_values)], color="red")
    axs[1].set_title(f"MCTS vs GED, MSE = {str(mcts_mse)}")
    axs[1].set_xlabel("Ground truth")
    axs[1].set_ylabel("Predicted")

    axs[2].scatter(munkres_values, mcts_values, c=(np.array(mcts_values) - np.array(munkres_values)) ** 2, cmap="viridis", alpha=0.5)
    axs[2].plot([0, max(munkres_values)], [0, max(munkres_values)], color="red")
    axs[2].set_title(f"Munkres vs MCTS, MSE = {str(np.mean((np.array(munkres_values) - np.array(mcts_values)) ** 2))}")
    axs[2].set_xlabel("Munkres")
    axs[2].set_ylabel("MCTS")

    plt.savefig(f"images/{model_name}_mcts_vs_munkres_vs_ged.png")


