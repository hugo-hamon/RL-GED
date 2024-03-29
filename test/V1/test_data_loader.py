from compute_ged import compute_ged
from cost_matrix import cost_path
import matplotlib.pyplot as plt
from munkres import Munkres
from data_loader import *
import seaborn as sns
import networkx as nx
import numpy as np
import pickle
import tqdm
import os

DATASET = "Linux"

TRAIN_PATH = f"../../../Dataset/{DATASET}_train/{DATASET}_train.json"
TEST_PATH = f"../../../Dataset/{DATASET}_test/{DATASET}_test.json"
GED_PATH = f"../../../Dataset/{DATASET}_csv/{DATASET}.csv"

GENERATE = False

if __name__ == "__main__":

    if GENERATE:
        data = load_data(TRAIN_PATH, GED_PATH)
        
        # Save the data with pickle
        if not os.path.exists("processed"):
            os.makedirs("processed")

        with open(f"processed/{DATASET}_train.pkl", "wb") as file:
            pickle.dump(data, file)

    # Load the train with pickle
    with open(f"processed/{DATASET}_train.pkl", "rb") as file:
        data = pickle.load(file)
    
    result = []
    print(f"Nombre de données: {data.shape[0]}")
    for i in tqdm.tqdm(range(data.shape[0])[:1000]):
        g1 = data.iloc[i]["g1"]
        g2 = data.iloc[i]["g2"]


        data_row = data.iloc[i]
        cost_matrix = np.array(data_row["cost_matrix"])
        # Compute the cost with the munkres algorithm
        m = Munkres()
        indexes = m.compute(cost_matrix.tolist())
        cost = 0
        for row, column in indexes:
            value = cost_matrix[row][column]
            cost += value
        cost /= 2
        estimated_ged = cost_path(g1, g2, indexes, COST)

        # assigment matrix
        assigment_matrix = np.zeros((len(g1.nodes) + len(g2.nodes), len(g1.nodes) + len(g2.nodes)))
        for row, column in indexes:
            assigment_matrix[row, column] = 1
        estimated_ged2 = compute_ged(g1, g2, list(g1.nodes), list(g2.nodes), assigment_matrix)
        result.append((cost, estimated_ged, estimated_ged2, data_row["ged"]))

    # Compute the MSE
    result = np.array(result)

    # Plot the result
    mse = np.mean((result[:, 0] - result[:, 3]) ** 2)
    sns.set_theme()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(result[:, 3], result[:, 0], c=(result[:, 0] - result[:, 3]) ** 2, cmap="viridis", alpha=0.5)
    axs[0].plot([0, max(result[:, 0])*1.2], [0, max(result[:, 0])*1.2], color="red")
    axs[0].set_xlabel("Ground truth")
    axs[0].set_ylabel("Predicted")
    axs[0].set_title(f"Lower bound, MSE: {mse}")

    mse = np.mean((result[:, 1] - result[:, 3]) ** 2)
    axs[1].scatter(result[:, 3], result[:, 1], c=(result[:, 1] - result[:, 3]) ** 2, cmap="viridis", alpha=0.5)
    axs[1].plot([0, max(result[:, 1])*1.2], [0, max(result[:, 1])*1.2], color="red")
    axs[1].set_xlabel("Ground truth")
    axs[1].set_ylabel("Predicted")
    axs[1].set_title(f"Upper bound, MSE: {mse}")

    mse = np.mean((result[:, 2] - result[:, 3]) ** 2)
    axs[2].scatter(result[:, 3], result[:, 2], c=(result[:, 2] - result[:, 3]) ** 2, cmap="viridis", alpha=0.5)
    axs[2].plot([0, max(result[:, 2])*1.2], [0, max(result[:, 2])*1.2], color="red")
    axs[2].set_xlabel("Ground truth")
    axs[2].set_ylabel("Predicted")
    axs[2].set_title(f"Upper bound, MSE: {mse}")

    plt.show()


    

