from cost_matrix import cost_path
import matplotlib.pyplot as plt
from munkres import Munkres
from data_loader import *
import seaborn as sns
import numpy as np
import pickle
import tqdm
import os

TRAIN_PATH = "../../Dataset/IMDB_train/IMDB_train.json"
TEST_PATH = "../../Dataset/IMDB_test/IMDB_test.json"
GED_PATH = "../../Dataset/IMDB_csv/IMDB.csv"

if __name__ == "__main__":
    """
    data = load_data(TRAIN_PATH, GED_PATH)
    
    # Save the data with pickle
    if not os.path.exists("processed"):
        os.makedirs("processed")

    with open("processed/IMDB_train.pkl", "wb") as file:
        pickle.dump(data, file)
    """
    # Load the train with pickle
    with open("processed/IMDB_train.pkl", "rb") as file:
        data = pickle.load(file)

    result = []
    print(f"Nombre de données: {data.shape[0]}")
    for i in tqdm.tqdm(range(data.shape[0])):
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
        estimated_ged = cost_path(g1, g2, indexes, COST)
        result.append((cost, estimated_ged, data_row["ged"]))

    # Compute the MSE
    result = np.array(result)
    mse = np.mean((result[:, 0] - result[:, 2]) ** 2)


    # Plot the result
    sns.set_theme()
    plt.scatter(result[:, 2], result[:, 0], c=(result[:, 0] - result[:, 2]) ** 2, cmap="viridis", alpha=0.5)
    plt.colorbar()
    plt.plot([0, max(result[:, 0])*1.2], [0, max(result[:, 0])*1.2], color="red")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.title(f"MSE: {mse}")
    plt.show()


    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].scatter(result[:, 2], result[:, 0], c=(result[:, 0] - result[:, 2]) ** 2, cmap="viridis", alpha=0.5)
    axs[0].plot([0, max(result[:, 0])*1.2], [0, max(result[:, 0])*1.2], color="red")
    axs[0].set_xlabel("Ground truth")
    axs[0].set_ylabel("Predicted")
    axs[0].set_title("Lower bound")

    axs[1].scatter(result[:, 2], result[:, 1], c=(result[:, 1] - result[:, 2]) ** 2, cmap="viridis", alpha=0.5)
    axs[1].plot([0, max(result[:, 0])*1.2], [0, max(result[:, 0])*1.2], color="red")
    axs[1].set_xlabel("Ground truth")
    axs[1].set_ylabel("Predicted")
    axs[1].set_title("Upper bound")

    plt.show()
    """
    

