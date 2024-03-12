import matplotlib.pyplot as plt
from munkres import Munkres
from data_loader import *
import seaborn as sns
import numpy as np
import pickle
import os

TRAIN_PATH = "../../Dataset/AIDS_train/AIDS_train.json"
TEST_PATH = "../../Dataset/AIDS_test/AIDS_test.json"
GED_PATH = "../../Dataset/AIDS_csv/AIDS.csv"

if __name__ == "__main__":
    """
    data = load_data(TRAIN_PATH, GED_PATH)
    
    # Save the data with pickle
    if not os.path.exists("processed"):
        os.makedirs("processed")

    with open("processed/train.pkl", "wb") as file:
        pickle.dump(data, file)
    """

    # Load the train with pickle
    with open("processed/train.pkl", "rb") as file:
        data = pickle.load(file)

    # print the first row
    first_row = data.head(1)
    cost_matrix = np.array(first_row["cost_matrix"])
    # Series to numpy array
    cost_matrix = cost_matrix[0]
    
    # Use the munkres algorithm to solve the cost matrix
    m = Munkres()
    indexes = m.compute(cost_matrix.tolist())
    cost = 0
    for row, column in indexes:
        value = cost_matrix[row][column]
        cost += value
    print(cost)
    print(first_row["ged"])

    result = []
    print(data.shape[0])
    for i in range(data.shape[0]):
        data_row = data.iloc[i]
        cost_matrix = np.array(data_row["cost_matrix"])
        # Compute the cost with the munkres algorithm
        m = Munkres()
        indexes = m.compute(cost_matrix.tolist())
        cost = 0
        for row, column in indexes:
            value = cost_matrix[row][column]
            cost += value
        result.append((cost, data_row["ged"]))

    # Compute the MSE
    result = np.array(result)
    mse = np.mean((result[:, 0] - result[:, 1]) ** 2)

    # Plot the result
    sns.set_theme()
    plt.scatter(result[:, 1], result[:, 0], c=(result[:, 0] - result[:, 1]) ** 2, cmap="viridis")
    plt.colorbar()
    plt.plot([0, max(result[:, 1])*1.5], [0, max(result[:, 1])*1.5], color="red")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.title(f"MSE: {mse}")
    plt.show()
    

