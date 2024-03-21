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


if __name__ == "__main__":
    data = generate_data(100, 12)
    print(f"Size of the data: {data.shape}")
    for i in tqdm.tqdm(range(data.shape[0])[:3]):
        g1 = data.iloc[i]["g1"]
        g2 = data.iloc[i]["g2"]

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        nx.draw(g1, ax=axs[0])
        nx.draw(g2, ax=axs[1])
        plt.show()

