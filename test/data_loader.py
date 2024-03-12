import networkx as nx
import pandas as pd
import numpy as np
import itertools
import json
import tqdm

COST = {
    'n_ins': 4, 'n_del': 4, 'n_sub': 2,
    'e_ins': 1, 'e_del': 1, 'e_sub': 1
}


def load_data(data_path: str, ged_path: str) -> pd.DataFrame:
    """
    Load the data from the given paths and return a dataframe.

    Args:
        train_path (str): The path to the data.
        ged_path (str): The path to the ground truth GED.

    Returns:
        pd.DataFrame: The dataframe containing for each row the two graphs, the cost matrix and the GED.
    """
    # Load graphs from the json
    with open(data_path, "r") as file:
        datas = json.load(file)
    

    names = datas["nom"]
    nodes = datas["nodes"]
    edges = datas["edges"]
    edges_label = datas["e_label"]

    # Create graphs objects
    graphs = {}
    for name in list(names.keys())[:100]:
        G = nx.Graph()
        for i, node in enumerate(nodes[name]):
            G.add_node(i, weight=node)
        for edge in edges[name]:
            for e in edge:
                G.add_edge(edge[0], edge[1], weight=edges_label[name][e])
        graphs[name] = G

    # Load the ground truth GED
    ged = pd.read_csv(ged_path, names=["g1", "g2", "ged"], dtype={"g1": str, "g2": str, "ged": float}, header=0)
    
    # Create the dataframe
    data = []
    for _, row in tqdm.tqdm(ged.iterrows(), total=ged.shape[0]):
        if row["g1"] not in graphs or row["g2"] not in graphs:
            continue
        g1 = graphs[row["g1"]]
        g2 = graphs[row["g2"]]
        cost_matrix = get_cost_matrix(g1, g2, COST)
        data.append([g1, g2, cost_matrix, row["ged"]])

    return pd.DataFrame(data, columns=["g1", "g2", "cost_matrix", "ged"])


def get_cost_matrix(g1: nx.Graph, g2: nx.Graph, cost: dict) -> np.ndarray:
    """
    Create the cost matrix between the two graphs.

    Args:
        g1 (nx.Graph): The first graph.
        g2 (nx.Graph): The second graph.
        cost (dict): The cost dictionary.

    Returns:
        pd.DataFrame: The cost matrix.
    """
    n = len(g1.nodes)
    m = len(g2.nodes)

    # Create the cost matrix
    degree_cost = np.zeros((n, m))
    for i, node_1 in enumerate(g1.nodes):
        for j, node_2 in enumerate(g2.nodes):
            degree_cost[i, j] = abs(g1.degree[node_1] - g2.degree[node_2])

    # Create the delete cost matrix
    delete_cost = np.zeros((n, n))
    for i, node_1 in enumerate(g1.nodes):
        delete_cost[i, :] = cost["n_del"]

    # Create the insert cost matrix
    insert_cost = np.zeros((m, m))
    for j, node_2 in enumerate(g2.nodes):
        insert_cost[:, j] = cost["n_ins"]

    # Create the substitution cost matrix
    sub_cost = np.zeros((m, n))

    # Create the cost matrix
    cost_matrix = np.zeros((n + m, n + m))
    for i, j in itertools.product(range(n + m), range(n + m)):
        if i < n and j < m:
            cost_matrix[i, j] = degree_cost[i, j]
        elif i < n:
            cost_matrix[i, j] = delete_cost[i, j - m - 1]
        elif j < m:
            cost_matrix[i, j] = insert_cost[i - n - 1, j]
        else:
            cost_matrix[i, j] = sub_cost[i - n - 1, j - m - 1]

    return cost_matrix
