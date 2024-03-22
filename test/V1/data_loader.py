import networkx as nx
import pandas as pd
import numpy as np
import itertools
import json
import tqdm
from itertools import permutations

COST = {
    'n_ins': 1, 'n_del': 1, 'n_sub': 1,
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


def generate_data(size: int, max_nodes: int) -> pd.DataFrame:
    """Generate random data."""

    graphs = []
    for _ in range(size):
        random_size = np.random.randint(5, max_nodes // 2)
        # Random graph with random connections
        G = nx.fast_gnp_random_graph(random_size, 0.5)
        # Check if the graph is connected
        while not nx.is_connected(G):
            # get the connected components
            components = list(nx.connected_components(G))
            # connect the components
            G.add_edge(components[0].pop(), components[1].pop())
        graphs.append(G)
        

    data = []
    for g1, g2 in permutations(graphs, 2):
        cost_matrix = get_cost_matrix(g1, g2, COST)
        # ged = nx.graph_edit_distance(g1, g2)
        ged = 0
        data.append([g1, g2, cost_matrix, ged])

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
            degree = abs(g1.degree[node_1] - g2.degree[node_2])
            cost_sub = 0 if g1.nodes[node_1]["weight"] == g2.nodes[node_2]["weight"] else 1
            degree_cost[i, j] = cost_sub + degree

    # Create the delete cost matrix
    delete_cost = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        node_degree = g1.degree[i]
        delete_cost[i, j] = 1 + node_degree

    # Create the insert cost matrix
    insert_cost = np.zeros((m, m))
    for j in range(m):
        node_degree = g2.degree[j]
        insert_cost[:, j] = 1 + node_degree

    # Create the substitution cost matrix
    sub_cost = np.zeros((m, n))

    # Create the cost matrix
    cost_matrix = np.zeros((n + m, n + m))
    for i, j in itertools.product(range(n + m), range(n + m)):
        if i < n and j < m:
            cost_matrix[i, j] = degree_cost[i, j]
        elif i < n:
            cost_matrix[i, j] = delete_cost[i, j - m]
        elif j < m:
            cost_matrix[i, j] = insert_cost[i - n, j]
        else:
            cost_matrix[i, j] = sub_cost[i - n, j - m]

    return cost_matrix


def generate_data(n: int, max_size: int, verbose: bool = False) -> pd.DataFrame:
    """
    Generate n random graphs and their GED.

    Args:
        n (int): The number of graphs to generate.
        max_size (int): The maximum size of the graphs.

    Returns:
        pd.DataFrame: The dataframe containing for each row the two graphs, the cost matrix and the GED.
    """
    graphs = []
    for _ in range(n):
        size = np.random.randint(2, max_size)
        graph = nx.fast_gnp_random_graph(size, 0.5)
        while not nx.is_connected(graph):
            graph = nx.fast_gnp_random_graph(size, 0.5)
        # Add random weights to the nodes
        for node in graph.nodes:
            graph.nodes[node]["weight"] = np.random.randint(0, 10)

        graphs.append(graph)

    data = []

    combinations = itertools.combinations(graphs, 2)
    if verbose:
        combinations = tqdm.tqdm(combinations, total=n * (n - 1) // 2)
    for g1, g2 in combinations:
        cost_matrix = get_cost_matrix(g1, g2, COST)
        ged = nx.graph_edit_distance(g1, g2)
        data.append([g1, g2, cost_matrix, ged])

    return pd.DataFrame(data, columns=["g1", "g2", "cost_matrix", "ged"])
