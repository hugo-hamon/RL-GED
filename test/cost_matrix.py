import matplotlib.pyplot as plt
from munkres import Munkres
from icecream import ic
import networkx as nx
import pandas as pd
import numpy as np
import json


def cost_path(g1, g2, path, cost):
    p_cost = 0
    # node cost
    nodes = list(path)
    for node in nodes:
        if node[0] == node[1]:
            continue
        if node[0] == -1:
            p_cost += cost["n_ins"]
        elif node[1] == -1:
            p_cost += cost["n_del"]
        else:
            p_cost += cost["n_sub"]

    # edge cost
    substitution = {i: j for i, j in nodes if i != -1 and j != -1}
    g1_edges_prime = {
        (substitution[i], substitution[j], k)
        for (i, j, k) in {(i, j, k["weight"]) for (i, j, k) in g1.edges(data=True)}
        if i in substitution and j in substitution
    }

    g2_edges = {(i, j, k["weight"]) for (i, j, k) in g2.edges(data=True)}

    g_diff = {
        (i, j): k
        for i, j, k in g1_edges_prime
        if (i, j) not in {(i, j) for (i, j, _) in g2_edges}
    }
    for i, j, k in g2_edges:
        if (i, j) not in {(i, j) for (i, j, _) in g1_edges_prime}:
            g_diff[(i, j)] = k
    # insert / delete cost
    p_cost += cost["e_ins"] * len(g_diff)
    g_inter = {
        (i, j): k
        for i, j, k in g1_edges_prime
        if (i, j) in {(i, j) for (i, j, _) in g2_edges}
    }
    for i, j, k in g2_edges:
        if (i, j) in {(i, j) for (i, j, _) in g1_edges_prime}:
            g_inter[(i, j)] = k
    ic(g_inter)

    return p_cost


def get_bound(g1, g2, cost_dict) -> tuple[float, float]:
    """Return the lower and upper bound of the ged between g1 and g2."""
    n = len(g1.nodes)
    m = len(g2.nodes)

    # Create the cost matrix using degree difference
    degree_cost = np.zeros((n, m))
    for i, node_1 in enumerate(g1.nodes):
        for j, node_2 in enumerate(g2.nodes):
            degree_cost[i, j] = abs(g1.degree[node_1] - g2.degree[node_2])

    # Create the delete and insert cost matrix
    delete_cost = np.zeros((n, m))
    for i, node_1 in enumerate(g1.nodes):
        for j, node_2 in enumerate(g2.nodes):
            delete_cost[i, j] = cost_dict["n_del"] if i == j else np.inf

    insert_cost = np.zeros((m, n))
    for i, node_1 in enumerate(g1.nodes):
        for j, node_2 in enumerate(g2.nodes):
            insert_cost[j, i] = cost_dict["n_ins"] if i == j else np.inf

    # Create the substitution cost matrix
    sub_cost = np.zeros((m, n))

    # Create the cost matrix
    cost_matrix = np.zeros((n + m, n + m))
    cost_matrix[:n, :m] = degree_cost
    cost_matrix[n:, :m] = insert_cost
    cost_matrix[:n, m:] = delete_cost
    cost_matrix[n:, m:] = sub_cost

    # Apply the Hungarian algorithm to find the best matching
    m = Munkres()
    indexes = m.compute(cost_matrix.tolist())[:-11]
    print(indexes)
    cost = 0

    for row, column in indexes:
        value = cost_matrix[row][column]
        cost += value

    cost_path_compute = cost_path(g1, g2, indexes, cost_dict)
    return cost, cost_path_compute


TRAIN_PATH = "../../Dataset/AIDS_train/AIDS_train.json"
TEST_PATH = "../../Dataset/AIDS_test/AIDS_test.json"
CSV_PATH = "../../Dataset/AIDS_csv/AIDS.csv"

if __name__ == "__main__":
    # Load the dataset
    with open(TRAIN_PATH, "r") as file:
        datas = json.load(file)

    """json:
    'nom': {'272': 272, '181': 181, '492': 492, '340': 340, ...}
    'nodes': {'272': [2, 2, ...], '181': [2, 3, ...], ...}
    'edges': {'272': [[0, 9], [1, 7], ...], '181': [[0, 3], [0, 6], ...], ...}
    'e_label': {'272': [0, 0, ...], '181': [0, 0, ...], ...}
    """

    names = datas["nom"]
    nodes = datas["nodes"]
    edges = datas["edges"]
    e_label = datas["e_label"]

    # Create graphs from the json
    graphs = []
    for name in names:
        G = nx.Graph()
        for i, node in enumerate(nodes[name]):
            G.add_node(i, weight=node)
        for edge in edges[name]:
            for e in edge:
                G.add_edge(edge[0], edge[1], weight=e_label[name][e])
        graphs.append(G)

    # Create the cost matrix between the first graph and the second graph
    cost1 = {'n_ins': 4, 'n_sub': 2, 'n_del': 4,
             'e_ins': 1, 'e_sub': 1, 'e_del': 1}
    graph_1 = graphs[0]
    graph_2 = graphs[3]

    print(f'Lower and upper bound: {get_bound(graph_1, graph_2, cost1)}')

    # Load the csv file
    df = pd.read_csv(CSV_PATH)
    print(df.head())
