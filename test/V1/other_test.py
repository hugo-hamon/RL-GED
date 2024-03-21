import networkx as nx
import pandas as pd
import numpy as np
import itertools
import pickle
import tqdm
import time
import ray

COST = {
    'n_ins': 1, 'n_del': 1, 'n_sub': 1,
    'e_ins': 1, 'e_del': 1, 'e_sub': 1
}

# Initialize Ray
ray.init()


@ray.remote
def generate_graph(max_size: int) -> nx.Graph:
    size = np.random.randint(2, max_size)
    graph = nx.fast_gnp_random_graph(size, 0.5)
    while not nx.is_connected(graph):
        graph = nx.fast_gnp_random_graph(size, 0.5)
    # Add random weights to the nodes
    for node in graph.nodes:
        graph.nodes[node]["weight"] = np.random.randint(0, 10)
    return graph


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
    for i, j in itertools.product(range(m), range(m)):
        node_degree = g2.degree[i]
        insert_cost[i, j] = 1 + node_degree

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


@ray.remote
def calculate_ged(g1, g2, cost_function):
    cost_matrix = get_cost_matrix(g1, g2, cost_function)
    ged = nx.graph_edit_distance(g1, g2)
    return g1, g2, cost_matrix, ged


def generate_data_parallel(n: int, max_size: int, verbose: bool = False) -> pd.DataFrame:
    graphs = ray.get([generate_graph.remote(max_size) for _ in range(n)])

    combinations = itertools.combinations(graphs, 2)
    if verbose:
        combinations = tqdm.tqdm(combinations, total=n * (n - 1) // 2)
    data = [calculate_ged.remote(g1, g2, COST) for g1, g2 in combinations]
    results = ray.get(data)

    return pd.DataFrame(results, columns=["g1", "g2", "cost_matrix", "ged"])


# Example usage
data_size = 1_000
n = int((1 + np.sqrt(1 + 8 * data_size)) / 2)
print(f"Generating {n} graphs")

time1 = time.time()
result_df = generate_data_parallel(n=n, max_size=7, verbose=True)
print(f"Dataframe length: {len(result_df)}, time: {time.time() - time1}")

with open(f"processed/{n}_graphs.pkl", "wb") as file:
    pickle.dump(result_df, file)

ray.shutdown()
