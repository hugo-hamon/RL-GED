import networkx as nx
import pandas as pd
import numpy as np
import itertools
import tqdm
import ray


@ray.remote
def generate_graph(size: int) -> nx.Graph:
    """Generate a random connected graph with random weights and size nodes."""
    graph = nx.fast_gnp_random_graph(size, 0.5)
    while not nx.is_connected(graph):
        graph = nx.fast_gnp_random_graph(size, 0.5)
    # Add random weights to the nodes
    for node in graph.nodes:
        graph.nodes[node]["weight"] = np.random.randint(0, 10)
    return graph

def get_cost_matrix(g1: nx.Graph, g2: nx.Graph) -> np.ndarray:
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


def generate_data(n: int, size: int, verbose: bool = False, max_ged_size: int = 10) -> pd.DataFrame:
    """
    Generate n random graphs and calculate the GED between all pairs.
    If the size of the graph is higher than max_ged_size, the GED is set to -1.
    """
    graphs = ray.get([generate_graph.remote(size) for _ in range(n)])

    combinations = itertools.combinations(graphs, 2)
    if verbose:
        combinations = tqdm.tqdm(combinations, total=n * (n - 1) // 2)

    @ray.remote
    def compute_graph_information(g1: nx.Graph, g2: nx.Graph):
        cost_matrix = get_cost_matrix(g1, g2)
        if len(g1.nodes) > max_ged_size or len(g2.nodes) > max_ged_size:
            ged = -1
        else:
            ged = nx.graph_edit_distance(g1, g2)
        return g1, g2, cost_matrix, ged

    data = [compute_graph_information.remote(g1, g2) for g1, g2 in combinations]
    results = ray.get(data)

    return pd.DataFrame(results, columns=["g1", "g2", "cost_matrix", "ged"])