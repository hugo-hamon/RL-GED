from munkres import Munkres
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import torch
import tqdm
import ray


# --- DATA GENERATION ---
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


@ray.remote
def generate_interval_graph(max_size: int) -> nx.Graph:
    """Generate a random connected graph with random weights and size nodes."""
    size = np.random.randint(2, max_size)
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
    ray.init(logging_level=40)
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

    data = [compute_graph_information.remote(
        g1, g2) for g1, g2 in combinations
    ]
    results = ray.get(data)
    ray.shutdown()
    return pd.DataFrame(results, columns=["g1", "g2", "cost_matrix", "ged"])


def generate_data_interval(n: int, max_size: int, verbose: bool = False, max_ged_size: int = 10) -> pd.DataFrame:
    """
    Generate n random graphs and calculate the GED between all pairs.
    If the size of the graph is higher than max_ged_size, the GED is set to -1.
    """
    ray.init(logging_level=40)
    graphs = ray.get([
        generate_interval_graph.remote(max_size)for _ in range(n)
    ])

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

    data = [compute_graph_information.remote(
        g1, g2) for g1, g2 in combinations
    ]
    results = ray.get(data)
    ray.shutdown()
    return pd.DataFrame(results, columns=["g1", "g2", "cost_matrix", "ged"])


# --- DATA PROCESSING ---
def augmented_matrix(matrix: np.ndarray, new_size: int, g1: nx.Graph, g2: nx.Graph) -> np.ndarray:
    if matrix.shape[0] > new_size or matrix.shape[1] > new_size:
        raise ValueError("New size is smaller than the original matrix")
    n = len(g1.nodes)
    m = len(g2.nodes)
    new_matrix = np.zeros((new_size, new_size))
    new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    for i, j in itertools.product(range(new_size), range(new_size)):
        if i < n and j >= matrix.shape[0]:
            to_replace = matrix[i, -1]
            new_matrix[i, j] = to_replace
        elif i >= matrix.shape[1] and j < m:
            to_replace = matrix[-1, j]
            new_matrix[i, j] = to_replace

    return new_matrix


def process_matrix(matrix: np.ndarray, size: int) -> np.ndarray:
    """Process the matrix to be used as input for the model."""
    matrix = matrix.astype(np.float32)
    for i in range(size):
        total_value = matrix[i].sum()
        probs = [abs((value - total_value) ** 2) for value in matrix[i]]
        normalized_probs = [prob / sum(probs) for prob in probs]
        matrix[i] = np.round(normalized_probs, 2)
    return matrix


def solve_matrix(matrix: np.ndarray) -> np.ndarray:
    """Solve the matrix with the Hungarian algorithm."""
    m = Munkres()
    indexes = m.compute(matrix.copy().tolist())
    solution = np.zeros(matrix.shape)
    for row, column in indexes:
        solution[row][column] = 1
    return solution


def get_solutions(matrix: np.ndarray, solution: np.ndarray, size: int) -> tuple[list[np.ndarray], list[float]]:
    """Get all sub-solutions of a solution."""
    valid_solutions = [solution]
    valid_solutions_sum = [round(np.multiply(matrix, solution).sum(), 2)]
    # Remove successively the last one by row
    for k in range(size - 1):
        line = size - k - 1
        indexes = np.argwhere(valid_solutions[-1][line] == 1).flatten()
        if len(indexes) > 0:
            new_solution = valid_solutions[-1].copy()
            new_solution[line][indexes[0]] = 0
            valid_solutions.append(new_solution)
            valid_solutions_sum.append(
                round(np.multiply(matrix, new_solution).sum(), 2))
    return valid_solutions, valid_solutions_sum


def get_random_solution(size: int) -> np.ndarray:
    """Get a random solution."""
    random_permutation = np.random.permutation(size)
    solution = np.zeros((size, size))
    for k in range(size):
        solution[k][random_permutation[k]] = 1
    return solution


def generate_model_datas(datas: pd.DataFrame, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return processed data for the model"""
    states = []
    values = []

    for i in tqdm.tqdm(range(len(datas))):
        # Create a random process matrix
        random_matrix = np.array(datas.loc[i, "cost_matrix"])
        if random_matrix.shape[0] != size:
            random_matrix = augmented_matrix(
                random_matrix, size, datas.loc[i, "g1"], datas.loc[i, "g2"]
            )

        # Solve the problem with the Hungarian algorithm
        valide_solution = solve_matrix(random_matrix)

        # Get all sub-solutions of the valid solution and add them to the dataset
        valid_solutions, _ = get_solutions(
            random_matrix, valide_solution, size
        )

        for valid_solution in valid_solutions:
            state = np.stack([random_matrix, valid_solution], axis=0)
            states.append(torch.tensor(state, dtype=torch.float32))
            values.append(1)

        # Create a random solution
        solution = get_random_solution(size)

        # Get all sub-solutions of the random solution and add them to the dataset
        random_solutions, _ = get_solutions(random_matrix, solution, size)

        for (random_solution, valid_solution) in zip(random_solutions, valid_solutions):
            state = np.stack([random_matrix, random_solution], axis=0)
            states.append(torch.tensor(state, dtype=torch.float32))
            values.append(1 if np.array_equal(
                random_solution, valid_solution) else 0)

    return np.stack(states, axis=0), np.array(values)


# --- MCTS ---
def row_probability(row_value: np.ndarray, model_prediction: np.ndarray, model_accuracy: float, temperature: float) -> list[float]:
    def model_probability(value: float, prediction: float, accuracy: float) -> float:
        return value if prediction == 0 else value * (1 - accuracy)

    model_values = [model_probability(value, prediction, model_accuracy)
                    for value, prediction in zip(row_value, model_prediction)]
    total_model_values = sum(model_values)

    subtract_values = [total_model_values - value for value in model_values]
    total_subtract_values = sum(subtract_values)
    if total_subtract_values == 0:
        return [1 / len(model_values) for _ in model_values]
    probs = [(total_model_values - value)**temperature /
             total_subtract_values for value in model_values]
    return [prob / sum(probs) for prob in probs]


def get_one_hot_vector(model: torch.nn.Module, state: np.ndarray, depth: int, device: torch.device) -> np.ndarray:
    """Return the one hot vector of the untried actions."""
    untried_actions = get_untried_actions(state, depth)
    matrix = state[0]
    action_matrix = state[2]

    model_matrix = matrix # process_matrix(matrix, matrix.shape[0])
    one_hot_vector = np.zeros(matrix.shape[0])
    for action in untried_actions:
        new_state = action_matrix.copy()
        new_state[depth][action] = 1
        model_state = np.stack(
            [model_matrix, new_state.astype(np.int32)], axis=0
        )
        model_state = torch.tensor(
            model_state, dtype=torch.float32
        ).unsqueeze(0).to(device)
        model_value = model(model_state)
        one_hot_vector[action] = np.argmax(model_value.cpu().detach().numpy())
    return one_hot_vector


def get_untried_actions(state: np.ndarray, depth: int) -> np.ndarray:
    """Return the untried actions for the current state."""
    if depth >= state[1].shape[0]:
        return np.array([])
    return np.argwhere(state[1][depth] == 1).flatten()


def get_action_probs(state: np.ndarray, depth: int, model: torch.nn.Module, config: dict) -> np.ndarray:
    """Return the action probabilities for each action."""
    one_hot_vector = get_one_hot_vector(model, state, depth, config["device"])
    probabilities = row_probability(
        state[0][depth], one_hot_vector, 1, config["temperature"]
    )

    return np.array(probabilities)

def get_easy_action_probs(state: np.ndarray, depth: int, model: torch.nn.Module, config: dict) -> np.ndarray:
    """Return the action probabilities for each action."""
    untried_actions = get_untried_actions(state, depth)
    matrix = state[0]
    action_matrix = state[2]

    probs_vector = np.ones(matrix.shape[0])
    for action in untried_actions:
        new_state = action_matrix.copy()
        new_state[depth][action] = 1
        model_state = np.stack(
            [matrix, new_state.astype(np.int32)], axis=0
        )
        model_state = torch.tensor(
            model_state, dtype=torch.float32
        ).unsqueeze(0).to(config["device"])
        model_value = model(model_state).cpu().detach().numpy()[0]
        probs_vector[action] = np.abs(model_value[1] - 1)

    probs_vector = 1 - probs_vector
    return probs_vector / probs_vector.sum()


# --- UTILS ---
def get_matrix_level(matrix: np.ndarray) -> int:
    """Return the index of the last row with a 1"""
    return next(
        (i - 1 for i in range(matrix.shape[0]) if matrix[i].sum() != 1),
        matrix.shape[0] - 1
    )
