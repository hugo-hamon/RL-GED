from munkres import Munkres
import numpy as np
import torch
import time


LOWER_BOUND = 1
UPPER_BOUND = 100


def process_matrix(matrix: np.ndarray, size: int) -> np.ndarray:
    """Process the matrix to be used as input for the model."""
    matrix = matrix.astype(np.float32)
    for i in range(size):
        total_value = matrix[i].sum()
        probs = [abs((value - total_value) ** 2) for value in matrix[i]]
        normalized_probs = [prob / sum(probs) for prob in probs]
        matrix[i] = np.round(normalized_probs, 2)
    return matrix


def get_random_process_matrix(size: int) -> np.ndarray:
    """Get a random matrix and process it."""
    random_matrix = np.random.randint(
        LOWER_BOUND, UPPER_BOUND, (size, size)
    ).astype(np.float32)
    return process_matrix(random_matrix, size)


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
            valid_solutions_sum.append(round(np.multiply(matrix, new_solution).sum(), 2))
    return valid_solutions, valid_solutions_sum


def get_random_solution(size: int) -> np.ndarray:
    """Get a random solution."""
    random_permutation = np.random.permutation(size)
    solution = np.zeros((size, size))
    for k in range(size):
        solution[k][random_permutation[k]] = 1
    return solution


def generate_datas(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    states = []
    values = []

    for _ in range(n):
        # Create a random process matrix
        random_matrix = get_random_process_matrix(size)

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
            values.append(1 if np.array_equal(random_solution, valid_solution) else 0)

    return np.stack(states, axis=0), np.array(values)


if __name__ == "__main__":
    # Paramètres
    n = 10000  # Number of samples
    size = 2  # Size of the matrix

    # Génération des données
    start = time.time()
    states, values = generate_datas(n, size)
    print(states[0].shape)
    print(
        f"Temps de generation des donnees : {round(time.time() - start, 2)}s")

    # Affichage de la répartition des valeurs
    print(f"Nombre de valeurs 1 : {np.sum(values)}")
    print(f"Nombre de valeurs 0 : {len(values) - np.sum(values)}")

    total = 0
    last = n
    for i in range(size, 1, -1):
        total += last * (1 / i)
        last = last * (1 / i)

    print(f"Nombre de combinaisons valides tirées aléatoirement : {total}")
