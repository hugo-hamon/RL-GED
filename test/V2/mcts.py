from __future__ import annotations
from tools import get_action_probs, get_untried_actions, get_easy_action_probs
from munkres import Munkres
from typing import Optional
from model import CNNModel
import networkx as nx
import numpy as np
import torch
import time


class Node:

    def __init__(self, state: np.ndarray, depth: int, prior: float, config: dict, parent=None, action=None) -> None:
        self.state = state
        self.depth = depth
        self.prior = prior
        self.config = config
        self.parent = parent
        self.action = action

        self.children: list[Node] = []

        self.visit_count = 0
        self.value = 0

    def is_expanded(self) -> bool:
        """Return True if the node has children"""
        return len(self.children) > 0

    def select(self) -> Optional[Node]:
        """Return the child node with the highest UCB1 value"""
        best_child = None
        best_value = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_value:
                best_child = child
                best_value = ucb

        return best_child

    def get_ucb(self, child: Node) -> float:
        """Return the UCB1 value of the child node"""
        if child.visit_count == 0:
            return np.inf

        exploitation = child.value / child.visit_count
        exploration = np.sqrt(np.log(self.visit_count) / child.visit_count)
        return exploitation + self.config["exploration_weight"] * child.prior * exploration

    def expand(self, action_probs: np.ndarray) -> None:
        """Expand the node by adding children"""
        untried_actions = get_untried_actions(self.state, self.depth)
        for action, prob in enumerate(action_probs):
            if prob > 0 and action in untried_actions:
                new_state = self.state.copy()
                new_state[1][self.depth] = 0
                new_state[1][:, action] = 0
                new_state[2][self.depth][action] = 1

                new_node = Node(new_state, self.depth + 1,
                                prob, self.config, self, action)
                self.children.append(new_node)

    def backpropagate(self, value: float) -> None:
        """Update the value and visit count of the node and its ancestors"""
        self.visit_count += 1
        self.value += value

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:

    def __init__(self, config: dict) -> None:
        self.config = config

    def search(self, root: Node, model: torch.nn.Module) -> int:
        """Run the MCTS search and return the action to take"""
        matrix = root.state[0]
        max_value = np.sum(np.max(matrix, axis=1))
        best_rollout = {"value": np.inf, "iteration": 0}

        for _ in range(self.config["num_simulations"]):
            node = self.select_node(root, model)

            match self.config["rollout_policy"]:
                case "random":
                    value = self.random_rollout(node, max_value)
                case "model":
                    value = self.model_rollout(node, model, max_value)
                case _:
                    raise ValueError("Invalid rollout policy")

            if value < best_rollout["value"]:
                best_rollout = {"value": value, "iteration": 0}
            else:
                best_rollout["iteration"] += 1
            node.backpropagate(value)
            if best_rollout["iteration"] >= self.config["stable_iterations"]:
                break

        best_child = max(root.children, key=lambda child: child.visit_count)
        if best_child.action is None:
            raise ValueError("No action selected")
        return best_child.action

    def select_node(self, node: Node, model: torch.nn.Module) -> Node:
        """Select a node to expand"""
        while node.is_expanded():
            select_node = node.select()
            if select_node is None:
                raise ValueError("Selecting a childless node")
            node = select_node

        if node.depth < node.state.shape[1]:
            match self.config["expansion_policy"]:
                case "random":
                    action_probs = np.ones(node.state.shape[1])
                case "model":
                    action_probs = get_easy_action_probs(
                        node.state, node.depth, model, self.config
                    )
                case _:
                    raise ValueError("Invalid expansion policy")

            node.expand(action_probs)
        return node

    def random_rollout(self, node: Node, max_value: int) -> float:
        """Simulate a rollout from the node"""
        state = node.state.copy()
        depth = node.depth

        while depth < state.shape[1]:
            untried_actions = get_untried_actions(state, depth)
            if len(untried_actions) == 0:
                break

            action = np.random.choice(untried_actions)
            state[1][depth] = 0
            state[1][:, action] = 0
            state[2][depth][action] = 1
            depth += 1

        return (max_value - np.multiply(state[0], state[2]).sum()) / max_value
    
    def model_rollout(self, node: Node, model: torch.nn.Module, max_value: int) -> float:
        """Simulate a rollout from the node using the model"""
        state = node.state.copy()
        depth = node.depth

        while depth < state.shape[1]:
            action = get_easy_action_probs(state, depth, model, self.config)
            state[1][depth] = 0
            state[1][:, action] = 0
            state[2][depth][action] = 1
            depth += 1

        return (max_value - np.multiply(state[0], state[2]).sum()) / max_value


def get_mcts_cost(matrix: np.ndarray, model: torch.nn.Module, config: dict, max_depth: int) -> tuple[list[int], float]:
    """Return the MCTS cost of the matrix"""
    state = np.stack(
        [matrix, np.ones(matrix.shape), np.zeros(matrix.shape)], axis=0
    )
    actions = []
    for i in range(max_depth):
        root = Node(state, i, 1, config)
        mcts = MCTS(config)
        action = mcts.search(root, model)
        actions.append(action)
        state[1][i] = 0
        state[1][:, action] = 0
        state[2][i][action] = 1

    return actions, sum(matrix[i][action] for i, action in enumerate(actions))


if __name__ == "__main__":
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "exploration_weight": 2,
        "num_simulations": 1000,
        "device": device,
        "temperature": 2,
        "stable_iterations": 100,
        "rollout_policy": "random",
        "expansion_policy": "random"
    }

    matrix_size = 12

    model = CNNModel(
        state_size=matrix_size**2,
        input_size=2,
        num_hidden=64
    ).to(device)

    model.load_state_dict(torch.load(
        "models/cnn_model_12.pth", map_location=device
    ))

    matrix = np.random.randint(1, 20, (matrix_size, matrix_size))
    print(f"Matrix:\n{matrix}")
    
    start_time = time.time()
    actions, value = get_mcts_cost(matrix, model, config, matrix_size)
    print(f"Time: {time.time() - start_time:.2f}s")
    print(f"MCTS value: {value}")
    print(f"MCTS actions: {actions}")

    munkres = Munkres()
    indexes = munkres.compute(matrix.tolist())
    cost = sum(matrix[i][j] for i, j in indexes)
    print(f"LSAP value: {cost}")
    print(f"LSAP indexes: {indexes}")
