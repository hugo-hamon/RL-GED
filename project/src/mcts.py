from __future__ import annotations
from typing import Optional
from .config import Config
from .model import ResNet
from .game import Game
import numpy as np
import torch
import math


class Node:

    def __init__(self, game: Game, config: Config, state: np.ndarray, parent=None, action_taken=None, prior=0) -> None:
        self.game = game
        self.config = config
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children: list[Node] = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self) -> bool:
        """Return whether all children have been expanded."""
        return len(self.children) > 0

    def select(self) -> Optional[Node]:
        """Return the child with the highest UCB score."""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: Node) -> float:
        q_value = 0 if child.visit_count == 0 else 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.config.mcts.exploration_constant * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy: np.ndarray) -> Optional[Node]:
        """Expand the node by creating all possible children."""
        child = None
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child = Node(
                    self.game, self.config, child_state, self, action, prob
                )
                self.children.append(child)

        return child

    def backpropagate(self, value: float) -> None:
        """Update the node's value and visit count."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:

    def __init__(self, game: Game, config: Config, model: ResNet) -> None:
        self.game = game
        self.config = config
        self.model = model

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
        root = Node(self.game, self.config, state)
        """
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state)).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - 0.25) * policy + 0.25 \
            * np.random.dirichlet([0.3] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        """
        
        for _ in range(self.config.mcts.simulations_per_iteration):
            node = root

            while node.is_fully_expanded():
                node = node.select()
                if node is None:
                    raise Exception('Node is None')

            value, is_terminal = self.game.get_value_and_terminated(node.state)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state)
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy_sum = np.sum(policy)
    
                policy /= policy_sum

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
