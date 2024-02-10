from torch.utils.tensorboard.writer import SummaryWriter
from .game import ConnectXGame, MunkresGame
from .alpha_zero import AlphaZero
from .config import load_config
from .model import ResNet
from .modes import Mode
from .mcts import MCTS
import numpy as np
import logging
import torch
import sys
import ray



class App:

    def __init__(self, config_path: str, mode: Mode) -> None:
        self.config = load_config(config_path)
        self.mode = mode

        matrix_size = 5
        self.game = MunkresGame(n=matrix_size, low=0, high=100)
        self.model = ResNet(
            game=self.game, input_size=matrix_size + 1, num_resBlocks=4, num_hidden=64
        )

    def run(self) -> None:
        """Run the app with the given mode."""
        if self.mode == Mode.TRAINING:
            self.train()
        elif self.mode == Mode.EVALUATE:
            self.evaluate()
        else:
            logging.error(f"Invalid mode {self.mode}")
            sys.exit(1)

    def train(self) -> None:
        """Train the model with the given config."""
        ray.init(num_cpus=10)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        writer = SummaryWriter('runs/')
        alpha_zero = AlphaZero(
            self.model, optimizer, self.game, self.config, writer
        )
        alpha_zero.learn()

    def evaluate(self) -> None:
        """Evaluate the model with the given config."""
        self.model.load_state_dict(torch.load(
            f'model/model_{self.config.model.iterations - 1}.pt'
        ))
        self.model.eval()

        mcts = MCTS(self.game, self.config, self.model)
        state = self.game.get_initial_state()

        print(f"Initial state:\n{state[0]}")

        while True:
            print("Valid moves: ", self.game.get_valid_moves(state))
            action_probs = mcts.search(state)
            action = int(np.argmax(action_probs))
            state = self.game.get_next_state(state, action)
            if self.game.get_value_and_terminated(state)[1]:
                reward = self.game.get_value_and_terminated(state)[0]
                print(f"Reward: {reward}")
                print(f"Last state: {state}")
                break
