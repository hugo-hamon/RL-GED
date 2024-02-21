from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from .config import Config
from .model import ResNet
from tqdm import trange
from .mcts import MCTS
from .game import Game
import numpy as np
import logging
import random
import torch
import time
import ray


class AlphaZero:

    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, game: Game, config: Config, writer: SummaryWriter) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.config = config
        self.writer = writer
        self.mcts = MCTS(game, config, model)

    def train(self, memory) -> tuple:
        random.shuffle(memory)
        policy_losses = []
        value_losses = []
        for batchIdx in range(0, len(memory), self.config.model.batch_size):
            sample = memory[batchIdx:min(
                len(memory) - 1, batchIdx + self.config.model.batch_size)]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(
                policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            print(f"state: {state}")
            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return np.mean(policy_losses), np.mean(value_losses)
    
    def selfPlay(self) -> list:
        memory = []
        state = self.game.get_initial_state()
        iteration = 0
        while True:
            neutral_state = state

            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs))

            temperature_action = action_probs ** (1 / self.config.model.temperature)
            temperature_action /= np.sum(temperature_action)
            action = np.random.choice(self.game.action_size, p=temperature_action)

            state = self.game.get_next_state(state, action)

            value, is_terminal = self.game.get_value_and_terminated(state)

            if is_terminal or iteration > self.config.model.maximal_iterations:
                if iteration > self.config.model.maximal_iterations:
                    print("Maximal iterations reached")
                returnMemory = []
                for hist_neutral_state, hist_action_probs in memory:
                    hist_outcome = value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            iteration += 1

    def learn(self):
        for iteration in range(self.config.model.iterations):
            memory = []

            start_time = time.time()
            self.model.eval()
            
            results_ids = [selfPlay.remote(self.game, self.mcts, self.config) for _ in range(
                self.config.model.self_play_iterations)]
            results = ray.get(results_ids)
            for result in results:
                memory.extend(result)
            """
            for _ in range(self.config.model.self_play_iterations):
                memory.extend(self.selfPlay())
            """
            logging.info(
                f"Self-play took {time.time() - start_time} seconds"
            )

            self.model.train()
            mean_value_loss, mean_policy_loss = (0, 0)
            for _ in trange(self.config.model.epochs):
                policy_loss, value_loss = self.train(memory)
                mean_value_loss += value_loss
                mean_policy_loss += policy_loss
            mean_value_loss /= self.config.model.epochs
            mean_policy_loss /= self.config.model.epochs

            self.writer.add_scalar(
                'value_loss', mean_value_loss, iteration
            )
            self.writer.add_scalar(
                'policy_loss', mean_policy_loss, iteration
            )

            torch.save(self.model.state_dict(), f"model/model_{iteration}.pt")
            torch.save(
                self.optimizer.state_dict(), f"model/optimizer_{iteration}.pt"
            )


@ray.remote
def selfPlay(game: Game, mcts: MCTS, config: Config) -> list:
    memory = []
    state = game.get_initial_state()
    iteration = 0
    while True:
        neutral_state = state

        action_probs = mcts.search(neutral_state)

        memory.append((neutral_state, action_probs))

        temperature_action = action_probs ** (1 / config.model.temperature)
        temperature_action /= np.sum(temperature_action)
        action = np.random.choice(game.action_size, p=temperature_action)

        state = game.get_next_state(state, action)

        value, is_terminal = game.get_value_and_terminated(state)

        if is_terminal or iteration > config.model.maximal_iterations:
            if iteration > config.model.maximal_iterations:
                print("Maximal iterations reached")
            returnMemory = []
            for hist_neutral_state, hist_action_probs in memory:
                hist_outcome = value
                returnMemory.append((
                    game.get_encoded_state(hist_neutral_state),
                    hist_action_probs,
                    hist_outcome
                ))
            return returnMemory
        iteration += 1
