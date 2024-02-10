from abc import ABC, abstractmethod
from munkres import Munkres
import numpy as np


class Game(ABC):

    def __init__(self) -> None:
        self.action_size = 0

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return a representation of the initial state of the game."""
        pass

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Return the state that results from taking action in state."""
        pass

    @abstractmethod
    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Return a list of the allowable moves at this point."""
        pass

    @abstractmethod
    def get_value_and_terminated(self, state: np.ndarray) -> tuple[int, bool]:
        """Return the value and whether the game has terminated after taking action in state."""
        pass

    @abstractmethod
    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """Return a tensor representing the state."""
        pass

    @abstractmethod
    def get_state_size(self) -> int:
        """Return the size of the state."""
        pass


class ConnectXGame(Game):

    def __init__(self, inarow: int = 4, width: int = 7) -> None:
        self.inarow = inarow
        self.width = width
        self.action_size = self.width

    def get_state_size(self) -> int:
        """Return the size of the state."""
        return self.width

    def get_initial_state(self):
        """Return a representation of the initial state of the game."""
        return np.zeros((self.width, 1))

    def get_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Return the state that results from taking action in state."""
        state[action] = 1
        return state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Return a list of the allowable moves at this point."""
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray) -> bool:
        """Check if the current player has won."""
        for i in range(self.width - self.inarow + 1):
            if np.all(state[i:i+self.inarow] == 1):
                return True
        return False

    def get_value_and_terminated(self, state: np.ndarray) -> tuple[int, bool]:
        """Return the value and whether the game has terminated after taking action in state."""
        if self.check_win(state):
            return 1, True
        elif np.count_nonzero(state) == self.inarow:
            return 0, True
        return 0, False

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """Return a tensor representing the state."""
        encoded_state = np.stack(
            (state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state


class MunkresGame(Game):

    def __init__(self, n: int = 5, low: int = 0, high: int = 100) -> None:
        super().__init__()
        self.n = n
        self.low = low
        self.high = high
        self.action_size = self.n**2

    def get_state_size(self) -> int:
        """Return the size of the state."""
        return self.n**2

    def get_initial_state(self) -> np.ndarray:
        """Return a representation of the initial state of the game."""
        np.random.seed(0)
        state = np.random.randint(self.low, self.high, (self.n, self.n))

        to_stack = [np.zeros((self.n, self.n)) for _ in range(self.n + 1)] # + 1 to track moves

        state = np.stack(
            (state, *to_stack)
        ).astype(np.float32)

        return state
    
    def get_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Return the state that results from taking action in state."""
        row = action // self.n
        col = action % self.n
        state[-1][0][0] += 1
        index = int(state[-1][0][0])
        state[index, row, col] = 1
        return state
    
    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Return a list of the allowable moves at this point."""
        valid_moves = np.zeros((self.n, self.n)).reshape(-1)

        # Concatenate all the matrices
        concat = np.zeros((self.n, self.n))

        for i in range(1, self.n + 1):
            concat += state[i]

        rows, cols = np.where(concat != 1)
        for row, col in zip(rows, cols):
            if 1 not in concat[row] and 1 not in concat[:, col]:
                valid_moves[row * self.n + col] = 1
        return valid_moves
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """Return whether the game has terminated."""
        return state[-1][0][0] == self.n
    
    def compute_score(self, state: np.ndarray) -> int:
        """Return the score of the state."""
        concat = np.zeros((self.n, self.n))
        for i in range(1, self.n + 1):
            concat += state[i]

        return int(np.sum(concat * state[0]))
    
    def compute_munkres_score(self, state: np.ndarray) -> int:
        """Return the score of the state."""
        M = Munkres()
        indexes = M.compute(state[0].tolist())
        zero_matrix = np.zeros((self.n, self.n))
        for row, col in indexes:
            zero_matrix[row, col] = 1
        return int(np.sum(zero_matrix * state[0]))
    
    def get_value_and_terminated(self, state: np.ndarray) -> tuple[int, bool]:
        """Return the value and whether the game has terminated after taking action in state."""
        if self.is_terminal(state):
            current_score = self.compute_score(state)
            munkres_score = self.compute_munkres_score(state)
            # compute = 1 - (current_score - munkres_score) / (current_score + munkres_score)
            return (1, True) if current_score == munkres_score else (0, True)
        return 0, False

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """Return a tensor representing the state."""
        return state[:self.n + 1]
    
if __name__ == '__main__':
    game = MunkresGame(n=4)
    print(game.get_initial_state())
    print(game.get_valid_moves(game.get_initial_state()))
    new_state = game.get_next_state(game.get_initial_state(), 0)
    new_state = game.get_next_state(new_state, 5)
    new_state = game.get_next_state(new_state, 10)
    new_state = game.get_next_state(new_state, 15)

    print(new_state)
    print(game.get_valid_moves(new_state).reshape(game.n, game.n))
    print(game.compute_score(new_state))
    print(game.compute_munkres_score(new_state))
    print(game.get_value_and_terminated(new_state))