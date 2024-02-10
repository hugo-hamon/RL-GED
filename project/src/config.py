from dataclasses import dataclass
from dacite.core import from_dict
import toml


@dataclass
class MCTSConfig:
    exploration_constant: float
    simulations_per_iteration: int


@dataclass
class ModelConfig:
    iterations: int
    self_play_iterations: int
    epochs: int
    batch_size: int
    maximal_iterations: int
    temperature: float


@dataclass
class Config:
    mcts: MCTSConfig
    model: ModelConfig


def load_config(config_path: str) -> Config:
    """Load the config from a file."""
    return from_dict(data_class=Config, data=toml.load(config_path))
