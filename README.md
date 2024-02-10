# RL-GED

## Note: Work in Progress

This project is currently under development and is not yet fully functional.

## Description

This project aims to implement a Reinforcement Learning algorithm for the Graph Edit Distance (GED) problem. The algorithm is based on the AlphaZero approach, which combines Monte Carlo Tree Search with a deep neural network. The project is developed in Python and utilizes the PyTorch library for the neural network.

## Usage

### Prerequisites

Before running the script, ensure you have **Python 3.10** and the necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Running the Script

To use the script, follow these steps:

1. Clone the repository:

```bash
https://github.com/hugo-hamon/RL-GED.git
```

2. Navigate to the project directory:

```bash
cd RL-GED/project
```

3. Run the script with the desired mode:

```bash
python run.py --config CONFIG_NAME MODE
```

Replace `CONFIG_NAME` with the name of the configuration file if you want to use a custom configuration. Replace `MODE` with either `training` or `evaluate` depending on the desired mode of operation.

## Project Structure

- `.gitignore`: Specifies files and directories to ignore in version control.
- `README.md`: Provides an overview of the project and instructions for use.
- `LICENSE`: The license for the project.
- `requirements.txt`: The list of Python dependencies required to run the project.
- `project/`: Main project directory.
  - `config/`: Configuration files, including `default.toml`.
  - `log/`: Log files.
  - `model/`: Model save files.
  - `runs/`: Tensorboard files.
  - `src/`: Source files.
  - `run.py`: Main script for training or evaluating a model

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
