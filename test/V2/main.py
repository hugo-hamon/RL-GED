from parser_manager import *
import pandas as pd
import argparse
import pickle


if __name__ == "__main__":
    # Make parser
    parser = argparse.ArgumentParser(
        description="Framework for graph edit distance using reinforcement learning"
    )

    # Required arguments
    parser.add_argument(
        "--generate_graphs",
        action="store_true",
        help="Generate graphs and save them to a file",
    )

    parser.add_argument(
        "--train",
        type=str,
        help="Train the model",
    )

    parser.add_argument(
        "--benchmark_model",
        type=str,
        help="Benchmark the model"
    )

    parser.add_argument(
        "--benchmark_mcts",
        type=str,
        help="Benchmark the MCTS algorithm"
    )

    parser.add_argument(
        "--mcts_ged",
        type=str,
        help="Calculate the GED using the MCTS algorithm"
    )

    # Manage parser arguments
    args = parser.parse_args()
    if args.train:
        manage_model_training(args.train)

    if args.generate_graphs:
        manage_graphs_generation()

    if args.benchmark_model:
        benchmark_model(args.benchmark_model)

    if args.benchmark_mcts:
        benchmark_mcts(args.benchmark_mcts)

    if args.mcts_ged:
        benchmark_mcts_ged(args.mcts_ged)