from src.modes import Mode
from src.app import App
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AlphaZero",
        description="Script for training or evaluating a model using AlphaZero."
    )
    parser.add_argument('--config', '-c', metavar='CONFIG_FILE', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the configuration file. If not provided, uses the default config.')
    parser.add_argument('mode', metavar='MODE', choices=[Mode.TRAINING, Mode.EVALUATE],
                        help='Mode of operation: "training" or "evaluate"')

    return parser.parse_args()


LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%d-%b-%y %H:%M:%S',
    'filename': 'log/log.log',
    'filemode': 'w'
}

DEFAULT_CONFIG_PATH = "config/default.toml"


if __name__ == "__main__":
    logging.basicConfig(**LOGGING_CONFIG)
    logging.info("Starting run.py")

    args = parse_args()
    config_path = args.config

    logging.info(f"Using config file {config_path}")
    logging.info(f"Selected mode: {args.mode}")

    app = App(config_path, args.mode)
    app.run()