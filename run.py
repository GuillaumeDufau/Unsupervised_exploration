import argparse
import json
import os
from core.trainer import Trainer


def _get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="unsup_explo_maze.json", help="path to json config"
    )
    args = parser.parse_args()
    return args


def read_config_file(config_path):
    json_config = json.load(open(os.path.join("experiments/configs/", config_path), "r"))
    return json_config


if __name__ == "__main__":

    args = _get_parser_args()
    config = read_config_file(args.config)

    trainer = Trainer(config)
    trainer.train()
