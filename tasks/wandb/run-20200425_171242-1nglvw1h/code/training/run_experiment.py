""" Script to run a single experiment. """

import argparse
import os
import sys
import wandb
import yaml
sys.path.append("../")

from training.util_training import run_experiment


def _parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int,
                        default=None, help="Index of GPU to use.")
    parser.add_argument("-c", "--config", type=str,
                        help="Experiment yaml file.")
    args = parser.parse_args()
    return args


def main():
    """ Run experiment. """
    args = _parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    run_experiment(cfg)


if __name__ == "__main__":
    main()