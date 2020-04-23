"""
Script to run a wandb sweep of hyperparameters.
Will unzip all parameters as command line arguments meaning
more argparse arguments are required.
"""

import argparse
import os
import sys
import yaml
sys.path.append("../")
sys.path.append("../../")  # If running from experiments/

from training.util_training import run_experiment


def _parse_args():
    """ Parse command-line arguments for sweeping. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="Index of GPU to use.")
    parser.add_argument("--comments", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_args_version", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--network", type=str)
    parser.add_argument("--network_args_n_channels", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--train_args_batch_size", type=int)
    parser.add_argument("--train_args_epochs", type=int)
    parser.add_argument("--train_args_learning_rate", type=float)
    args = parser.parse_args()
    return args


def main():
    """ Run sweep. """
    args = _parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    cfg = {
        "comments": args.comments,
        "dataset": args.dataset,
        "dataset_args": {
            "version": args.dataset_args_version
        },
        "loss": args.loss,
        "model": args.model,
        "name": args.name,
        "network": args.network,
        "network_args": {
            "n_channels": args.network_args_n_channels
        },
        "optimizer": args.optimizer,
        "train_args": {
            "batch_size": args.train_args_batch_size,
            "epochs": args.train_args_epochs,
            "learning_rate": args.train_args_learning_rate
        }
    }

    run_experiment(cfg)


if __name__ == "__main__":
    main()
