"""Script to run a single experiment."""

import argparse
import yaml

from spots_predictor import SpotPredictor


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Experiment yaml file.")
    args = parser.parse_args()
    return args


def main():
    """Run evaluation."""
    args = _parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    predictor = SpotPredictor(cfg)
    f1_score, l2_norm = predictor.evaluate()

    dataset_args = cfg.get("dataset_args", {})
    name = dataset_args["version"]

    cfg["evaluation"] = {"f1_score": float(f1_score), "l2_norm": float(l2_norm)}
    with open(f"{name}_evaluation.yaml", "a") as file:
        yaml.dump(cfg, file)


if __name__ == "__main__":
    main()
