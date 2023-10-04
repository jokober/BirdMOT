import argparse
import json
from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.detection.evaluate import EvaluationController

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_config", type=Path, required=True)
    parser.add_argument("--assembly_config", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    parser.add_argument("--train_missing",  default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite_existing",  default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    eval_controller = EvaluationController()

    experiments_config_path = args.experiment_config.resolve()
    assert experiments_config_path, f"Experiment config {experiments_config_path} does not exist"
    with open(experiments_config_path) as json_file:
        experiment_config = json.load(json_file)

    assembly_path = args.assembly_config.resolve()
    assert assembly_path.exists(), f"Assembly config {assembly_path} does not exist"
    with open(assembly_path) as json_file:
        assembly_config = json.load(json_file)

    for one_experiment_config in experiment_config['experiments']:
        returned_prediction = eval_controller.find_or_create_evaluation(one_experiment_config, assembly_config,
                                                                        device=args.device,
                                                                        train_missing=args.train_missing,
                                                                        overwrite_existing=args.overwrite_existing)
