import json
from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.yolov8 import sliced_yolov8_train_2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_config", type=Path, required=True)
    parser.add_argument("--assembly_config", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False)
    args = parser.parse_args()

    assert args.experiment_config.exists(), f"Experiment config {args.experiment_config} does not exist"
    with open(args.experiment_config) as json_file:
        experiment_config = json.load(json_file)

    assert args.assembly_config.exists(), f"Assembly config {args.assembly_config} does not exist"
    with open(args.assembly_config) as json_file:
        assembly_config = json.load(json_file)

    for experiment in experiment_config["experiments"]:
        print(experiment)
        experiment["model_config"]["project"] = (DatasetCreator().models_dir / experiment_config["dataset_assembly_id"]).as_posix()
        sliced_yolov8_train_2(
            assembly_configs=assembly_config,
            sliced_dataset_configs=experiment["sliced_datasets"],
            yolo_train_params=experiment["model_config"],
            device=args.device)