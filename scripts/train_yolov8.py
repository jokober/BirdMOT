import json
from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.data.dataset_tools import rapair_absolute_image_paths
from BirdMOT.detection.yolov8 import sliced_yolov8_train, Yolov8TrainParams

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_coco_path", type=Path, required=True)
    parser.add_argument("--val_coco_path", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--experiment_config", type=Path, required=False)
    parser.add_argument("--device", type=str, required=False)
    args = parser.parse_args()


    if args.experiment_config:
        assert args.experiment_config.exists(), f"Experiment config {args.experiment_config} does not exist"
        with open(args.experiment_config) as json_file:
            experiment_config = json.load(json_file)

        for experiment in experiment_config["experiments"]:
            print(experiment)
            experiment["model_config"]["project"] = (DatasetCreator().models_dir / experiment_config["dataset_assembly_id"]).as_posix()
            rapair_absolute_image_paths(args.train_coco_path, args.image_path, overwrite_file=True)
            sliced_yolov8_train(Yolov8TrainParams(**experiment["model_config"]), SliceParams(**experiment["train_slice_params"]), args.train_coco_path, args.val_coco_path, args.image_path, device=args.device)