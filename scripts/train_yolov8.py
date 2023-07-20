import json
from argparse import ArgumentParser
from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.detection.yolov8 import sliced_yolov8_train, Yolov8TrainParams

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_coco_path", type=Path, required=True)
    parser.add_argument("--val_coco_path", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--experiment_config", type=Path, required=False)
    args = parser.parse_args()

    #yolov8_train_params = Yolov8TrainParams(
    #    epochs=5,
    #    batch=30,
    #    model = 'YOLOv8n',
    #    patience =10,
    #    name = "pre_experiment",
    #    project = "BirdMOT Yolov8",
    #    device= "0",
    #)

    yolov8_train_params_list =[
        Yolov8TrainParams(
            epochs=400,
            batch=30,
            model = 'YOLOv8x',
            patience =50,
            name = "pre_experiment",
            project = DatasetCreator().models_dir,
            device= "0",
            imgsz=640
        ),
        #     Yolov8TrainParams(
        #     epochs=150,
        #     batch=15,
        #     model = 'YOLOv8s',
        #     patience =50,
        #     name = "pre_experiment",
        #     project = "BirdMOT Yolov8",
        #     device= "0",
        # ),
        #     Yolov8TrainParams(
        #     epochs=150,
        #     batch=10,
        #     model = 'YOLOv8m',
        #     patience =50,
        #     name = "pre_experiment",
        #     project = "BirdMOT Yolov8",
        #     device= "0",
        # ),
        #     Yolov8TrainParams(
        #     epochs=150,
        #     batch=5,
        #     model = 'YOLOv8l',
        #     patience =50,
        #     name = "pre_experiment",
        #     project = "BirdMOT Yolov8",
        #     device= "0",
        # ),
        #     Yolov8TrainParams(
        #     epochs=150,
        #     batch=3,
        #     model = 'YOLOv8x',
        #     patience =50,
        #     name = "pre_experiment",
        #     project = "BirdMOT Yolov8 no_hyp",
        #     device= "0",
        # ),

    ]

    slice_params = SliceParams(
        height=640,
        width=640,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        min_area_ratio=0.2,
        ignore_negative_samples=True
    )

    if args.experiment_config:
        assert args.experiment_config.exists(), f"Experiment config {args.experiment_config} does not exist"
        with open(args.experiment_config) as json_file:
            experiment_config = json.load(json_file)

        for experiment in experiment_config["experiments"]:
            print(experiment)
            experiment["project"] = DatasetCreator().models_dir.as_posix()
            sliced_yolov8_train(Yolov8TrainParams(**experiment["model_config"]), SliceParams(**experiment["train_slice_params"]), args.train_coco_path, args.val_coco_path, args.image_path, device='0')

    else:
        for yolov8_train_params in yolov8_train_params_list:
            yolov8_train_params.name = f'{yolov8_train_params.model} no_hyp'
            sliced_yolov8_train(yolov8_train_params, slice_params, Path(args.coco_annotation_file_path), Path(args.image_path), device='0')