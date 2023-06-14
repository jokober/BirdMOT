from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import mlflow
from sahi.predict import predict
from sahi.prediction import PredictionResult
from ultralytics import YOLO

from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.data.dataset_tools import coco2yolov5, create_sliced_dataset
from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams

YOLOV8_PRETRAINED_MODELS = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

@dataclass
class Yolov8TrainParams:
    """
    Check out more possible configuration parameterts here:
    https://docs.ultralytics.com/usage/cfg/#predict

    Attributes:
        epochs:int
            number of epochs to train
        batch:int
            number of images per batch (-1 for AutoBatch)
        model:str
            path to model file, i.e. yolov8n.pt, yolov8n.yaml
        patience:int
            epochs to wait for no observable improvement for early stopping of training
        name:str
            nam of the experiment
        project:str
            name of the project
        device:str
            device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    """
    epochs: int
    batch: int
    model: str
    patience: int
    name: str
    project: str
    device: str





def train_yolov8(model: str, yolo_data_path: Path, yolo_train_params: Yolov8TrainParams):
    # Load a model
    model = YOLO(model)

    # Use the model
    model.train(data=yolo_data_path.as_posix(), **asdict(yolo_train_params))  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg", yolo_train_params.device)  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format





def sliced_yolov8_train(yolov8_params: Union[dict, Yolov8TrainParams], slice_params: Union[dict, SliceParams],
                        coco_annotation_file_path: Path, image_dir: Path, device: str = 'cpu'):
    if type(yolov8_params) == dict:
        yolov8_params = Yolov8TrainParams(**yolov8_params)

    if type(slice_params) == dict:
        slice_params = SliceParams(**slice_params)

    if yolov8_params.model in YOLOV8_PRETRAINED_MODELS:
        slice_params.slice_width = 640
        slice_params.slice_height = 640

    dataset_path = create_sliced_dataset(coco_annotation_file_path=coco_annotation_file_path, image_dir=image_dir,
                                         slice_params=slice_params)

    coco2yolov5(dataset_path=dataset_path, coco_images_dir=dataset_path / "images")
    train_yolov8("yolov8n.pt", (dataset_path / "yolov5_files" / "data.yaml"), yolo_train_params=yolov8_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_annotation_file_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    yolov8_params = Yolov8TrainParams(
        epochs=20,
        batch_size=20,
        model = 'YOLOv8n',
    )

    slice_params = SliceParams(
        height=640,
        width=640,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        min_area_ratio=0.2,
        ignore_negative_samples=True
    )

    sliced_yolov8_train(yolov8_params, slice_params, Path(args.coco_annotation_file_path), Path(args.image_path), device='0')