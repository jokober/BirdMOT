from dataclasses import dataclass, asdict
from pathlib import Path

import mlflow
from ultralytics import YOLO

from BirdMOT.data.DatasetCreator import DatasetCreator

YOLOV8_PRETRAINED_MODELS = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]


@dataclass
class Yolov8TrainParams:  # Todo: Deprecated Safe Delete
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
    imgsz: int
    exist_ok: bool = True


def train_yolov8(yolo_data_path: Path, yolo_train_params: Yolov8TrainParams):
    # Load a model
    model = YOLO(yolo_train_params.model)

    # Use the model
    model.train(data=yolo_data_path.as_posix(), flipud=0.5, degrees=180,
                **asdict(yolo_train_params))  # ToDo: Put degrees somewhere else after getting rid of data class
    metrics = model.val()  # evaluate model performance on the validation set


def train_yolov8_2(yolo_data_path: Path,
                   yolo_train_params: dict,
                   cli_subprocess=False):  # ToDo: New version. Delete old one and rename this one.
    # Load a model
    model = YOLO(yolo_train_params['model'])

    model.train(data=yolo_data_path.as_posix(), save=True, val=True, flipud=0.5, degrees=180,
                **yolo_train_params)  # ToDo: Put degrees somewhere else after getting rid of data class

    save_dir = Path(yolo_train_params['project']) / yolo_train_params['name']
    results = {'save_dir': save_dir}
    return results  # evaluate model performance on the validation set


def sliced_yolov8_train_2(assembly_configs, sliced_dataset_configs, yolo_train_params: dict, device: str = 'cpu'):
    dataset_creator = DatasetCreator()
    yolo_train_params["device"] = device
    yolov5_fine_tune_dataset = dataset_creator.find_or_create_yolov5_dataset(assembly_configs, sliced_dataset_configs)

    # Train the model
    return train_yolov8_2(yolov5_fine_tune_dataset['data_yml_path'], yolo_train_params=yolo_train_params)
