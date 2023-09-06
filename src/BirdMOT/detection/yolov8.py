from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union
from ultralytics.yolo.utils import RANK

import mlflow
from sahi.predict import predict
from sahi.prediction import PredictionResult
from ultralytics import YOLO

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.data.dataset_tools import merge_coco_datasets
from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams

YOLOV8_PRETRAINED_MODELS = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()


@dataclass
class Yolov8TrainParams: #Todo: Deprecated Safe Delete
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
    model.train(data=yolo_data_path.as_posix(), flipud=0.5, degrees = 180, **asdict(yolo_train_params))  #  ToDo: Put degrees somewhere else after getting rid of data class
    metrics = model.val()  # evaluate model performance on the validation set

def train_yolov8_2(yolo_data_path: Path, yolo_train_params: dict): #ToDo: New version. Delete old one and rename this one.
    # Load a model
    model = YOLO(yolo_train_params['model'])

    #import torch
    #assert torch.cuda.is_available()
    #torch.distributed.init_process_group()

    # Use the model
    model.train(data=yolo_data_path.as_posix(), flipud=0.5, degrees = 180, **yolo_train_params)  #  ToDo: Put degrees somewhere else after getting rid of data class
    if RANK in (0, -1):
        results = model.val()
    return results  # evaluate model performance on the validation set

def sliced_yolov8_train_2(assembly_configs, sliced_dataset_configs, yolo_train_params:dict,  device: str = 'cpu'):
    dataset_creator = DatasetCreator()
    yolo_train_params["device"]=device
    yolov5_fine_tune_dataset = dataset_creator.find_or_create_yolov5_dataset(assembly_configs, sliced_dataset_configs)

    # Train the model
    train_yolov8_2(yolov5_fine_tune_dataset['data_yml_path'], yolo_train_params=yolo_train_params)


def sliced_yolov8_train(yolov8_params: Union[dict, Yolov8TrainParams], slice_params: Union[dict, SliceParams],
                        train_coco_path: Path, val_coco_path: Path, image_dir: Path, device: str = 'cpu'): #ToDo: Deprecated
    if type(yolov8_params) == dict:
        yolov8_params = Yolov8TrainParams(**yolov8_params)

    if type(slice_params) == dict:
        slice_params = SliceParams(**slice_params)

    # Create sliced dataset and based on that the yolov5 dataset
    dataset_path, sliced_train_coco_path, sliced_val_coco_path = DatasetCreator().createSlicedDataset(train_coco_path= train_coco_path,
                                                                                                    val_coco_path=val_coco_path,
                                                                                                    image_dir=image_dir,
                                                                                                    slice_params=slice_params,
                                                                                                    overwrite_existing=True)

    #merge_coco_datasets([sliced_train_coco_path, train_coco_path], )

    yolov5_dataset_path = DatasetCreator().createYolov5Dataset(dataset_path, sliced_train_coco_path,
                                                               sliced_val_coco_path)

    # Train the model
    return train_yolov8((yolov5_dataset_path / "data.yaml"), yolo_train_params=yolov8_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_annotation_file_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    yolov8_params = Yolov8TrainParams(
        epochs=20,
        batch_size=20,
        model='YOLOv8n',
    )

    slice_params = SliceParams(
        height=640,
        width=640,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        min_area_ratio=0.2,
        ignore_negative_samples=True
    )

    sliced_yolov8_train(yolov8_params, slice_params, Path(args.coco_annotation_file_path), Path(args.image_path),
                        device='0')
