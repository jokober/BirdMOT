from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.predict import sliced_batch_predict
from BirdMOT.detection.yolov8 import train_yolov8, sliced_yolov8_train
from fixtures.fixtures import coco_images_fixture_path, \
    primary_coco_annotations_fixture_path, slice_params_fixture, yolov8_params_fixture, \
    sahi_prediction_params, primary_val_images_path_fixture, train_coco_fixture_path, val_coco_fixture_path
from conftest import local_data_path_fixture


def test_yolov8_train_from_nested_coco(local_data_path_fixture, slice_params_fixture, yolov8_params_fixture):
    # Create sliced dataset and based on that the yolov5 dataset
    dataset_path, sliced_train_coco_path, sliced_val_coco_path = DatasetCreator().createSlicedDataset(
        train_coco_fixture_path,
        val_coco_fixture_path,
        image_dir=coco_images_fixture_path,
        slice_params=slice_params_fixture,
        overwrite_existing=True)
    yolov5_dataset_path = DatasetCreator().createYolov5Dataset(dataset_path, sliced_train_coco_path,
                                                               sliced_val_coco_path)

    assert (dataset_path / "yolov5_files" / "data.yaml").exists(), "data.yaml file in yolov5_files does not exist"
    assert (yolov5_dataset_path).exists(), "yolov5_dataset_path does not exist"

    yolov8_params_fixture.project = (DatasetCreator().models_dir / "test_assembly").as_posix()
    train_yolov8((dataset_path / "yolov5_files" / "data.yaml"), yolov8_params_fixture)


def test_sliced_yolov8_prediction(sahi_prediction_params, local_data_path_fixture):
    sahi_prediction_params.project = (local_data_path_fixture / "sahi_prediction_test").as_posix()
    sahi_prediction_params.project = (DatasetCreator().models_dir / "test_assembly/predictions").as_posix()
    prediction_res = sliced_batch_predict(image_dir=coco_images_fixture_path,
                                          sahi_prediction_params=sahi_prediction_params)
    assert Path(prediction_res['export_dir']).exists(), "export_dir does not exist"


def test_sliced_yolov8_train(yolov8_params_fixture, slice_params_fixture):
    yolov8_params_fixture.project = (DatasetCreator().models_dir / "test_assembly/predictions").as_posix()
    sliced_yolov8_train(yolov8_params=yolov8_params_fixture, slice_params=slice_params_fixture,
                        train_coco_path=train_coco_fixture_path, val_coco_path=val_coco_fixture_path,
                        image_dir=coco_images_fixture_path)
