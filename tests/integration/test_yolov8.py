from BirdMOT.data.dataset_tools import coco2yolov5, create_sliced_dataset
from BirdMOT.detection.predict import sliced_batch_predict
from BirdMOT.detection.yolov8 import train_yolov8, sliced_yolov8_train
from fixtures.fixtures import coco_images_fixture_path, \
    primary_coco_annotations_fixture_path, local_data_path_fixture, slice_params_fixture, yolov8_params_fixture, \
    sahi_prediction_params, primary_val_images_path_fixture


def test_yolov8_train_from_nested_coco(local_data_path_fixture, slice_params_fixture, yolov8_params_fixture):
    dataset_path, train_coco_path, val_coco_path = create_sliced_dataset(train_coco_path, val_coco_path, image_dir=coco_images_fixture_path, slice_params=slice_params_fixture)

    coco2yolov5(dataset_path=dataset_path, coco_images_dir=dataset_path / "images")

    assert (dataset_path / "yolov5_files" / "data.yaml").exists(), "data.yaml file in yolov5_files does not exist"
    # coco2yolov5(dataset_path=dataset_path, coco_images_dir=local_data_path_fixture / "images")
    dataset_content = list(dataset_path.glob('**/*'))
    train_yolov8("yolov8n.pt", (dataset_path / "yolov5_files" / "data.yaml"), yolov8_params_fixture)


def test_sliced_yolov8_prediction(sahi_prediction_params, local_data_path_fixture):
    sahi_prediction_params.project = (local_data_path_fixture / "sahi_prediction_test").as_posix()
    sliced_batch_predict(image_dir = primary_val_images_path_fixture, sahi_prediction_params=sahi_prediction_params)

def test_sliced_yolov8_train(yolov8_params_fixture, slice_params_fixture):
    sliced_yolov8_train(yolov8_params=yolov8_params_fixture, slice_params=slice_params_fixture, train_coco_path=primary_coco_annotations_fixture_path, image_dir=coco_images_fixture_path)