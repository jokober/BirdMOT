from pathlib import Path

import pytest

from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams
from BirdMOT.detection.yolov8 import Yolov8TrainParams

config_fixture_path = Path(__file__).parents[1] / 'fixtures' / 'birdnet_config.ini'

# COCO
coco_annotations_fixture_dir = (Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures/annotations')
primary_coco_annotations_fixture_path = (Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures/annotations' / 'C0085_125820_125828_scalabel_converted_coco_format_track_box.json')
secondary_coco_annotations_fixture_path = (Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures/annotations' / 'C0085_204724_204728_scalabel_converted_coco_format_track_box.json')
coco_categories_fixture_path = (Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures/coco_categories.json')
train_coco_fixture_path = Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures' / "dataset_assemblies/test_assembly/test_assembly_train.json"
val_coco_fixture_path = Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures' / "dataset_assemblies/test_assembly/test_assembly_val.json"


# Images
coco_images_fixture_path = Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures' / 'images'
primary_coco_images_fixture_path = Path(__file__).parents[1] / 'fixtures' / 'coco_fixtures' / 'images' / 'good_04_2021' / 'C0085_125820_125828'
primary_val_images_path_fixture = coco_images_fixture_path / "good_04_2021" / "C0085_125820_125828"

# Models
yolov8_test_model = (Path(__file__).parents[1] / 'fixtures' / "models" / "yolov8n.pt")

# Configs
dataset_config_fixture_path = Path(__file__).parents[1] / 'fixtures/config/dataset_assembly' / 'test_assembly_config.json'
experiment_config_fixture_path = Path(__file__).parents[1] / 'fixtures/config/experiment_config' / 'test_experiment_config.json'

# Results

# Other


@pytest.fixture(scope='session')
def local_data_path_fixture(tmp_path_factory):
    fn = tmp_path_factory.mktemp('local_data')
    return fn


@pytest.fixture
def yolov8_params_fixture():
    return Yolov8TrainParams(
        epochs=3,
        batch=3,
        model = 'YOLOv8n',
        patience=10,
        name="pytest_experiment",
        project="BirdMOT Yolov8",
        device="cpu",
        imgsz=640
    )
@pytest.fixture
def slice_params_fixture():
    return SliceParams(
        height= 640,
        width= 640,
        overlap_height_ratio= 0.1,
        overlap_width_ratio= 0.1,
        min_area_ratio= 0.2,
        ignore_negative_samples = True
    )

@pytest.fixture
def sahi_prediction_params():
    return SahiPredictionParams(
        model_type = "yolov8",
        model_path = yolov8_test_model.as_posix(),
        model_device="cpu",
        model_confidence_threshold=0.2,
        source=None,
        no_standard_prediction = False,
        no_sliced_prediction = True,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        export_crop = False,
        postprocess_type = 'GREEDYNMM',
        postprocess_match_metric = 'IOS',
        postprocess_match_threshold = 0.5,
        export_pickle = False,
        dataset_json_path = primary_coco_annotations_fixture_path.as_posix(),
        project = 'Test BirdMOT',
        name = 'test_exp',
        return_dict = True,

    )
