from shutil import copy

from BirdMOT.detection.evaluate import evaluate_coco
from BirdMOT.detection.predict import sliced_batch_predict
from fixtures.fixtures import primary_val_images_path_fixture, sahi_prediction_params, \
    primary_coco_annotations_fixture_path, coco_images_fixture_path, primary_coco_images_fixture_path


def test_predict_evaluate_analyse(tmp_path ,sahi_prediction_params):
    sahi_prediction_params.dataset_json_path = primary_coco_annotations_fixture_path.as_posix()
    predictions = sliced_batch_predict(coco_images_fixture_path, sahi_prediction_params)
    #assert (tmp_path / "result_coco.json").exists(), "result_coco.json should exist"

    evaluate_coco(primary_coco_annotations_fixture_path, f"{predictions['export_dir'].as_posix()}/result.json")