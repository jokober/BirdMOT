from pathlib import Path

from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate

from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams
from BirdMOT.detection.predict import sliced_batch_predict


def evaluate_coco(dataset_json_path, result_json_path, iou_thrs=0.50, out_dir=None):
    eval_results = evaluate(
        dataset_json_path =dataset_json_path,
        result_json_path =result_json_path,
        out_dir = out_dir,
        type = "bbox",
        classwise = True,
        max_detections = 10000,
        iou_thrs = iou_thrs,
        areas = [900, 70000, 7000000], #ToDo: make this a parameter Docs: object area ranges for evaluation https://github.com/cocodataset/cocoapi/issues/289
        return_dict = True,
    )

    print(eval_results)

if __name__ == "__main__":
    sahi_prediction_params = SahiPredictionParams(
        model_type = "yolov8",
        model_path = "/home/fids/fids/BirdMOT/yolov8n_best.pt",
        model_device="0",
        model_confidence_threshold=0.2,
        source=None,
        no_standard_prediction = False,
        no_sliced_prediction = False,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2,
        export_crop = False,
        postprocess_type = 'GREEDYNMM',
        postprocess_match_metric = 'IOS',
        postprocess_match_threshold = 0.5,
        export_pickle = False,
        dataset_json_path = "/home/jo/coding_projects/fids/BirdMOT/tests/Test BirdMOT/exp27/results.json",
        project = 'Test BirdMOT',
        name = 'test_exp',
        return_dict = True,

    )

    sahi_prediction_params.dataset_json_path = "/media/data/BirdMOT/local_data/dataset/coco_files/merged/birdmot_05_2023_coco.json"
    predictions = sliced_batch_predict(Path("/media/data/BirdMOT/local_data/dataset/images"), sahi_prediction_params)
    #assert (tmp_path / "result_coco.json").exists(), "result_coco.json should exist"

    with open(f"{predictions['export_dir'].as_posix()}/result.json", "rt") as fin:
        with open(f"{predictions['export_dir'].as_posix()}/result_cat_mapped.json", "wt") as fout:
            for line in fin:
                fout.write(line.replace("""
                "category_id":0"
                """,
                """
                "category_id":1
                """)
                )

    evaluate_coco( sahi_prediction_params.dataset_json_path, f"{predictions['export_dir'].as_posix()}/result_cat_mapped.json")
