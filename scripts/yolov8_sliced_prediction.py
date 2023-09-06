from argparse import ArgumentParser
from pathlib import Path
import json
import os
import shutil
import time
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Union, Dict, List

from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate
from sahi.predict import predict

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.dataset_tools import rapair_absolute_image_paths, find_correct_image_path
from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams
from BirdMOT.detection.predict import sliced_batch_predict
from BirdMOT.helper.mlflow_tracking import log_evaluation

from BirdMOT.data.SliceParams import SliceParams
from BirdMOT.detection.yolov8 import sliced_yolov8_train, Yolov8TrainParams

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_coco_file", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--weights_path", type=Path, required=True)
    parser.add_argument("--model_device", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    args = parser.parse_args()

    predictions = predict(**dict(
        model_type='yolov8',
        model_path=args.weights_path.as_posix(),
        model_device=args.model_device,
        model_confidence_threshold=0.001,
        # one_experiment_config['sahi_prediction_params']['model_confidence_threshold'],
        source=args.image_path.as_posix(),
        no_standard_prediction=False,
        no_sliced_prediction=False,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        export_crop=False,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.2,
        export_pickle=False,
        dataset_json_path=args.dataset_coco_file.as_posix(),
        project=args.project_name,
        name=args.weights_path.parent.name,
        return_dict=True)
    )
    print(predictions)

    #results_dir = model_path / "results"
    #results_dir.mkdir(exist_ok=True)
    ## Evaluation
    #eval_results = evaluate_coco(dataset_json_path=pred_coco_path,
    #                             result_json_path=(predictions['export_dir'] / 'result.json').as_posix(),
    #                             iou_thrs=one_experiment_config['evaluation_config']['iou_thrs'],
    #                             out_dir=results_dir.as_posix())

    # Analysis
    #analysis_results = analyse(dataset_json_path=pred_coco_path.as_posix(),
    #                           result_json_path=(predictions['export_dir'] / 'result.json').as_posix(),
    #                           out_dir=results_dir.as_posix(), type="bbox", return_dict=True, )
    #print(analysis_results)

