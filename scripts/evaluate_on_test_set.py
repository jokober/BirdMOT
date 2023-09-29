from argparse import ArgumentParser
from pathlib import Path

from sahi.scripts.coco_error_analysis import analyse

from BirdMOT.detection.evaluate import evaluate_coco
from BirdMOT.detection.fiftyone_evaluation import fiftyone_evaluation
from sahi.predict import predict
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--weights_path", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    args = parser.parse_args()


    weights_path = args.weights_path
    model_type = "yolov8"




    sahi_prediction_params = dict(
        model_type=model_type,
        model_path=weights_path.as_posix(),
        model_device=args.device,
        model_confidence_threshold=one_experiment_config['sahi_prediction_params']['model_confidence_threshold'],
        no_standard_prediction=one_experiment_config['sahi_prediction_params']['no_standard_prediction'],
        no_sliced_prediction=one_experiment_config['sahi_prediction_params']['no_sliced_prediction'],
        slice_height=one_experiment_config['sahi_prediction_params']['slice_height'],
        slice_width=one_experiment_config['sahi_prediction_params']['slice_width'],
        overlap_height_ratio=one_experiment_config['sahi_prediction_params']['overlap_height_ratio'],
        overlap_width_ratio=one_experiment_config['sahi_prediction_params']['overlap_width_ratio'],
        export_crop=False,
        postprocess_type=one_experiment_config['sahi_prediction_params']['postprocess_type'],
        postprocess_match_metric=one_experiment_config['sahi_prediction_params']['postprocess_match_metric'],
        postprocess_match_threshold=one_experiment_config['sahi_prediction_params']['postprocess_match_threshold'],
        export_pickle=True,
        project=(model_path / "sahi_predictions").as_posix(),
        name=one_experiment_config['model_config']['name'],
        return_dict=True,
    )

    coco, pred_coco_path, source = self.generate_prediction_folder_with_coco_file(
        assembly_id=assembly_config['dataset_assembly_id'], coco_path=coco_path,
        image_path=DatasetCreator().images_dir)

    sahi_prediction_params["source"] = source.as_posix()
    sahi_prediction_params["dataset_json_path"] = pred_coco_path.as_posix()

    sahi_prediction_params["name"] = prediction_hash
    predictions = predict(**sahi_prediction_params)

    predictions_path = Path(prediction_results['data']['export_dir']) / 'result.json'
    # Evaluation
    eval_results = evaluate_coco(dataset_json_path=prediction_results['coco_path'],
                                 result_json_path=predictions_path.as_posix(),
                                 iou_thrs=one_experiment_config['evaluation_config']['iou_thrs'],
                                 out_dir=(self.tmp_eval_path / evaluation_hash / "evaluation").as_posix())

    # Analysis
    analysis_results = analyse(dataset_json_path=prediction_results['coco_path'],
                               result_json_path=(Path(
                                   prediction_results['data']['export_dir']) / 'result.json').as_posix(),
                               out_dir=(self.tmp_eval_path / evaluation_hash / "analysis").as_posix(),
                               type="bbox", return_dict=True, )

    fiftyone_eval_results = fiftyone_evaluation(image_path=DatasetCreator().images_dir,
                                                labels_path=prediction_results['coco_path'],
                                                predictions_path=predictions_path)