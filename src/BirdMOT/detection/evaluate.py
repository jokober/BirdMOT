import json
import pickle
import shutil
from copy import deepcopy
from os import environ
from pathlib import Path
from typing import List

from deepdiff import DeepHash
from sahi.predict import predict
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.dataset_tools import find_correct_image_path
from BirdMOT.detection.TrainingController import TrainingController
from BirdMOT.detection.fiftyone_evaluation import fiftyone_evaluation
from BirdMOT.helper.config import get_local_data_path
from BirdMOT.helper.mlflow_tracking import log_evaluation
from BirdMOT.helper.sahi_utils import get_model_type, get_setup_name
from BirdMOT.helper.yolov_args_reader import load_filtered_yolo_args_yaml


class EvaluationController:
    def __init__(self):
        self.local_data_path = get_local_data_path()
        self.tmp_eval_dir_path = self.local_data_path / 'tmp_eval_dir'
        self.tmp_eval_dir_path.mkdir(parents=True, exist_ok=True)
        self.tmp_eval_state_path: Path = self.tmp_eval_dir_path / 'tmp_eval_state.pkl'
        self.tmp_pred_path: Path = self.tmp_eval_dir_path / 'predictions'
        self.tmp_pred_path.mkdir(parents=True, exist_ok=True)
        self.tmp_eval_path: Path = self.tmp_eval_dir_path / 'evaluations'
        self.tmp_eval_path.mkdir(parents=True, exist_ok=True)

        self.state = None
        self.load_state()

    def write_state(self):
        with open(self.tmp_eval_state_path, 'wb') as handle:
            pickle.dump(self.state, handle)

    def load_state(self):
        if self.tmp_eval_state_path.exists():
            with open(self.tmp_eval_state_path, 'rb') as handle:
                self.state = pickle.load(handle)
        else:
            print("tmp_state_path does not exist. Creating new state.")
            self.create_new_eval_state()
            self.write_state()

    def create_new_eval_state(self):
        self.state = {
            'predictions': [],
            'evaluations': [],
        }

    def delete_existing(self, key, hash):
        self.load_state()
        print("Deleting existing")

        self.state[key] = [item for item in self.state[key] if item["hash"] != hash]
        assert not hash in [item for item in self.state[key]]
        self.write_state()

    def update_state(self, type, key, value):
        self.load_state()
        if type == 'append':
            self.state[key].append(value)
        else:
            raise NotImplementedError("The type is not implemented.")
        self.write_state()

    def find_or_create_prediction(self, one_experiment_config: dict, assembly_config, device='cpu',
                                  train_missing=False):
        one_experiment_config = deepcopy(one_experiment_config)
        assembly_config = deepcopy(assembly_config)

        assembly = DatasetCreator().find_or_create_dataset_assembly(assembly_config)
        coco_path = assembly['data']['val']['path']

        model = TrainingController().find_or_train_model(one_experiment_config, assembly_config, device=device,
                                                         train_missing=train_missing)
        model_path = model['data']['model_path']
        weights_path = model['data']['weights_path']

        model_type = get_model_type(one_experiment_config, include_version=False)
        # model_type = 'yolov8' if one_experiment_config['model_config']['model'].contains('yolov8')

        sahi_prediction_params = dict(
            model_type=model_type,
            model_path=weights_path.as_posix(),
            model_device=device,
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

        prediction_config = {
            "sahi_prediction_params": sahi_prediction_params,
            "coco_path": coco_path,
            "hash": None,
            "data": None
        }

        deephash_exclude_paths = [
            "root['sahi_prediction_params']['dataset_json_path']",
            "root['sahi_prediction_params']['source']",
            "root['sahi_prediction_params']['name']",
            "root['sahi_prediction_params']['project']",
            "root['hash']",
            "root['data']",
        ]
        prediction_hash = DeepHash(prediction_config, exclude_paths=deephash_exclude_paths)[prediction_config]

        # Create predictions if they do not exist yet
        if prediction_hash not in [pred_conf["hash"] for pred_conf in self.state['predictions']]:
            coco, pred_coco_path, source = self.generate_prediction_folder_with_coco_file(
                assembly_id=assembly_config['dataset_assembly_id'], coco_path=coco_path,
                image_path=DatasetCreator().images_dir)

            sahi_prediction_params["source"] = source.as_posix()
            sahi_prediction_params["dataset_json_path"] = pred_coco_path.as_posix()

            sahi_prediction_params["name"] = prediction_hash
            predictions = predict(**sahi_prediction_params)
            prediction_config['data'] = predictions
            prediction_config['data']['weights_path'] = weights_path.as_posix()
            prediction_config['hash'] = prediction_hash
            self.update_state(type="append", key='predictions', value=prediction_config)

        else:
            prediction_config = [prediction_config for prediction_config in self.state['predictions'] if
                                 prediction_config["hash"] == prediction_hash][0]

        return prediction_config

    def find_or_create_evaluation(self, one_experiment_config: dict, assembly_config: dict, device: str = 'cpu',
                                  train_missing: bool = False, overwrite_existing=False):
        one_experiment_config = deepcopy(one_experiment_config)
        assembly_config = deepcopy(assembly_config)

        prediction_results = self.find_or_create_prediction(one_experiment_config=one_experiment_config,
                                                            assembly_config=assembly_config, device=device,
                                                            train_missing=train_missing)

        sahi_evaluation = {
            "one_experiment_config": one_experiment_config,
            "assembly_config": assembly_config,
            "sahi_prediction_params": prediction_results['sahi_prediction_params'],
            "data": {}
        }
        deephash_exclude_paths = [
            "root['one_experiment_config']['model_config']['project']",
            "root['one_experiment_config']['model_config']['name']",
            "root['one_experiment_config']['model_config']['device']",
            "root['one_experiment_config']['model_config']['exists_ok']",
            "root['one_experiment_config']['hash']",
            "root['assembly_config']['hash']",
            "root['sahi_prediction_params']['hash']",
            "root['data']",
            "root['hash']",
            "root['sahi_setup_name']",
            "root['setup_name']",
        ]
        evaluation_hash = DeepHash(sahi_evaluation, exclude_paths=deephash_exclude_paths)[sahi_evaluation]

        # Delete existing evaluation if overwrite_existing is True
        if overwrite_existing:
            self.delete_existing('evaluations', evaluation_hash)

        # Check if evaluation already exists return if so or create otherwise
        if evaluation_hash not in [eval_hash["hash"] for eval_hash in self.state['evaluations']]:
            predictions_path = Path(prediction_results['data']['export_dir']) / 'result.json'
            # Evaluation
            eval_results = evaluate_coco(dataset_json_path=prediction_results['coco_path'],
                                         result_json_path=predictions_path.as_posix(),
                                         #iou_thrs=one_experiment_config['evaluation_config']['iou_thrs'],
                                         out_dir=(self.tmp_eval_path / evaluation_hash / "evaluation").as_posix())

            # Analysis
            analysis_results = analyse(dataset_json_path=prediction_results['coco_path'],
                                       result_json_path=(Path(
                                           prediction_results['data']['export_dir']) / 'result.json').as_posix(),
                                       out_dir=(self.tmp_eval_path / evaluation_hash / "analysis").as_posix(),
                                       type="bbox", return_dict=True, )

            fiftyone_eval_results = fiftyone_evaluation(image_path=DatasetCreator().images_dir,
                                                        labels_path=prediction_results['coco_path'],
                                                        predictions_path=predictions_path,
                                                        iou=0.4)

            eval_results['eval_results'].pop('bbox_mAP_copypaste')
            eval_results.pop('export_path')

            sahi_evaluation['sahi_prediction_params'] = prediction_results['sahi_prediction_params']
            sahi_evaluation['data']['eval_results'] = eval_results['eval_results']
            sahi_evaluation['data']['eval_results_pycocotools'] = eval_results['eval_results_pycocotools']
            sahi_evaluation['data']['eval_results_fiftyone'] = fiftyone_eval_results
            sahi_evaluation['data']['analysis_results'] = analysis_results
            sahi_evaluation['data']['prediction_result'] = prediction_results
            sahi_evaluation['hash'] = evaluation_hash
            sahi_evaluation['setup_name'] = get_setup_name(one_experiment_config)
            self.update_state(type="append", key='evaluations', value=sahi_evaluation)
        else:
            sahi_evaluation = \
                [eval_config for eval_config in self.state['evaluations'] if eval_config["hash"] == evaluation_hash][0]

        # Log evaluation
        if environ.get('MLFLOW_TRACKING_URI') is not None:
            mlflow_params = sahi_evaluation['sahi_prediction_params']
            mlflow_params.update({'setup_name': sahi_evaluation['setup_name']})
            mlflow_params.update({'iou_thrs': one_experiment_config['evaluation_config']['iou_thrs']})
            mlflow_params.update(
                {i: one_experiment_config['sliced_datasets'][0][i] for i in one_experiment_config['sliced_datasets'][0]
                 if i != 'height' and i != 'width'})
            train_w_slices = sorted(
                [train_slice_conf['width'] for train_slice_conf in one_experiment_config['sliced_datasets']])
            train_h_slices = sorted(
                [train_slice_conf['height'] for train_slice_conf in one_experiment_config['sliced_datasets']])
            mlflow_params.update({'train_width_slices_config': ','.join([str(i) for i in train_w_slices])})
            mlflow_params.update({'train_height_slices_config': ','.join([str(i) for i in train_h_slices])})
            # Add args used for yolo training to mlflow params
            mlflow_params.update(
                load_filtered_yolo_args_yaml(Path(prediction_results['data']['weights_path']).parents[1] / 'args.yaml'))

            mlflow_metrics = {}
            # mlflow_metrics.update(analysis_results)
            mlflow_metrics.update(sahi_evaluation['data']['eval_results'])
            # mlflow_metrics.update(sahi_evaluation['data']['eval_results_pycocotools']) #mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Invalid metric name: 'bbox_AR@1'. Names may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).

            mlflow_metrics.update({'prediction_speed': prediction_results['data']['durations_in_seconds']})
            mlflow_artifact_paths = [
                Path(sahi_evaluation['data']["analysis_results"]['bbox']['overall']['gt_area_histogram']).parent,
                prediction_results['data']['weights_path']

            ]
            log_evaluation(f"BirdMOT Yolov8", mlflow_params,
                           mlflow_metrics, mlflow_artifact_paths)

        return sahi_evaluation

    def generate_prediction_folder_with_coco_file(self, assembly_id: str, coco_path: Path, image_path: Path):
        assembly_img_symlink_dir = self.tmp_pred_path / assembly_id

        # Go through all image paths in coco file and adjust them to absolute path on disk
        with open(coco_path) as json_file:
            coco_dict = json.load(json_file)

        if assembly_img_symlink_dir.exists():
            # Check if number of files in symlink dir is equal to number of images in val set. Recreate folder if not.
            if len(coco_dict['images']) != len(list(assembly_img_symlink_dir.glob('*'))):
                shutil.rmtree(assembly_img_symlink_dir)
                assembly_img_symlink_dir.mkdir(parents=True, exist_ok=True)

        for idx, it in enumerate(coco_dict['images']):
            new_abs_image_path = Path(find_correct_image_path(image_path, it["file_name"]))

            image_path_symlink = (assembly_img_symlink_dir / f"{idx}{Path(it['file_name']).suffix}")
            it["file_name"] = image_path_symlink.as_posix()
            if image_path_symlink.is_symlink() and image_path_symlink.exists():
                pass
            else:
                target = new_abs_image_path
                print(f"target: {target} {target.exists()}")
                my_symlink = image_path_symlink
                print(f"my_symlink: {my_symlink}  {my_symlink.exists()}")
                my_symlink.parent.mkdir(parents=True, exist_ok=True)
                my_symlink.symlink_to(target)

            coco_pred_path = assembly_img_symlink_dir / f"{assembly_id}_pred.json"
            with open(coco_pred_path, 'w') as fp:
                json.dump(coco_dict, fp)

        return coco_dict, coco_pred_path, assembly_img_symlink_dir


def evaluate_coco(dataset_json_path, result_json_path, iou_thrs=None, out_dir=None):
    eval_results = evaluate(
        dataset_json_path=dataset_json_path,
        result_json_path=result_json_path,
        out_dir=out_dir,
        type="bbox",
        classwise=False,
        max_detections=1000000,
        iou_thrs=iou_thrs,
        areas=[32 ** 2, 96 ** 2, 1e5 ** 2],
        # [[0 ** 2, 32 ** 2],     # small - objects that have area between 0 sq pixels and 32*32 (1024) sq pixels
        #        [32 ** 2, 96 ** 2],   # medium - objects that have area between 32*32 sq pixels and 96*96 (9216) sq pixels
        #        [80 ** 2, 1e5 ** 2]],

        # Found in sahi code:
        # cocoEval.params.areaRng = [
        # [0 ** 2, areas[2]],
        # [0 ** 2, areas[0]],
        # [areas[0], areas[1]],
        # [areas[1], areas[2]],
        # ]
        # ToDo: make this a parameter Docs: object area ranges for evaluation https://github.com/cocodataset/cocoapi/issues/289
        return_dict=True,

    )

    return eval_results


def generate_images_symlink_folder(image_paths: List[Path], dataset_assembly_id: str) -> Path:
    """
    Generate folder with images symlink files of validation set
    """
    assembly_img_symlink_dir = DatasetCreator().val_images_dir / dataset_assembly_id

    if assembly_img_symlink_dir.exists():
        # Check if number of files in symlink dir is equal to number of images in val set. Recreate folder if not.
        if len(image_paths) != len(list(assembly_img_symlink_dir.glob('*'))):
            shutil.rmtree(assembly_img_symlink_dir)
            assembly_img_symlink_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all images in val set and create symlink to them in symlink folder
    for image_path in image_paths:
        image_path_symlink = assembly_img_symlink_dir / image_path.name
        if image_path_symlink.is_symlink() and image_path_symlink.exists():
            pass
        else:
            target = image_path
            print(f"target: {target} {target.exists()}")
            my_symlink = image_path_symlink
            print(f"my_symlink: {my_symlink}  {my_symlink.exists()}")
            my_symlink.parent.mkdir(parents=True, exist_ok=True)
            my_symlink.symlink_to(target)

    return assembly_img_symlink_dir
