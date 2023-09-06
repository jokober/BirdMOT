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


def evaluate_coco(dataset_json_path, result_json_path, iou_thrs=0.50, out_dir=None):
    eval_results = evaluate(
        dataset_json_path=dataset_json_path,
        result_json_path=result_json_path,
        out_dir=out_dir,
        type="bbox",
        classwise=True,
        max_detections=1000000,
        iou_thrs=iou_thrs,
        areas=[32 ** 2, 96 ** 2, 1e5 ** 2],
        #[[0 ** 2, 32 ** 2],     # small - objects that have area between 0 sq pixels and 32*32 (1024) sq pixels
        #        [32 ** 2, 96 ** 2],   # medium - objects that have area between 32*32 sq pixels and 96*96 (9216) sq pixels
        #        [80 ** 2, 1e5 ** 2]],

        # Found in sahi code:
        #cocoEval.params.areaRng = [
        #[0 ** 2, areas[2]],
        #[0 ** 2, areas[0]],
        #[areas[0], areas[1]],
        #[areas[1], areas[2]],
   # ]
        # ToDo: make this a parameter Docs: object area ranges for evaluation https://github.com/cocodataset/cocoapi/issues/289
        return_dict=True,

    )

    return eval_results


def evaluate_coco_from_config(experiment_configs: Union[Path, Dict], coco_path: Path, device: str):
    if isinstance(experiment_configs, Path):
        with open(experiment_configs) as exp_file:
            experiment_configs = json.load(exp_file)

#    rapair_absolute_image_paths(coco_path, DatasetCreator().images_dir, overwrite_file=True)

    coco, pred_coco_path , source = generate_prediction_folder_with_coco_file(dataset_assembly_id= experiment_configs['dataset_assembly_id'], coco_path= coco_path, image_path= DatasetCreator().images_dir,)




    for one_experiment_config in experiment_configs['experiments']:
        print(one_experiment_config)
        print(one_experiment_config['model_config'])
        print(one_experiment_config['model_config']['name'])
        # Get model path
        model_path = DatasetCreator().models_dir / experiment_configs["dataset_assembly_id"] / one_experiment_config['model_config']['name']
        dataset_result_json = model_path / "result.json"
        # result_path = (model_path / "results.json").as_posix()

        model_type = 'yolov8' if 'yolov8' in one_experiment_config['model_config']['model'] else None
        #model_type = 'yolov8' if one_experiment_config['model_config']['model'].contains('yolov8')

        sahi_prediction_params = SahiPredictionParams(
            model_type=model_type,
            model_path=(model_path / "weights" / "best.pt").as_posix(),
            model_device=device,
            model_confidence_threshold=0.001, #one_experiment_config['sahi_prediction_params']['model_confidence_threshold'],
            source=source.as_posix(),
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
            export_pickle=False,
            dataset_json_path=pred_coco_path.as_posix(),
            project=(model_path / "predictions").as_posix(),
            name=one_experiment_config['model_config']['name'],
            return_dict=True,

        )
        predictions = predict( **asdict(sahi_prediction_params))
        print(predictions)

        results_dir= model_path / "results"
        results_dir.mkdir(exist_ok=True)
        # Evaluation
        eval_results = evaluate_coco(dataset_json_path=pred_coco_path, result_json_path=(predictions['export_dir'] / 'result.json' ).as_posix(), iou_thrs=one_experiment_config['evaluation_config']['iou_thrs'],
                      out_dir=results_dir.as_posix())


        # Analysis
        analysis_results = analyse(dataset_json_path= pred_coco_path.as_posix(), result_json_path=(predictions['export_dir'] / 'result.json' ).as_posix(), out_dir=results_dir.as_posix(), type="bbox",  return_dict=True,)
        print(analysis_results)

        # Log evaluation
        mlflow_params = asdict(sahi_prediction_params)
        mlflow_params.update({'iou_thrs': one_experiment_config['evaluation_config']['iou_thrs']})
        log_evaluation(f"Eval Sahi {one_experiment_config['model_config']['name']}", mlflow_params, eval_results['eval_results'], results_dir)


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


def generate_prediction_folder_with_coco_file(dataset_assembly_id: str, coco_path: Path, image_path: Path):
        assembly_img_symlink_dir = DatasetCreator().val_images_dir / dataset_assembly_id

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

            coco_pred_path = assembly_img_symlink_dir / f"{dataset_assembly_id}_pred.json"
            with open(coco_pred_path, 'w') as fp:
                json.dump(coco_dict, fp)

        return coco_dict, coco_pred_path, assembly_img_symlink_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--coco_path", type=Path, required=True)
    parser.add_argument("--experiment_config", type=Path, required=True)
    parser.add_argument("--device", type=str, required=False)
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = "cpu"

    evaluate_coco_from_config(experiment_configs=args.experiment_config, coco_path=args.coco_path, device=device)
    # sahi_prediction_params = SahiPredictionParams(
    #    model_type = "yolov8",
    #    model_path = "/home/fids/fids/BirdMOT/yolov8n_best.pt",
    #    model_device="0",
    #    model_confidence_threshold=0.2,
    #    source=None,
    #    no_standard_prediction = False,
    #    no_sliced_prediction = False,
    #    slice_height = 640,
    #    slice_width = 640,
    #    overlap_height_ratio = 0.2,
    #    overlap_width_ratio = 0.2,
    #    export_crop = False,
    #    postprocess_type = 'GREEDYNMM',
    #    postprocess_match_metric = 'IOS',
    #    postprocess_match_threshold = 0.5,
    #    export_pickle = False,
    #    dataset_json_path = "/home/jo/coding_projects/fids/BirdMOT/tests/Test BirdMOT/exp27/results.json",
    #    project = 'Test BirdMOT',
    #    name = 'test_exp',
    #    return_dict = True,

#   )
#
# sahi_prediction_params.dataset_json_path = "/media/data/BirdMOT/local_data/dataset/coco_files/merged/birdmot_05_2023_coco.json"
# predictions = sliced_batch_predict(Path("/media/data/BirdMOT/local_data/dataset/images"), sahi_prediction_params)
# assert (tmp_path / "result_coco.json").exists(), "result_coco.json should exist"

# with open(f"{predictions['export_dir'].as_posix()}/result.json", "rt") as fin:
#    with open(f"{predictions['export_dir'].as_posix()}/result_cat_mapped.json", "wt") as fout:
#        for line in fin:
#            fout.write(line.replace("""
#            "category_id":0"
#            """,
#            """
#            "category_id":1
#            """)
#            )

# evaluate_coco( sahi_prediction_params.dataset_json_path, f"{predictions['export_dir'].as_posix()}/result_cat_mapped.json")
