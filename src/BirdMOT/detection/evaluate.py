import json
import os
import shutil
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Union, Dict, List

from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate
from sahi.predict import predict

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.data.dataset_tools import rapair_absolute_image_paths
from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams
from BirdMOT.detection.predict import sliced_batch_predict


def evaluate_coco(dataset_json_path, result_json_path, iou_thrs=0.50, out_dir=None):
    eval_results = evaluate(
        dataset_json_path=dataset_json_path,
        result_json_path=result_json_path,
        out_dir=out_dir,
        type="bbox",
        classwise=True,
        max_detections=10000,
        iou_thrs=iou_thrs,
        areas=[900, 70000, 7000000],
        # ToDo: make this a parameter Docs: object area ranges for evaluation https://github.com/cocodataset/cocoapi/issues/289
        return_dict=True,

    )

    print(eval_results)


def evaluate_coco_from_config(experiment_configs: Union[Path, Dict], coco_path: Path, device: str):
    if isinstance(experiment_configs, Path):
        with open(experiment_configs) as exp_file:
            experiment_configs = json.load(exp_file)


        # Load coco file
        with open(coco_path) as coco_file:
            coco = json.load(coco_file)

        rapair_absolute_image_paths(coco_path, DatasetCreator().images_dir, overwrite_file=True)

        # Create symlink folder
        image_paths = [Path(image['file_name']) for image in coco['images']]
        source = generate_images_symlink_folder(image_paths, coco_path.stem)

    for one_experiment_config in experiment_configs['experiments']:
        print(one_experiment_config)
        print(one_experiment_config['model_config'])
        print(one_experiment_config['model_config']['name'])
        # Get model path
        model_path = DatasetCreator().models_dir / one_experiment_config['model_config']['name']
        dataset_result_json = model_path / "result.json"
        # result_path = (model_path / "results.json").as_posix()

        model_type = 'yolov8' if 'yolov8' in one_experiment_config['model_config']['model'] else None
        #model_type = 'yolov8' if one_experiment_config['model_config']['model'].contains('yolov8')

        sahi_prediction_params = SahiPredictionParams(
            model_type=model_type,
            model_path=(model_path / "weights" / "best.pt").as_posix(),
            model_device=device,
            model_confidence_threshold=one_experiment_config['sahi_prediction_params']['model_confidence_threshold'],
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
            dataset_json_path=coco_path.as_posix(),
            project=(model_path / "predictions").as_posix(),
            name=one_experiment_config['model_config']['name'],
            return_dict=True,

        )
        predictions = predict( **asdict(sahi_prediction_params))
        print(predictions)
        evaluate_coco(dataset_json_path=coco_path, result_json_path=(predictions['export_dir'] / 'result.json' ).as_posix(), iou_thrs=0.5,
                      out_dir=f"{predictions['export_dir'].as_posix()}")


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
