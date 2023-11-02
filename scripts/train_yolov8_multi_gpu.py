import json

from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.yolov8 import sliced_yolov8_train_2

"""
This File does not accept ArgumentParser arguments as it lead to errors in the multiprocessing module. https://github.com/ultralytics/ultralytics/issues/1476
"""
if __name__ == "__main__":
    device = [0,1,2,3]
    experiment_config_path= Path("/data/jokoko/local_data/configs/experiments/yolov8n_setup_comparison_320_640.json").expanduser()
    #experiment_config_path= Path("~/fids/local_data/configs/experiments/yolov8n_sahi_dataset_assembly2_rc_4good_tracks_in_val_full_resolution.json").expanduser()
    #experiment_config_path= Path("~/fids/local_data/configs/experiments/yolov8m_sahi_dataset_assembly2_rc_4good_tracks_in_val.json").expanduser()
    #assembly_config_path= Path("~/fids/local_data/configs/dataset_assembly/dataset_assembly2_rc_4good_tracks_in_val.json").expanduser()
    assembly_config_path= Path("~/searchwing/local_data/configs/dataset_assembly/dataset_assembly1.json").expanduser()

    assert experiment_config_path.exists(), f"Experiment config {experiment_config_path} does not exist"
    with open(experiment_config_path) as json_file:
        experiment_config = json.load(json_file)

    assert assembly_config_path.exists(), f"Assembly config {assembly_config_path} does not exist"
    with open(assembly_config_path) as json_file:
        assembly_config = json.load(json_file)


    for experiment in experiment_config["experiments"]:
        print(experiment)
        experiment["model_config"]["project"] = (DatasetCreator().models_dir / experiment_config["dataset_assembly_id"]).as_posix()
        sliced_yolov8_train_2(
            assembly_configs=assembly_config,
            sliced_dataset_configs=experiment["sliced_datasets"],
            yolo_train_params=experiment["model_config"],
            device=device)