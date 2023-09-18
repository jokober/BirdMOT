import json
import shutil
from copy import deepcopy
from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.TrainingController import TrainingController
from BirdMOT.detection.evaluate import EvaluationController
from fixtures.fixtures import assembly_configs, sliced_dataset_configs, one_experiment_config_fixture


def test_find_or_train_model(one_experiment_config_fixture, assembly_configs):
    eval_controller = TrainingController()
    shutil.rmtree(Path(eval_controller.tmp_train_dir_path), ignore_errors=True)
    shutil.copytree((Path(__file__).parents[1] / 'fixtures' / "local_data" / 'tmp_train_dir' ), Path(eval_controller.tmp_train_dir_path), dirs_exist_ok=True)
    shutil.rmtree(Path(eval_controller.tmp_train_dir_path).parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(Path(eval_controller.tmp_train_dir_path).parent / 'tmp_dir', ignore_errors=True)
    eval_controller = TrainingController()

    assert len(eval_controller.state['models']) == 0, "There should be no models in the state"

    returned_model_without_train = eval_controller.find_or_train_model(one_experiment_config_fixture, assembly_configs, train_missing=False)
    assert len(eval_controller.state['models']) == 1, "There should be one in the state which is loaded from disk"
    assert 'hash' in returned_model_without_train, "The returned model should have a hash"

    # Change relevant model config
    one_experiment_config_fixture['model_config']['imgsz'] = 123
    returned_model = eval_controller.find_or_train_model(one_experiment_config_fixture, assembly_configs, train_missing=True)
    assert 'hash' in returned_model, "The returned model should have a hash"
    assert returned_model['hash'] in [one_model['hash'] for one_model in eval_controller.state[
        'models']], "The returned model should be in the state"
    assert len(eval_controller.state['models']) == 2

    # Change irrelevant model config
    one_experiment_config_fixture['model_config']['name'] = 'yolov8_changed_name'
    returned_model_changed_name = eval_controller.find_or_train_model(one_experiment_config_fixture, assembly_configs, train_missing=True)
    assert len(eval_controller.state['models']) == 2, "There should still be only two models in the state"
    assert 'hash' in returned_model_changed_name, "The returned model should have a hash"
    assert returned_model_changed_name['hash'] in [one_model['hash'] for one_model in eval_controller.state[
        'models']], "The returned model should be in the state"
    assert Path(returned_model_changed_name['data']["weights_path"]).exists()

    # Change relevant assembly config
    assembly_config2 = assembly_configs
    assembly_config2['dataset_config'][1]["train_split_rate"] = 0.3
    returned_model3 = eval_controller.find_or_train_model(one_experiment_config_fixture, assembly_config2, train_missing=True)
    assert returned_model3['hash'] in [one_model['hash'] for one_model in eval_controller.state[
        'models']], "The returned model should be in the state"
    assert len(eval_controller.state['models']) == 3, "There should be two models in the state"