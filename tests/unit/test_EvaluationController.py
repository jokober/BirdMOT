import json
import shutil
from copy import deepcopy
from pathlib import Path

from BirdMOT.data.DatasetCreator import DatasetCreator
from BirdMOT.detection.evaluate import EvaluationController
from fixtures.fixtures import assembly_configs, sliced_dataset_configs, one_experiment_config_fixture


def test_find_or_create_prediction(one_experiment_config_fixture, assembly_configs):
    eval_controller = EvaluationController()
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_dir', ignore_errors=True)

    assert len(eval_controller.state['predictions']) == 0, "There should be no predictions in the state"

    returned_prediction = eval_controller.find_or_create_prediction(one_experiment_config_fixture, assembly_configs, device=0)
    assert 'hash' in returned_prediction, "The returned prediction should have a hash"
    assert returned_prediction['hash'] in [one_prediction['hash'] for one_prediction in eval_controller.state[
        'predictions']], "The returned prediction should be in the state"
    assert len(eval_controller.state['predictions']) == 1
    assert Path(returned_prediction['data']["export_dir"]).exists()

    returned_prediction2 = eval_controller.find_or_create_prediction(one_experiment_config_fixture, assembly_configs, device=0)
    assert len(eval_controller.state['predictions']) == 1, "There should still be only one prediction in the state"

    assembly_config2 = assembly_configs
    assembly_config2['dataset_config'][1]["train_split_rate"] = 0.3

    returned_prediction3 = eval_controller.find_or_create_prediction(one_experiment_config_fixture, assembly_config2, device=0)
    assert returned_prediction3['hash'] in [one_prediction['hash'] for one_prediction in eval_controller.state[
        'predictions']], "The returned prediction should be in the state"
    assert len(eval_controller.state['predictions']) == 2, "There should be two predictions in the state"


def test_find_or_create_evaluations(one_experiment_config_fixture, assembly_configs):
    eval_controller = EvaluationController()
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_train_dir' / 'tmp_train_state.pkl', ignore_errors=True)
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_eval_dir', ignore_errors=True)
    shutil.rmtree(Path(eval_controller.tmp_eval_dir_path).parent / 'tmp_dir', ignore_errors=True)
    eval_controller = EvaluationController()

    # Check if evaluations after initializations are empty
    assert len(eval_controller.state['evaluations']) == 0, "There should be no evaluations in the state"

    # Create a evaluation
    returned_evaluation = eval_controller.find_or_create_evaluation(one_experiment_config_fixture, assembly_configs, device='0', train_missing=True)
    assert 'hash' in returned_evaluation, "The returned sliced dataset should have a hash"
    assert returned_evaluation['hash'] in [one_evaluation['hash'] for one_evaluation in
                                               eval_controller.state[
                                                   'evaluations']], "The returned evaluation should be in the state"
    assert len(eval_controller.state['evaluations']) == 1

    # Create a evaluation with the same config and make sure it is not created again
    returned_evaluation2 = eval_controller.find_or_create_evaluation(one_experiment_config_fixture, assembly_configs, device='0', train_missing=True)
    assert len(
        [eval_controller.state['evaluations'] == 1]), "There should still be only one evaluation in the state"

    # Change relevant evaluation data and make sure a new evaluation is created
    one_experiment_config2 = one_experiment_config_fixture
    one_experiment_config2['evaluation_config']['iou_thrs'] =0.6
    returned_evaluation3 = eval_controller.find_or_create_evaluation(one_experiment_config2, assembly_configs, device='0', train_missing=True)
    assert returned_evaluation3['hash'] in [one_evaluation['hash'] for one_evaluation in
                                                eval_controller.state[
                                                    'evaluations']], "The returned evaluation should be in the state"
    assert len(eval_controller.state['evaluations']) == 2, "There should be two evaluations in the state"

    # Change relevant evaluation data and make sure a new evaluation is created
    one_experiment_config3 = one_experiment_config_fixture
    one_experiment_config3['sahi_prediction_params']['postprocess_type'] ='NMS'
    returned_evaluation4 = eval_controller.find_or_create_evaluation(one_experiment_config3, assembly_configs, device='0', train_missing=True)
    assert returned_evaluation4['hash'] in [one_evaluation['hash'] for one_evaluation in
                                                eval_controller.state[
                                                    'evaluations']], "The returned evaluation should be in the state"
    assert len(eval_controller.state['evaluations']) == 3, "There should be three evaluations in the state"
    assert returned_evaluation4['sahi_prediction_params']['postprocess_type'] == 'NMS'

    # Change irrelevant evaluation data and make sure a new evaluation is created
    one_experiment_config4 = deepcopy(one_experiment_config_fixture)
    one_experiment_config4['model_config']['name'] = "another name 2"
    returned_evaluation5 = eval_controller.find_or_create_evaluation(one_experiment_config4, assembly_configs, device='0', train_missing=True)
    assert returned_evaluation5['hash'] in [one_evaluation['hash'] for one_evaluation in
                                                eval_controller.state[
                                                    'evaluations']], "The returned evaluation should be in the state"
    assert len(eval_controller.state['evaluations']) == 3, "There should be two evaluations in the state"

    assert returned_evaluation['data']['analysis_results']
    assert returned_evaluation['data']['eval_results']
    assert returned_evaluation['data']['eval_results']